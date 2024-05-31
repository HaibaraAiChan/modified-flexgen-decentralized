import asyncio
import itertools
from collections import deque
from typing import Iterable, List, Optional, Sequence, Tuple, Any, Dict

import torch
from hivemind import MSGPackSerializer
from hivemind.moe.client.remote_expert_worker import RemoteExpertWorker
from hivemind.moe.server.connection_handler import ConnectionHandler
from hivemind.utils.logging import get_logger
from hivemind.p2p import StubBase
from hivemind.p2p.p2p_daemon_bindings.control import DEFAULT_MAX_MSG_SIZE, MAX_UNARY_PAYLOAD_SIZE
from hivemind.proto import runtime_pb2
from hivemind.utils.streaming import split_for_streaming
from hivemind.utils.asyncio import aiter_with_timeout, iter_as_aiter
from hivemind.compression.serialization import deserialize_tensor_stream, deserialize_torch_tensor
from hivemind import nested_compare, nested_flatten, nested_pack, serialize_torch_tensor

from flexgen.client_manager import SequenceManagerConfig

#
from backend import DistOptLMBackend

logger = get_logger(__name__)
RPCInfo = Dict[str, Any]

## from Petals sequential_autograd.py
## Implementation: to achieve the generation using a sequence of servers
async def sequential_generation(
    inputs: torch.Tensor,
    prompts: torch.Tensor,
    sequence_manager: client_manager,
    start_index: int = 0,
    end_index: Optional[int] = None,):

    assert isinstance(inputs, torch.Tensor) and inputs.ndim == 3, f"{type(inputs)}: {inputs.ndim}"

    inputs_device = inputs.device
    inputs_dtype = inputs.dtype
    inputs = inputs.cpu()
    prompts = prompts.cpu()

    end_index = end_index if end_index is not None else len(sequence_manager.block_uids)
    # assert start_index >= 0 and end_index <= len(sequence_manager.block_uids)
    # assert is_dummy(prompts) or len(prompts) == len(
    #     sequence_manager.block_uids
    # )  # should be n_layers - 1 but add extra prompts for convenience

    sequences = deque()
    intermediate_inputs = []
    done_sequences = []

    block_idx = start_index
    while block_idx < end_index:
        for attempt_no in itertools.count():
            logger.debug(f"Forward: block {block_idx}, attempt {attempt_no}")
            span = None
            try:
                if not sequences or attempt_no >= 1:
                    sequences = deque(sequence_manager.make_sequence(block_idx, end_index, mode="max_throughput"))
                    # make_sequence() could return a longer sequence
                    sequences[-1].end = min(sequences[-1].end, end_index)
                    logger.debug(f"Found path from block {block_idx} to {end_index} via {len(sequences)} servers")

                span = sequences.popleft()

                stub = ConnectionHandler.get_stub(sequence_manager.state.p2p, span.peer_id) #
                inputs_and_prompts = [inputs, prompts[span.start : span.end]]

                span_uids = " ".join(sequence_manager.block_uids[span.start : span.end])
                metadata = sequence_manager.get_request_metadata("rpc_forward", span_uids, *inputs_and_prompts)
                (outputs,) = await run_remote_generation(span_uids, stub, sequence_manager.rpc_info, 
                *inputs_and_prompts, config=sequence_manager.config, metadata=MSGPackSerializer.dumps(metadata),)

                assert isinstance(outputs, torch.Tensor)
                assert outputs.shape == inputs.shape, f"Expected output {inputs.shape}, got {outputs.shape}"

                # Save intermediate inputs and subsequences if the forward is already done for them
                intermediate_inputs.append(inputs)
                done_sequences.append(span)

                inputs = outputs
                block_idx = span.end
                sequence_manager.on_request_success(span.peer_id)
                break
            except Exception as e:
                sequence_manager.on_request_failure(span.peer_id if span is not None else None)
                if attempt_no + 1 == sequence_manager.config.max_retries:
                    raise
                delay = sequence_manager.get_retry_delay(attempt_no)
                logger.warning(
                    f"Caught exception when running forward via {span} (retry in {delay:.0f} sec): {repr(e)}"
                )
                # maybe_log_traceback(e)
                await asyncio.sleep(delay)

    outputs = inputs.to(device=inputs_device, dtype=inputs_dtype)
    intermediate_inputs = [tensor.to(device=inputs_device, dtype=inputs_dtype) for tensor in intermediate_inputs]
    return outputs, intermediate_inputs, done_sequences

## from Petals remote_forward_backward.py
## If one server wants to run the generation, use this function
async def run_remote_generation(uid: ModuleUID,
    stub: StubBase,
    rpc_info: RPCInfo,
    *inputs: torch.Tensor,
    config: SequenceManagerConfig,
    metadata: Optional[bytes] = None,
    **kwargs,):
    assert len(kwargs) == len(rpc_info["keyword_names"]), f"Keyword args should be {rpc_info['keyword_names']}"
    kwargs = {key: kwargs[key] for key in rpc_info["keyword_names"]}

    # Note: we put keyword arguments in the same order as on a server to prevent f(a=1, b=2) != f(b=2, a=1) errors
    forward_inputs = (inputs, kwargs)

    # Modify forward_schema to support prompts
    args_schema, kwargs_schema = rpc_info["forward_schema"]
    # TODO: rm this assert when support arbitrary number of input tensors
    assert len(args_schema) == 1 and len(inputs) == 2
    forward_schema_with_prompts = (tuple(args_schema * len(inputs)), kwargs_schema)

    if not nested_compare(forward_inputs, forward_schema_with_prompts):
        raise TypeError(f"Inputs do not match expert input schema. Did you pass the right number of parameters?")

    forward_inputs = nested_flatten(forward_inputs)
    inputs = tuple(tensor.cpu().detach() for tensor in forward_inputs)

    # Asynchronous serialization
    loop = asyncio.get_running_loop()
    serialized_tensors = await asyncio.gather(
        *(
            loop.run_in_executor(None, serialize_torch_tensor, tensor.to(proto.dtype), proto.compression)
            for tensor, proto in zip(inputs, nested_flatten(forward_schema_with_prompts))
        )
    )

    size = sum(t.element_size() * t.nelement() for t in inputs)
    forward_fn = _generation_stream if size > MAX_UNARY_PAYLOAD_SIZE // 2 else _generation_unary
    # Hotfix: we use "// 2" since hivemind==1.1.5 serializes bfloat16 tensors in float32, so they take 2x more space
    deserialized_outputs = await forward_fn(uid, serialized_tensors, stub, config, metadata=metadata, **kwargs)
    return nested_pack(deserialized_outputs, structure=rpc_info["outputs_schema"])


## from Petal remote_forward_backward.py
## _forward_fn + _forward_stream
# def forward_generation_fn():
#     return
async def _generation_stream(
    uid: str, serialized_tensors: Iterable[runtime_pb2.Tensor], stub, config: SequenceManagerConfig, **kwargs
) -> List[torch.Tensor]:
    parts = (
        runtime_pb2.ExpertRequest(uid=uid, tensors=[part], **kwargs)
        for tensor in serialized_tensors
        for part in split_for_streaming(tensor, DEFAULT_MAX_MSG_SIZE)
    )
    outputs = await asyncio.wait_for(stub.rpc_generate_stream(iter_as_aiter(parts)), config.connect_timeout)
    outputs = aiter_with_timeout(outputs, config.request_timeout)
    return await deserialize_tensor_stream(msg.tensors async for msg in outputs)


async def _generation_unary(
    uid: str, serialized_tensors: Iterable[runtime_pb2.Tensor], stub, config: SequenceManagerConfig, **kwargs
) -> List[torch.Tensor]:
    outputs: runtime_pb2.ExpertResponse = await stub.rpc_generate(
        runtime_pb2.ExpertRequest(uid=uid, tensors=list(serialized_tensors), **kwargs),
        timeout=config.request_timeout,
    )
    return [deserialize_torch_tensor(t) for t in outputs.tensors]

