from __future__ import annotations

from collections import Counter
from itertools import chain
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import hivemind
import torch
from hivemind import BatchTensorDescriptor, TensorDescriptor
from hivemind.moe.expert_uid import ExpertUID
from hivemind.moe.server.module_backend import ModuleBackend
from hivemind.utils import get_logger
from tensor_parallel import TensorParallel
#from tensor_parallel.tensor_parallel import PerDeviceTensors
from transformers import PretrainedConfig

from flexgen.dist_flex_opt import DistOptLM, OptLM
from flexgen.flex_opt import (Policy, InputEmbed, OutputEmbed, SelfAttention,
                              MLP, TransformerLayer, OptLM, get_filename,
                              add_parser_arguments, get_test_inputs,
                              DUMMY_WEIGHT)
from flexgen.client_manager import RemoteSequential, RemoteSequenceManager
import msgpack
import msgpack_numpy as m
from flexgen.timer import timers

class DecOptLM(DistOptLM):
    """ Decentralized Opt LM model """
    def __init__(self, config, env, path, policy, 
                 pipeline_rank, num_pipeline_stages, 
                 comm_device, num_inner_iterations=None, async_comm=False, dht: Optional[hivemind.DHT] = None):
        super().__init__(config, env, path, policy, 
                         pipeline_rank, num_pipeline_stages, 
                         comm_device, num_inner_iterations, async_comm)
        
        self.config = config
        self.env = env
        self.path = path
        self.policy = policy
        self.num_gpu_batches = self.policy.num_gpu_batches
        self.pipeline_rank = pipeline_rank
        self.num_pipeline_stages = num_pipeline_stages
        self.num_inner_iterations = num_inner_iterations if num_inner_iterations is not None else num_pipeline_stages
        self.async_comm = async_comm
        if comm_device == "cpu":
            self.comm_device = self.env.cpu
        elif comm_device == "gpu":
            self.comm_device = self.env.gpu
        else:
            raise ValueError(f"Invalid comm_device: {comm_device}")

        layers = []
        if pipeline_rank == 0:
            layers.append(InputEmbed(self.config, self.env, self.policy))
        pipeline_stage_sizes = [config.num_hidden_layers // num_pipeline_stages
                                + int(i < config.num_hidden_layers % num_pipeline_stages)
                                for i in range(num_pipeline_stages)]
        layer_start_ids = [0]
        for stage_size in pipeline_stage_sizes:
            layer_start_ids.append(layer_start_ids[-1] + stage_size)
        for i in range(layer_start_ids[pipeline_rank], layer_start_ids[pipeline_rank + 1]):
            if self.policy.sep_layer:
                layers.append(SelfAttention(self.config, self.env, self.policy, i))
                layers.append(MLP(self.config, self.env, self.policy, i))
            else:
                layers.append(TransformerLayer(self.config, self.env, self.policy, i))
        if pipeline_rank == num_pipeline_stages - 1:
            layers.append(OutputEmbed(self.config, self.env, self.policy))
        self.layers = layers
        self.num_layers = len(layers)

        if self.policy.act_gpu_percent == 100:
            self.act_home = self.env.gpu
        elif self.policy.act_cpu_percent == 100:
            self.act_home = self.env.cpu
        elif self.policy.act_disk_percent == 100:
            self.act_home = self.env.disk
        else:
            raise NotImplementedError()

        # CUDA streams
        self.load_weight_stream = torch.cuda.Stream()
        self.load_cache_stream = torch.cuda.Stream()
        self.store_cache_stream = torch.cuda.Stream()

        self.task = None
        self.init_all_weights()

        self.h = RemoteSequential(config, dht=dht)
        self.dht = dht

    ## connect for loop in dist_flex_opt & send receive
    def generation_step(self):
        self.sending_tag = 0
        self.receiving_tag = 0
        last_sending_job = None

        for b in range(self.num_pipeline_batches // self.num_inner_iterations):
            for i in range(self.execute_gen_len):
                for t in range(self.num_inner_iterations):
                    timer_name = "generate-prompt" if i == 0 else "generate"
                    # timers(timer_name).start()
                    for k in range(self.num_gpu_batches):
                        self.update_attention_mask(b, t, i, k)

                    # if self.num_pipeline_stages > 1:
                    #     self.send_recv_hidden(last_sending_job, (t, i))

                    for j in range(self.num_layers):
                        for k in range(self.num_gpu_batches):
                            self.load_weight(b, t, i, j, k)
                        self.sync()

                        for k in range(self.num_gpu_batches):
                            self.load_cache(t, i, j, k)
                            self.load_hidden(b, t, i, j, k)
                            self.sync()
                            self.compute_layer(t, i, j, k)
                            self.sync()
                            self.store_hidden(b, t, i, j, k)
                            self.store_cache(t, i, j, k)
                            self.sync()

    def send_hidden(self, t, i, j, k, tag=0, async_=False):
        # Suppose we need to send tensors on GPUs
        x = self.hidden[t][i][j][k]
        val = x.pop().move(self.comm_device)
        receiver_rank = (self.pipeline_rank + 1) % self.num_pipeline_stages
        # if async_:
        #     future = dist.isend(val.data, receiver_rank, tag=tag)
        #     return future
        # else:
        #     dist.send(val.data, receiver_rank, tag=tag) 

        # Share your model and optimizer on the DHT
        # self.dht.store('model', val.data, tags=['model'], expiration_time=1) # expiration_time
        print(val.data)
        print(type(val.data))
        future = self.dht.store((t,i,j,k), msgpack.packb(val.data.numpy(), default=m.encode), expiration_time=1)

        return future

    def recv_hidden(self, t, i, j, k, tag=0, async_=False):
        sender_rank = (self.pipeline_rank - 1) % self.num_pipeline_stages
        val_holder = self.hidden[t][i][j][k]
        seq_len = self.task.prompt_len if i == 0 else 1
        shape, dtype = self.layers[j].input_act_shape_and_dtype(
            self.policy.gpu_batch_size, seq_len)
        if val_holder.val is None:
            val_holder.val = self.comm_device.allocate(shape, dtype)
        else:
            val_holder.val = val_holder.val.move(self.comm_device)
        def move_value_callback():
            val_holder.val = val_holder.val.move(self.act_home)
        # if async_:
        #     future = dist.irecv(val_holder.val.data, sender_rank, tag=tag)
        #     return future, move_value_callback
        # else:
        #     dist.recv(val_holder.val.data, sender_rank, tag=tag)
        
        # future = self.dht.get('model', latest=True)
        future = self.dht.get((t,i,j,k), latest=True)
        return future, move_value_callback
    
    def generation_loop_overlap_one_batch(self):
        assert self.num_gpu_batches == 1
        # Prologue
        self.load_weight(0, 0, 0, 0, 0)
        self.sync()
        self.sending_tag = 0
        self.receiving_tag = 0
        last_sending_job = None

        # Generate
        for b in range(self.num_pipeline_batches // self.num_inner_iterations):
            for i in range(self.execute_gen_len):
                for t in range(self.num_inner_iterations):
                    timer_name = "generate-prompt" if i == 0 else "generate"
                    timers(timer_name).start()
                    self.update_attention_mask(b, t, i, 0)

                    if self.num_pipeline_stages > 1:
                        self.send_recv_hidden(last_sending_job, (t, i))

                    for j in range(self.num_layers):
                        self.load_weight(b, t, i, j+1, 0)
                        self.load_cache(t, i, j+1, 0)
                        self.load_hidden(b, t, i, j, 0)
                        self.compute_layer(t, i, j, 0)
                        self.store_cache(t, i, j-1, 0)
                        self.store_hidden(b, t, i, j, 0)
                        self.sync()

                    last_sending_job = (t, i)

                    timers(timer_name).stop()

        if self.num_pipeline_stages > 1:
            self.send_recv_hidden(last_sending_job, None)
            
