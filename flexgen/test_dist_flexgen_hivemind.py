"""Complete sentences with FlexGen and OPT models."""
import argparse
import pytest

from flexgen.dist_flex_opt import *
from flexgen.opt_config import get_opt_config
from transformers import AutoTokenizer

UID_DELIMITER = "."
INITIAL_PEERS = [
    # IPv4 DNS addresses
    "ec2-54-177-237-94.us-west-1.compute.amazonaws.com",
    "ec2-52-53-152-100.us-west-1.compute.amazonaws.com",
    # Reserved IPs
    "/ip4/54.177.237.94/",
    "/ip4/52.53.152.100/"]

def main(args):
    # Prompts
    prompts = [
        "Question: Where were the 2004 Olympics held?\n"
        "Answer: Athens, Greece\n"
        "Question: What is the longest river on the earth?\n"
        "Answer:",

        "Extract the airport codes from this text.\n"
        "Text: \"I want a flight from New York to San Francisco.\"\n"
        "Airport codes: JFK, SFO.\n"
        "Text: \"I want you to book a flight from Phoenix to Las Vegas.\"\n"
        "Airport codes:",
    ]

    # Initialize environment
    # env = ExecutionEnv.create(args.offload_dir)

    # Offloading policy
    # policy = Policy(args.gpu_batch_size, args.num_gpu_batches,
    #                 args.percent[0], args.percent[1],
    #                 args.percent[2], args.percent[3],
    #                 args.percent[4], args.percent[5],
    #                 args.overlap, args.sep_layer, args.pin_weight,
    #                 args.cpu_cache_compute, args.attn_sparsity,
    #                 args.compress_weight,
    #                 CompressionConfig(num_bits=4, group_size=64,
    #                                   group_dim=0, symmetric=False),
    #                 args.compress_cache,
    #                 CompressionConfig(num_bits=4, group_size=64,
    #                                   group_dim=2, symmetric=False))

    policy = Policy(1, 1,
                    args.percent[0], args.percent[1],
                    args.percent[2], args.percent[3],
                    args.percent[4], args.percent[5],
                    overlap=True, sep_layer=True, pin_weight=args.pin_weight,
                    cpu_cache_compute=args.cpu_cache_compute, attn_sparsity=1.0,
                    compress_weight=args.compress_weight,
                    comp_weight_config=CompressionConfig(
                        num_bits=4, group_size=64,
                        group_dim=0, symmetric=False),
                    compress_cache=args.compress_cache,
                    comp_cache_config=CompressionConfig(
                        num_bits=4, group_size=64,
                        group_dim=2, symmetric=False))

    # Model
    print("Initialize...")
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b", padding_side="left")
    tokenizer.add_bos_token = False
    stop = tokenizer("\n").input_ids[0]
    # print("Stop is %d\n "%stop)

    # model = DistOptLM(args.model, env, args.path, policy)
    # from dist_flex_opt.py
    num_inner_iterations = args.num_inner_iterations

    gpu = TorchDevice(f"cuda:{args.local_rank}")
    cpu = TorchDevice("cpu")
    disk = TorchDisk(args.offload_dir, None, args.local_rank)
    env = ExecutionEnv(gpu=gpu, cpu=cpu, disk=disk, mixed=TorchMixedDevice([gpu, cpu, disk]))
    TorchTensor.name_count = count(start=args.rank, step=2)
    model = DistOptLM(get_opt_config(args.model), env, args.path, policy, args.rank,
                      2, args.comm_device, num_inner_iterations=num_inner_iterations)

    # Generate
    print("Generate...")
    inputs = tokenizer(prompts, padding="max_length", max_length=128)
    # print(inputs)
    print("Start Generating...")

    output_ids = model.generate(
        inputs.input_ids,
        do_sample=True,
        temperature=0.7,
        max_new_tokens=32,
        stop=stop)
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    print("Outputs:\n" + 70 * '-')
    for i in [0, len(outputs)-1]:
        print(f"{i}: {outputs[i]}")
        print("-" * 70)

    # Shutdown
    print("Shutdown...")
    env.close_copy_threads()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="facebook/opt-1.3b",
        help="The model name.")
    parser.add_argument("--path", type=str, default="~/opt_weights",
        help="The path to the model weights. If there are no cached weights, "
             "FlexGen will automatically download them from HuggingFace.")
    parser.add_argument("--offload-dir", type=str, default="~/flexgen_offload_dir",
        help="The directory to offload tensors. ")
    parser.add_argument("--percent", nargs="+", type=int,
        default=[0, 100, 0, 100, 0, 100],
        help="Six numbers. They are "
         "the percentage of weight on GPU, "
         "the percentage of weight on CPU, "
         "the percentage of attention cache on GPU, "
         "the percentage of attention cache on CPU, "
         "the percentage of activations on GPU, "
         "the percentage of activations on CPU")
    parser.add_argument("--pin-weight", type=str2bool, nargs="?",
        const=True, default=True)
    parser.add_argument("--cpu-cache-compute", action="store_true")
    parser.add_argument("--compress-weight", action="store_true",
        help="Whether to compress weight.")
    parser.add_argument("--compress-cache", action="store_true",
        help="Whether to compress cache.")
    parser.add_argument("--comm-device", type=str, default="cpu")
    parser.add_argument("--pipeline_rank")
    parser.add_argument('--num-inner-iterations', metavar='I', type=int, default=None)
    
    args = parser.parse_args()

    args.world_size = 1
    args.rank = 0
    args.local_rank = 0
    
    assert len(args.percent) == 6

    main(args)