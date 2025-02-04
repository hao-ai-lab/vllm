"""
torchrun --nproc_per_node=4 gloo_comm.py
"""

import os
import torch
import torch.distributed as dist

from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

def init_process_groups(backend="gloo"):
    """Initialize the default process group (world) using the given backend."""
    # Typically you rely on environment variables set by torchrun or mpirun:
    #   MASTER_ADDR, MASTER_PORT, RANK, WORLD_SIZE
    # Here, just do the standard initialization. 
    dist.init_process_group(backend=backend)

def main():
    # Initialize a GLOO-based group for rank discovery (non-NCCL).
    init_process_groups(backend="gloo")

    world_size = dist.get_world_size()
    world_rank = dist.get_rank()
    print(f"[Rank {world_rank}] world_size: {world_size}")
    # For this example, we expect exactly 4 ranks
    assert world_size == 4, "This example expects exactly 4 ranks."

    # ---------------------------------------------------------
    # 1) Create two 'subgroups' from the default process group.
    #    In this example, both subgroups contain all ranks [0..3]
    #    but you get two *independent* NCCL communicators.
    #    You could also create subgroups with different subsets.
    # ---------------------------------------------------------
    # Create a list of all ranks
    all_ranks = list(range(world_size))

    group1 = dist.new_group(ranks=all_ranks, backend="gloo")
    group2 = dist.new_group(ranks=all_ranks, backend="gloo")

    # ---------------------------------------------------------
    # 2) Create a separate PyNcclCommunicator for each subgroup.
    # ---------------------------------------------------------
    # Typically, you'd bind each rank to "cuda:<local_rank>" or 
    # similar, but let's just do world_rank -> device here:
    device = torch.device(f"cuda:{world_rank % torch.cuda.device_count()}")

    comm1 = PyNcclCommunicator(group=group1, device=device)
    comm2 = PyNcclCommunicator(group=group2, device=device)

    # ---------------------------------------------------------
    # 3) Verify that both communicators work by running an
    #    all_reduce on each.
    # ---------------------------------------------------------
    # Prepare some test tensors on each rank
    # (distinguish the data for clarity).
    tensor1 = torch.tensor([world_rank], device=device, dtype=torch.float32)
    tensor2 = torch.tensor([10 + world_rank], device=device, dtype=torch.float32)

    # Run a collective on communicator 1
    if not comm1.disabled:
        comm1.all_reduce(tensor1)
    else:
        print(f"[Rank {world_rank}] comm1 is disabled (world_size=1 or no NCCL).")

    # Run a collective on communicator 2
    if not comm2.disabled:
        comm2.all_reduce(tensor2)
    else:
        print(f"[Rank {world_rank}] comm2 is disabled (world_size=1 or no NCCL).")

    # Synchronize to ensure collectives are done
    torch.cuda.synchronize(device)

    # Print out final results from each communicator
    print(f"[Rank {world_rank}] "
          f"After comm1.all_reduce, tensor1={tensor1.item()} | "
          f"After comm2.all_reduce, tensor2={tensor2.item()}")

    # Clean up
    dist.destroy_process_group(group1)
    dist.destroy_process_group(group2)
    dist.destroy_process_group()  # destroy default (world) group

if __name__ == "__main__":
    main()