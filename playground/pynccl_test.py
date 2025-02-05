"""
torchrun --nproc_per_node=4 pynccl_test.py
"""
import torch
import torch.distributed as dist
from vllm.distributed.device_communicators.pynccl_wrapper import NCCLLibrary, ncclUniqueId

def test_nccl_ids():
    # Initialize process group
    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    # Initialize NCCL library
    nccl = NCCLLibrary()
    
    # Test 1: Original behavior
    if rank == 0:
        print(f"Rank {rank}: Generating unique ID (original behavior)")
        unique_id_1 = nccl.ncclGetUniqueId()
    else:
        print(f"Rank {rank}: Creating empty unique ID")
        unique_id_1 = ncclUniqueId()
    
    # Broadcast the ID using the process group
    tensor_1 = torch.ByteTensor(list(unique_id_1.internal))
    dist.broadcast(tensor_1, src=0)
    print(f"Rank {rank}: Received broadcast ID: {tensor_1[:5].tolist()}")  # Print first 5 bytes
    
    # Test 2: Try generating ID on non-zero rank
    try:
        print(f"Rank {rank}: Attempting to generate unique ID")
        unique_id_2 = nccl.ncclGetUniqueId()
        tensor_2 = torch.ByteTensor(list(unique_id_2.internal))
        print(f"Rank {rank}: Successfully generated unique ID: {tensor_2[:5].tolist()}")
        
        # Broadcast from the current rank
        if rank == 1:
            print(f"Rank {rank}: Broadcasting my generated ID: {tensor_2[:5].tolist()}")
        # dist.broadcast(tensor_2, src=rank)
        dist.broadcast(tensor_2, src=1)
        print(f"Rank {rank}: Broadcast my generated ID: {tensor_2[:5].tolist()}")
    except Exception as e:
        print(f"Rank {rank}: Error generating unique ID: {str(e)}")
    
    dist.destroy_process_group()

if __name__ == "__main__":
    test_nccl_ids()