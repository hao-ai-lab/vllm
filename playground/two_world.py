


import asyncio
import argparse
import time
from typing import List, Tuple
import torch
from vllm import EngineArgs, SamplingParams

from vllm.config import ModelConfig, VllmConfig
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.core.scheduler import Scheduler
from vllm.v1.request import Request
from vllm.v1.engine.async_llm import AsyncLLM
from typing import AsyncGenerator
from vllm.outputs import RequestOutput

from vllm.v1.executor.multiproc_executor import WorkerProc
from vllm.v1.engine.core_client import InprocClient, EngineCoreClient
from vllm.v1.executor.abstract import UniProcExecutor
from vllm.v1.engine.core import EngineCoreRequest

from vllm.v1.engine.processor import Processor
from vllm.transformers_utils.tokenizer_group import init_tokenizer_from_configs

from vllm.v1.engine import EngineCoreOutputs, EngineCoreOutput
from vllm.v1.executor.multiproc_executor import MultiprocExecutor
from collections import defaultdict

from vllm.utils import (get_distributed_init_method, get_mp_context,
                        get_open_port, get_open_zmq_ipc_path, zmq_socket_ctx)

from vllm.v1.executor.multiproc_executor import MessageQueue

def get_tokenizer(vllm_config):
    tokenizer = init_tokenizer_from_configs(
        model_config=vllm_config.model_config,
        scheduler_config=vllm_config.scheduler_config,
        parallel_config=vllm_config.parallel_config,
        lora_config=vllm_config.lora_config
    )
    
    tokenizer.ping()
    return tokenizer


def main():
    args = AsyncEngineArgs(
        # model="facebook/opt-125m",
        model="meta-llama/Llama-3.1-8B-Instruct",
        enforce_eager=True,
        tensor_parallel_size=4,
    )
    vllm_config = args.create_engine_config()
    tokenizer = get_tokenizer(vllm_config)

    distributed_init_method = get_distributed_init_method("127.0.0.1", get_open_port())
    world_size = 4
    rpc_broadcast_mq = MessageQueue(world_size, world_size)
    scheduler_output_handle = rpc_broadcast_mq.export_handle()

    workers = []
    for rank in range(4):
        local_rank = rank
        worker = WorkerProc.make_worker_process(
            vllm_config, 
            local_rank, # local_rank
            rank, # global_rank
            distributed_init_method,
            scheduler_output_handle
        )
        workers.append(worker)

    rpc_broadcast_mq.wait_until_ready()
    for w in workers:
        w.worker_response_mq.wait_until_ready()

    print("All workers are ready")



def main_2():
    args = AsyncEngineArgs(
        # model="facebook/opt-125m",
        model="meta-llama/Llama-3.1-8B-Instruct",
        enforce_eager=True,
        tensor_parallel_size=4,
    )
    distserve_config = DistServeConfig(
        prefill_pp=1,
        prefill_tp=2,
        decode_pp=2,
        decode_tp=1
    )
    vllm_config: VllmConfig = args.create_engine_config()
    vllm_config.distserve_config = distserve_config
    breakpoint()
    tokenizer = get_tokenizer(vllm_config)

    distributed_init_method = get_distributed_init_method("127.0.0.1", get_open_port())
    world_size = 4
    rpc_broadcast_mq = MessageQueue(world_size, world_size)
    scheduler_output_handle = rpc_broadcast_mq.export_handle()

    workers = []
    for rank in range(world_size):
        local_rank = rank
        worker = WorkerProc.make_worker_process(
            vllm_config, 
            local_rank, # local_rank
            rank, # global_rank
            distributed_init_method,
            scheduler_output_handle
        )
        workers.append(worker)

    rpc_broadcast_mq.wait_until_ready()
    for w in workers:
        w.worker_response_mq.wait_until_ready()

    print("All workers are ready")


def main_3():
    """Making the MultiProcExector (and similar) logic to work with DistServeConfig."""
    args = AsyncEngineArgs(
        # model="facebook/opt-125m",
        model="meta-llama/Llama-3.1-8B-Instruct",
        enforce_eager=True,
        # tensor_parallel_size=1,
    )
    distserve_config = DistServeConfig(
        prefill_pp=1,
        prefill_tp=2,
        decode_pp=1,
        decode_tp=1
    )
    vllm_config: VllmConfig = args.create_engine_config()
    vllm_config.distserve_config = distserve_config
    # vllm_config.parallel_config.world_size = distserve_config.world_size
    tokenizer = get_tokenizer(vllm_config)

    distributed_init_method = get_distributed_init_method("127.0.0.1", get_open_port())
    world_size = distserve_config.world_size
    rpc_broadcast_mq = MessageQueue(world_size, world_size)
    scheduler_output_handle = rpc_broadcast_mq.export_handle()

    workers = []
    for rank in range(world_size):
        local_rank = rank
        worker = WorkerProc.make_worker_process(
            vllm_config, 
            local_rank, # local_rank
            rank, # global_rank
            distributed_init_method,
            scheduler_output_handle
        )
        workers.append(worker)

    rpc_broadcast_mq.wait_until_ready()
    for w in workers:
        w.worker_response_mq.wait_until_ready()

    print("All workers are ready")


from vllm.config import DistServeConfig

def test_distserve_config():
    distserve_config = DistServeConfig(
        prefill_pp=2,
        prefill_tp=4,
        decode_pp=4,
        decode_tp=2
    )
    print(f"world_size: {distserve_config.world_size}")
    print(f"prefill_world_size: {distserve_config.prefill_world_size} = {distserve_config.prefill_pp} * {distserve_config.prefill_tp}")
    print(f"decode_world_size: {distserve_config.decode_world_size} = {distserve_config.decode_pp} * {distserve_config.decode_tp}")
    prefill_world_placement = distserve_config.get_prefill_world_placement()
    print(f"prefill_world_placement: {prefill_world_placement}")
    print(f"prefill_world_placement_pp: {list(zip(*prefill_world_placement))}")
    decode_world_placement = distserve_config.get_decode_world_placement()
    print(f"decode_world_placement: {decode_world_placement}")
    print(f"decode_world_placement_pp: {list(zip(*decode_world_placement))}")
    

if __name__ == "__main__":
    # main()
    # main_2()
    main_3()
    # test_distserve_config()

