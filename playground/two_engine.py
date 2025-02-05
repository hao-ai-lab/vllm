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

from vllm.v1.engine.core_client import InprocClient, EngineCoreClient
from vllm.v1.executor.abstract import UniProcExecutor
from vllm.v1.engine.core import EngineCoreRequest

from vllm.v1.engine.processor import Processor
from vllm.transformers_utils.tokenizer_group import init_tokenizer_from_configs

from vllm.v1.engine import EngineCoreOutputs, EngineCoreOutput
from vllm.v1.executor.multiproc_executor import MultiprocExecutor
from collections import defaultdict

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
        tensor_parallel_size=2,
    )
    vllm_config = args.create_engine_config()
    tokenizer = get_tokenizer(vllm_config)
    client: InprocClient = EngineCoreClient.make_client(
        multiprocess_mode=False,
        asyncio_mode=False,
        vllm_config=vllm_config,
        # executor_class=UniProcExecutor,
        executor_class=MultiprocExecutor,
    )

    prompt = "Fun fact about Paris: "
    prompt_token_ids = tokenizer.encode(prompt)
    
    
    requests = []
    for i in reversed(range(4)):
        request = EngineCoreRequest(
            request_id=str(i),
            prompt=prompt,
            prompt_token_ids=prompt_token_ids,
            mm_inputs=None,
            mm_hashes=None,
            mm_placeholders=None,
            sampling_params=SamplingParams(),
            eos_token_id=None,
            arrival_time=0.0,
            lora_request=None,
        )
        requests.append(request)
        pass

    new_token_ids = defaultdict(list)

    while True:
        if requests:
            request = requests.pop()
            client.add_request(request)
        
        outputs: EngineCoreOutputs = client.get_output()
        if len(outputs.outputs) == 0:
            break
        for output in outputs.outputs:
            rid = output.request_id
            new_token_ids[rid].extend(output.new_token_ids)

    for rid, token_ids in new_token_ids.items():
        print(f"[Request {rid}] {request.prompt} || {repr(tokenizer.tokenizer.decode(token_ids))}")
    return



if __name__ == "__main__":
    main()