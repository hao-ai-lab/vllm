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

class PrefillScheduler(Scheduler):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.waiting_kv_transfer_queue = {}
    
    def add_request(self, request: Request) -> None:
        request.sampling_params.max_tokens = 1 
        request.sampling_params.min_tokens = 1 
        super().add_request(request)

    def _free_request(self, request: Request) -> None:
        assert request.is_finished()
        # self.kv_cache_manager.free(request)
        # self.encoder_cache_manager.free(request)
        self.running_reqs_data.pop(request.request_id, None)
        del self.requests[request.request_id]
        self.finished_req_ids.add(request.request_id)

        self.waiting_kv_transfer_queue[request.request_id] = request

    def _free_request_after_kv_transfer(self, request: Request) -> None:
        self.kv_cache_manager.free(request)
        self.encoder_cache_manager.free(request)
        self.running_reqs_data.pop(request.request_id, None)
        del self.waiting_kv_transfer_queue[request.request_id]


def create_test_prompts() -> List[Tuple[str, SamplingParams]]:
    """Create a list of test prompts with their sampling parameters."""
    return [
        ("A robot may not injure a human being",
         SamplingParams(temperature=0.0, logprobs=1, prompt_logprobs=1,
                        min_tokens=128, max_tokens=128)),
        ("To be or not to be,",
         SamplingParams(temperature=0.0, presence_penalty=0.2,
                        min_tokens=128, max_tokens=128)),
    ]



async def consume_stream(stream: AsyncGenerator[RequestOutput, None]):
    prev_ts = time.time()
    tpot = []
    ttft = -1
    last_output = None
    idx = 0
    async for output in stream:
        cur_ts = time.time()
        interval = cur_ts - prev_ts
        prev_ts = cur_ts
        last_output = output
        # if output.is_finished():
        #     break
        if idx == 0:
            ttft = interval
        else:
            tpot.append(interval)
        pass
        idx += 1
    
    return dict(
        # tpot=tpot,
        # ttft=ttft,
        # output=last_output,
        request_id=last_output.request_id,
        prompt=last_output.prompt,
        text=last_output.outputs[0].text,
    )



async def main():
    args = AsyncEngineArgs(
        model="facebook/opt-125m",
        enforce_eager=True,
    )

    llm = AsyncLLM.from_engine_args(args)

    prompt_and_params = create_test_prompts()
    streams: List[AsyncGenerator[RequestOutput, None]] = []
    for req_id, (prompt, params) in enumerate(prompt_and_params):
        stream = llm.generate(prompt, params, str(req_id))
        streams.append(consume_stream(stream))

    results = await asyncio.gather(*streams)

    for result in results:
        print(result)



    
if __name__ == "__main__":
    asyncio.run(main())