from __future__ import annotations

import inspect

from app.schemas import ChatRequest, ChatResponse
from app.constants import (
    ENABLE_CHUNKED_PREFILL,
    ENABLE_PREFIX_CACHING,
    GPU_MEMORY_UTILIZATION,
    KV_CACHE_DTYPE,
    MAX_MODEL_LENGTH,
    MAX_NUM_BATCHED_TOKENS,
    MAX_NUM_SEQS,
    MODEL_NAME,
    NUM_SCHEDULER_STEPS,
    RESPONSE_CACHE_MAX_SIZE,
)
from app.response_cache import CachedResponse, ResponseCache
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid


class ChatEngine:
    """
    This engine uses vLLM's AsyncLLMEngine for high-performance inference.
    """
    def __init__(self):
        self.model_name = MODEL_NAME
        self.engine: AsyncLLMEngine | None = None
        self.tokenizer = None
        self.is_ready = False
        self.cache = ResponseCache(max_size=RESPONSE_CACHE_MAX_SIZE)

    async def initialize(self):
        if self.is_ready:
            return
            
        print(f"Initializing vLLM with model: {self.model_name}...")

        print(f"  KV-cache dtype       : {KV_CACHE_DTYPE}")
        print(f"  Chunked prefill      : {ENABLE_CHUNKED_PREFILL}")
        print(f"  GPU memory util      : {GPU_MEMORY_UTILIZATION}")
        print(f"  Max num sequences    : {MAX_NUM_SEQS}")
        print(f"  Prefix caching (APC) : {ENABLE_PREFIX_CACHING}")
        print(f"  Scheduler steps      : {NUM_SCHEDULER_STEPS}")
        print(f"  Max batched tokens   : {MAX_NUM_BATCHED_TOKENS}")
        print(f"  Max model length     : {MAX_MODEL_LENGTH}")

        engine_kwargs = dict(
            model=self.model_name,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            max_model_len=MAX_MODEL_LENGTH,
            trust_remote_code=True,
            kv_cache_dtype=KV_CACHE_DTYPE,
            enable_chunked_prefill=ENABLE_CHUNKED_PREFILL,
            max_num_seqs=MAX_NUM_SEQS,
            enable_prefix_caching=ENABLE_PREFIX_CACHING,
            num_scheduler_steps=NUM_SCHEDULER_STEPS,
            max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
        )

        valid_params = set(inspect.signature(AsyncEngineArgs.__init__).parameters.keys())
        unsupported = [k for k in engine_kwargs if k not in valid_params]
        for key in unsupported:
            print(f"  WARNING: dropping unsupported arg '{key}' (not in this vLLM version)")
            del engine_kwargs[key]

        engine_args = AsyncEngineArgs(**engine_kwargs)

        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        self.tokenizer = await self.engine.get_tokenizer()

        self.is_ready = True
        print("vLLM Engine initialized and ready.")

    async def generate(self, request: ChatRequest) -> ChatResponse:
        if not self.is_ready:
            raise Exception("Engine is still initializing. Please try again later.")

        messages_dicts = [{"role": m.role, "content": m.content} for m in request.messages]
        prompt = self.tokenizer.apply_chat_template(
            messages_dicts, 
            tokenize=False, 
            add_generation_prompt=True
        )

        is_deterministic = request.temperature == 0 or request.temperature is None
        cache_key = None
        if is_deterministic:
            cache_key = ResponseCache.make_key(prompt, request.temperature, request.max_tokens)
            cached = self.cache.get(cache_key)
            if cached is not None:
                return ChatResponse(output=cached.output, logprobs=cached.logprobs)

        sampling_params = SamplingParams(
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            logprobs=1
        )
        results_generator = self.engine.generate(
            prompt,
            sampling_params,
            random_uuid(), # Unique request ID for vLLM tracking
        )

        final_output = None
        async for request_output in results_generator:
            final_output = request_output

        if final_output is None:
            raise Exception("No output generated")

        text_output = final_output.outputs[0].text
        
        output_data = final_output.outputs[0]
        if output_data.logprobs is None:
            raise RuntimeError("logprobs are missing from vLLM output")
        
        if not output_data.token_ids:
            raise RuntimeError("token_ids is empty, cannot provide logprobs")

        logprobs: list[float] = []
        for i, token_id in enumerate(output_data.token_ids):
            if i < len(output_data.logprobs):
                step_logprobs = output_data.logprobs[i]
                if token_id in step_logprobs:
                    logprobs.append(step_logprobs[token_id].logprob)
                else:
                    raise RuntimeError(f"Token ID {token_id} not found in logprobs at step {i}")

        if is_deterministic and cache_key is not None:
            self.cache.put(cache_key, CachedResponse(output=text_output, logprobs=logprobs))

        return ChatResponse(output=text_output, logprobs=logprobs)
