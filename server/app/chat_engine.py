import inspect
import time
import json
import os
import warnings

from app.constants import *
from app.prompt_analytics import PromptAnalytics
from app.response_cache import CachedResponse, ResponseCache
from app.schemas import ChatRequest, ChatResponse
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid


class ChatEngine:
    def __init__(self) -> None:
        self.model_name: str = MODEL_NAME
        self.engine: AsyncLLMEngine | None = None
        self.tokenizer = None
        self.is_ready: bool = False
        self.analytics = PromptAnalytics()
        self.cache = ResponseCache(max_size=RESPONSE_CACHE_MAX_SIZE)

    async def initialize(self) -> None:
        if self.is_ready:
            return
        
        enable_chunked_prefill = ENABLE_CHUNKED_PREFILL
        if SPECULATIVE_METHOD:
            enable_chunked_prefill = False
            spec_config = {                                                                                                                          
                "method": SPECULATIVE_METHOD,
                "num_speculative_tokens": NUM_SPECULATIVE_TOKENS,
            }
            if SPECULATIVE_MODEL == "ngram":
                spec_config["ngram_prompt_lookup_max"] = NGRAM_PROMPT_LOOKUP_MAX
                spec_config["ngram_prompt_lookup_min"] = NGRAM_PROMPT_LOOKUP_MIN
                
            else:                                                                                                                   
                spec_config["model"] = SPECULATIVE_MODEL

        engine_kwargs: dict = dict(
            model=self.model_name,         
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            max_model_len=MAX_MODEL_LENGTH,
            trust_remote_code=True,
            kv_cache_dtype=KV_CACHE_DTYPE,
            enable_chunked_prefill=enable_chunked_prefill,
            max_num_seqs=MAX_NUM_SEQS,
            enable_prefix_caching=ENABLE_PREFIX_CACHING,
            max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
            speculative_config=spec_config if SPECULATIVE_METHOD else None,
            # attention_backend="FLASHINFER",
        )

        if SPECULATIVE_MODEL:
            engine_kwargs["enable_chunked_prefill"] = False
            engine_kwargs["speculative_model"] = SPECULATIVE_MODEL
            engine_kwargs["num_speculative_tokens"] = NUM_SPECULATIVE_TOKENS
            
            if SPECULATIVE_MODEL == "[ngram]":
                engine_kwargs["ngram_prompt_lookup_max"] = NGRAM_PROMPT_LOOKUP_MAX
                engine_kwargs["ngram_prompt_lookup_min"] = NGRAM_PROMPT_LOOKUP_MIN

                
        valid_params = set(inspect.signature(AsyncEngineArgs.__init__).parameters.keys())
        unsupported = [k for k in engine_kwargs if k not in valid_params]
        for k in unsupported:
            warnings.warn(f"  WARNING: dropping unsupported arg '{k}' (not in this vLLM version)")
            del engine_kwargs[k]

        self.engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(**engine_kwargs))

        # Handle both sync and async get_tokenizer across vLLM versions
        tokenizer = self.engine.get_tokenizer()
        if inspect.isawaitable(tokenizer):
            tokenizer = await tokenizer
        self.tokenizer = tokenizer

        self.is_ready = True
        print("vLLM Engine initialized and ready.")

    async def generate(self, request: ChatRequest) -> ChatResponse:
        t0 = time.perf_counter()

        if not self.is_ready:
            raise Exception("Engine is still initializing. Please try again later.")

        # ── Stage 1-2: Preprocessing ──
        messages_dicts = [{"role": m.role, "content": m.content} for m in request.messages]
        prompt = self.tokenizer.apply_chat_template(
            messages_dicts, tokenize=False, add_generation_prompt=True,
        )

        is_deterministic = request.temperature == 0 or request.temperature is None
        cache_key = ""
        if is_deterministic:
            cache_key = ResponseCache.make_key(prompt, request.temperature, request.max_tokens)
            cached = self.cache.get(cache_key)
            if cached is not None:
                return ChatResponse(output=cached.output, logprobs=cached.logprobs)

        exact_hash, template_hash = self.analytics.get_hashes(prompt)
        t_preprocess = time.perf_counter()

        # ── Stage 3-5: Queue + Prefill + Decode ──
        sampling_params = SamplingParams(
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            logprobs=1,
        )
        request_id = random_uuid()
        results_generator = self.engine.generate(
            prompt, sampling_params, request_id,
        )

        first_token_time = None
        final_output = None
        async for request_output in results_generator:
            if first_token_time is None:
                first_token_time = time.perf_counter()
            final_output = request_output
        t_gen_done = time.perf_counter()

        # ── Stage 6: Response assembly ──
        if final_output is None:
            raise Exception("No output generated")

        output_data = final_output.outputs[0]
        text_output = output_data.text

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

        # ── Stage 7: Metrics + cache ──
        output_token_count = len(output_data.token_ids)
        self.analytics.record_completion(exact_hash, template_hash, output_token_count)

        if is_deterministic and cache_key:
            self.cache.put(cache_key, CachedResponse(output=text_output, logprobs=logprobs))
        t_done = time.perf_counter()
        return ChatResponse(output=text_output, logprobs=logprobs)
