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
        self.template_cache = (
            ResponseCache(max_size=TEMPLATE_RESPONSE_CACHE_MAX_SIZE)
            if ENABLE_TEMPLATE_RESPONSE_CACHING
            else None
        )
        self.priority_scheduling_enabled: bool = False
        self.async_scheduling_enabled: bool = False
        self.generate_accepts_priority: bool = False

    async def initialize(self) -> None:
        if self.is_ready:
            return

        enable_chunked_prefill = ENABLE_CHUNKED_PREFILL
        priority_requested = ENABLE_PRIORITY_SCHEDULING
        async_requested = ENABLE_ASYNC_SCHEDULING and not SPECULATIVE_METHOD
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
        if priority_requested:
            engine_kwargs["scheduling_policy"] = "priority"
        if async_requested:
            engine_kwargs["async_scheduling"] = True
        if enable_chunked_prefill:
            if MAX_NUM_PARTIAL_PREFILLS is not None:
                engine_kwargs["max_num_partial_prefills"] = MAX_NUM_PARTIAL_PREFILLS
            if MAX_LONG_PARTIAL_PREFILLS is not None:
                engine_kwargs["max_long_partial_prefills"] = MAX_LONG_PARTIAL_PREFILLS
            if LONG_PREFILL_TOKEN_THRESHOLD is not None:
                engine_kwargs["long_prefill_token_threshold"] = LONG_PREFILL_TOKEN_THRESHOLD

        if SPECULATIVE_MODEL:
            engine_kwargs["enable_chunked_prefill"] = False
            engine_kwargs["speculative_model"] = SPECULATIVE_MODEL
            engine_kwargs["num_speculative_tokens"] = NUM_SPECULATIVE_TOKENS
            
            if SPECULATIVE_MODEL == "[ngram]":
                engine_kwargs["ngram_prompt_lookup_max"] = NGRAM_PROMPT_LOOKUP_MAX
                engine_kwargs["ngram_prompt_lookup_min"] = NGRAM_PROMPT_LOOKUP_MIN

                
        valid_params = set(inspect.signature(AsyncEngineArgs.__init__).parameters.keys())
        unsupported = [k for k in engine_kwargs if k not in valid_params]
        dropped_args: set[str] = set()
        for k in unsupported:
            dropped_args.add(k)
            warnings.warn(f"  WARNING: dropping unsupported arg '{k}' (not in this vLLM version)")
            del engine_kwargs[k]

        self.engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(**engine_kwargs))
        self.generate_accepts_priority = "priority" in inspect.signature(self.engine.generate).parameters
        self.priority_scheduling_enabled = (
            priority_requested
            and "scheduling_policy" not in dropped_args
            and self.generate_accepts_priority
        )
        self.async_scheduling_enabled = async_requested and "async_scheduling" not in dropped_args
        if priority_requested and not self.priority_scheduling_enabled:
            warnings.warn(
                "Priority scheduling was requested but is not fully supported by this vLLM runtime; "
                "falling back to default request ordering."
            )
        if ENABLE_ASYNC_SCHEDULING and SPECULATIVE_METHOD:
            warnings.warn(
                "Async scheduling is disabled because speculative decoding is enabled."
            )
        elif async_requested and not self.async_scheduling_enabled:
            warnings.warn(
                "Async scheduling was requested but is not supported by this vLLM runtime; "
                "falling back to synchronous scheduling."
            )

        # Handle both sync and async get_tokenizer across vLLM versions
        tokenizer = self.engine.get_tokenizer()
        if inspect.isawaitable(tokenizer):
            tokenizer = await tokenizer
        self.tokenizer = tokenizer

        self.is_ready = True
        print("vLLM Engine initialized and ready.")

    def _get_request_priority(self, request: ChatRequest, template_hash: str) -> int:
        if request.priority is not None:
            return int(request.priority)
        return int(round(self.analytics.estimate_decode_budget(template_hash)))

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
        template_cache_key = ""
        if is_deterministic:
            cache_key = ResponseCache.make_key(prompt, request.temperature, request.max_tokens)
            cached = self.cache.get(cache_key)
            if cached is not None:
                return ChatResponse(output=cached.output, logprobs=cached.logprobs)

        exact_hash, template_hash = self.analytics.get_hashes(prompt)
        if is_deterministic and self.template_cache is not None:
            template_cache_key = ResponseCache.make_key(
                template_hash,
                request.temperature,
                request.max_tokens,
            )
            template_cached = self.template_cache.get(template_cache_key)
            if template_cached is not None:
                if cache_key:
                    self.cache.put(cache_key, template_cached)
                return ChatResponse(output=template_cached.output, logprobs=template_cached.logprobs)
        t_preprocess = time.perf_counter()

        # ── Stage 3-5: Queue + Prefill + Decode ──
        sampling_params = SamplingParams(
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            logprobs=1,
        )
        request_id = random_uuid()
        generate_kwargs: dict = {}
        if self.priority_scheduling_enabled:
            generate_kwargs["priority"] = self._get_request_priority(request, template_hash)
        results_generator = self.engine.generate(
            prompt, sampling_params, request_id, **generate_kwargs,
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
            cached_response = CachedResponse(output=text_output, logprobs=logprobs)
            self.cache.put(cache_key, cached_response)
            if template_cache_key and self.template_cache is not None:
                self.template_cache.put(template_cache_key, cached_response)
        t_done = time.perf_counter()
        return ChatResponse(output=text_output, logprobs=logprobs)
