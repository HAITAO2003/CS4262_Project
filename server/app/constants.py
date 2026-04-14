"""
Configuration Constants for the Chat Engine.

This module defines all static configuration parameters used by the vLLM engine,
including the model identifier, hardware utilization constraints, scheduling
policies, and advanced optimizations such as caching and speculative decoding.
"""

MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
MAX_MODEL_LENGTH = 8192
KV_CACHE_DTYPE = "fp8_e5m2"
ENABLE_CHUNKED_PREFILL = False
GPU_MEMORY_UTILIZATION = 0.85
MAX_NUM_SEQS = 256

ENABLE_PREFIX_CACHING = True
MAX_NUM_BATCHED_TOKENS =  8192
ENABLE_PRIORITY_SCHEDULING = True
ENABLE_TEMPLATE_RESPONSE_CACHING = False
ENABLE_ASYNC_SCHEDULING = True
ENABLE_OFFLINE_RESPONSE_CACHE_ARTIFACT = True
MAX_NUM_PARTIAL_PREFILLS = None
MAX_LONG_PARTIAL_PREFILLS = None
LONG_PREFILL_TOKEN_THRESHOLD = None
OFFLINE_RESPONSE_CACHE_ARTIFACT_PATH = "../cache_artifacts/benchmark_exact_cache.jsonl"

SPECULATIVE_MODEL = "Zjcxy-SmartAI/Eagle3-Qwen3-4B-Instruct-2507-zh"
# SPECULATIVE_METHOD = "eagle3"   # OR "ngram" OR None
SPECULATIVE_METHOD = "ngram"
NGRAM_PROMPT_LOOKUP_MAX = 5
NGRAM_PROMPT_LOOKUP_MIN = 2
NUM_SPECULATIVE_TOKENS = 3

RESPONSE_CACHE_MAX_SIZE = 16_384
TEMPLATE_RESPONSE_CACHE_MAX_SIZE = RESPONSE_CACHE_MAX_SIZE
