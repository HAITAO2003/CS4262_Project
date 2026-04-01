"""
Configuration Constants for the Chat Engine.

This module defines all static configuration parameters used by the vLLM engine,
including the model identifier, hardware utilization constraints, scheduling
policies, and advanced optimizations such as caching and speculative decoding.
"""

MODEL_NAME = "Qwen/Qwen3-1.7B-FP8"
MAX_MODEL_LENGTH = 512         
KV_CACHE_DTYPE = "fp8"
ENABLE_CHUNKED_PREFILL = True
GPU_MEMORY_UTILIZATION = 0.85
MAX_NUM_SEQS = 320

ENABLE_PREFIX_CACHING = True
MAX_NUM_BATCHED_TOKENS = 16384

#SPECULATIVE_MODEL = "AngelSlim/Qwen3-1.7B_eagle3"
SPECULATIVE_MODEL = ""
SPECULATIVE_METHOD = "eagle3" 
NGRAM_PROMPT_LOOKUP_MAX = 4
NUM_SPECULATIVE_TOKENS = 3

RESPONSE_CACHE_MAX_SIZE = 16_384
