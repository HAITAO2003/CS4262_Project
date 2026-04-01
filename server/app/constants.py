"""
Configuration Constants for the Chat Engine.

This module defines all static configuration parameters used by the vLLM engine,
including the model identifier, hardware utilization constraints, scheduling
policies, and advanced optimizations such as caching and speculative decoding.
"""

MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
MAX_MODEL_LENGTH = 512         
KV_CACHE_DTYPE = "fp8_e5m2"
ENABLE_CHUNKED_PREFILL = False
GPU_MEMORY_UTILIZATION = 0.95
MAX_NUM_SEQS = 128

ENABLE_PREFIX_CACHING = True
NUM_SCHEDULER_STEPS = 1         
MAX_NUM_BATCHED_TOKENS = 8192  

#list of models to test out here: "Qwen/Qwen3-1.7B", "Qwen/Qwen3-0.6B" and "ngram"
SPECULATIVE_MODEL = "Qwen/Qwen3-1.7B"
NGRAM_PROMPT_LOOKUP_MAX = 4
NUM_SPECULATIVE_TOKENS = 5

RESPONSE_CACHE_MAX_SIZE = 16_384
