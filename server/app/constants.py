"""
Configuration Constants for the Chat Engine.

This module defines all static configuration parameters used by the vLLM engine,
including the model identifier, hardware utilization constraints, scheduling
policies, and advanced optimizations such as caching and speculative decoding.
"""

MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
MAX_MODEL_LENGTH = 2048

KV_CACHE_DTYPE = "fp8_e5m2"
ENABLE_CHUNKED_PREFILL = True
GPU_MEMORY_UTILIZATION = 0.80
MAX_NUM_SEQS = 128
SCHEDULING_POLICY = "priority"

ENABLE_PREFIX_CACHING = True
NUM_SCHEDULER_STEPS = 10
MAX_NUM_BATCHED_TOKENS = 4096

ENABLE_SPECULATIVE = False
SPECULATIVE_MODEL = "[ngram]"
NUM_SPECULATIVE_TOKENS = 5
NGRAM_PROMPT_LOOKUP_MAX = 4

RESPONSE_CACHE_MAX_SIZE = 16_384