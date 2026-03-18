# Benchmark Suite

This folder contains the evaluation tools for the LLM engines.

## Setup

We recommend using [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
uv sync
```

## Running the Benchmark

The benchmark scripts measure throughput, latency, and quality metrics (Perplexity and Trace Length).
Supply an experiment name so that the results can be stored neatly for future reference.

```bash
uv run runner_chat.py --url $ENGINE_URL --data data/train.jsonl --concurrency 128 --experiment <experiment_name>
```
