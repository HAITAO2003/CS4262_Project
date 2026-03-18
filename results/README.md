# Results

Experiment results are automatically saved to this directory.
Each directory represents one benchmark run.

* `requests.jsonl` has the request level breakdown, so that we can do some simple profiling
* `result.json` has the summary statistics

## Important defaults

By default, let's run experiments with concurrency `128` against `benchmark/data/train.jsonl`.
If we run on any other dataset or concurrency level (or non RTX 5080 hardware), let's try to label that in the experiment name so its clear.
