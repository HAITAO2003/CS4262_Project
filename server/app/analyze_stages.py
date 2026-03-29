"""
Analyze stage-level profiling from /tmp/stage_profile.jsonl.

Usage: python analyze_stages.py [path_to_stage_profile.jsonl]
"""

import json
import sys
import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/stage_profile.jsonl"

data = [json.loads(line) for line in open(path)]

print(f"Total profiled requests: {len(data)}\n")

fields = [
    ("preprocess_ms",     "Preprocessing (template + cache check)"),
    ("ttft_ms",           "TTFT (queue wait + prefill)"),
    ("decode_ms",         "Decode (token generation)"),
    ("postprocess_ms",    "Postprocessing (logprobs + cache insert)"),
    ("total_ms",          "Total request time"),
    ("output_tokens",     "Output token count"),
    ("effective_max_tokens", "Effective max_tokens used"),
    ("tpot_ms",           "Time per output token"),
]

print(f"{'Metric':<45s}  {'P50':>8s}  {'P95':>8s}  {'P99':>8s}  {'Mean':>8s}  {'Max':>8s}")
print("-" * 100)

for field_name, label in fields:
    vals = [d[field_name] for d in data if d.get(field_name) is not None]
    if not vals:
        continue
    arr = np.array(vals)
    print(f"{label:<45s}  {np.percentile(arr,50):8.2f}  {np.percentile(arr,95):8.2f}  "
          f"{np.percentile(arr,99):8.2f}  {np.mean(arr):8.2f}  {np.max(arr):8.2f}")

# Output length distribution
print("\n=== Output Length Distribution ===")
output_lens = np.array([d["output_tokens"] for d in data if d.get("output_tokens")])
at_ceiling = np.sum(output_lens >= 256)
print(f"  Requests hitting 256 ceiling:  {at_ceiling} ({at_ceiling/len(output_lens)*100:.1f}%)")
print(f"  Requests finishing with EOS:   {len(output_lens) - at_ceiling} ({(len(output_lens) - at_ceiling)/len(output_lens)*100:.1f}%)")
print(f"  Mean wasted reservation:       {256 - np.mean(output_lens):.0f} tokens/request (if max_tokens=256)")

# Time breakdown as percentage of total
print("\n=== Time Breakdown (% of total) ===")
for field_name, label in [("preprocess_ms", "Preprocessing"), ("ttft_ms", "TTFT"), ("decode_ms", "Decode"), ("postprocess_ms", "Postprocessing")]:
    vals = [d[field_name] for d in data if d.get(field_name) is not None and d.get("total_ms")]
    totals = [d["total_ms"] for d in data if d.get(field_name) is not None and d.get("total_ms")]
    if vals and totals:
        pcts = [v / t * 100 for v, t in zip(vals, totals) if t > 0]
        print(f"  {label:<20s}  Mean: {np.mean(pcts):5.1f}%   Median: {np.median(pcts):5.1f}%")
