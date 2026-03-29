import click
import json
import os
import sys
import asyncio
import aiohttp
import time
import numpy as np
from dataclasses import asdict, dataclass
from typing import Optional
from tqdm.asyncio import tqdm
import traceback

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class Request:
    prompt: str
    response: Optional[str]
    request_start_time: float
    request_end_time: float
    latency: float
    error: Optional[str]
    perplexity: Optional[float]
    output_token_count: Optional[int] = None


@dataclass
class Result:
    total_requests: int
    passed: int
    failed: int
    pass_rate: float
    fail_rate: float
    throughput: float
    mean_latency: Optional[float]
    p50_latency: Optional[float]
    p99_latency: Optional[float]
    average_perplexity: Optional[float]

class AsyncEngineClient:
    def __init__(self, base_url="http://localhost:8000", timeout=30):
        self.base_url = base_url
        self.timeout = aiohttp.ClientTimeout(total=timeout)

    async def check_health(self):
        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            max_retries = 60
            for i in range(max_retries):
                try:
                    async with session.get(f"{self.base_url}/ready") as resp:
                        if resp.status == 200:
                            return True
                        print(f"Waiting for engine... (HTTP {resp.status})")
                except Exception as e:
                    print(f"Waiting for engine... ({e})")
                
                await asyncio.sleep(5)
            return False

    async def run_chat_completion(self, session, chat_req):
        start = time.time()
        async with session.post(f"{self.base_url}/v1/chat/completions", json=chat_req, timeout=self.timeout) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise Exception(f"HTTP {resp.status}: {text}")
            try:
                data = await resp.json()
            except Exception as e:
                text = await resp.text()
                raise Exception(f"JSON Parse Error: {e} | Body: {text[:200]}")
            end = time.time()
            return data, start, end

@click.command()
@click.option('--url', default='http://localhost:8000', help='URL of the engine')
@click.option('--data', required=True, type=click.Path(exists=True), help='Path to the performance prompts JSONL file')
@click.option('--concurrency', default=128, help='Concurrency level')
@click.option('--timeout', default=30, type=int, help='Request timeout in seconds')
@click.option('--experiment-name', '--experiment', required=True, help='Name for the benchmark result directory')
def main(url, data, concurrency, timeout, experiment_name):
    """Benchmark suite for Track 2: Customer Chat."""
    if not data.endswith('.jsonl') and not data.endswith('.txt'): # Allowing .txt if it contains JSONL
        pass 
    # Strict JSONL check
    print(f"Connecting to engine at {url} using data {data} (concurrency={concurrency}, timeout={timeout}s)...")
    asyncio.run(run_performance(url, data, concurrency, timeout, experiment_name))

def create_result_dir(experiment_name):
    results_root = os.path.join(REPO_ROOT, "results")
    os.makedirs(results_root, exist_ok=True)

    candidate = os.path.join(results_root, experiment_name)
    suffix = 1
    while os.path.exists(candidate):
        candidate = os.path.join(results_root, f"{experiment_name}-{suffix}")
        suffix += 1

    os.makedirs(candidate)
    return candidate


async def run_performance(url, data_path, concurrency, timeout, experiment_name):
    client = AsyncEngineClient(url, timeout=timeout)
    if not await client.check_health():
        print("Error: Engine not healthy.")
        sys.exit(1)

    print("=== Track 2: Performance Check ===")
    
    prompts = []
    with open(data_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            
            try:
                data = json.loads(line)
                prompts.append(data['instruction'])
            except json.JSONDecodeError:
                print(f"Error: Performance data must be JSONL. Failed to parse line: {line[:50]}...")
                sys.exit(1)
        
    if not prompts:
        print(f"Error: No prompts found in {data_path}")
        sys.exit(1)

    # Generate workload
    work_items = prompts
    
    results = []
    sem = asyncio.Semaphore(concurrency)
    
    async def worker(prompt):
        async with sem:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0,
                    "max_tokens": 256 # Higher for performance test
                }
                try:
                    response, request_start_time, request_end_time = await client.run_chat_completion(session, payload)
                    latency = request_end_time - request_start_time
                    if 'output' not in response:
                        print(f"\\n[FAIL] Prompt: {prompt[:30]}... Missing 'output' in response: {response}")
                        return Request(
                            prompt=prompt,
                            response=None,
                            request_start_time=request_start_time,
                            request_end_time=request_end_time,
                            latency=latency,
                            error=f"Missing 'output' in response: {response}",
                            perplexity=None,
        
                        )
                    logprobs = response['logprobs']
                    avg_logprob = sum(logprobs) / len(logprobs)
                    perplexity = np.exp(-avg_logprob)
                    output_token_count = len(logprobs)

                    return Request(
                        prompt=prompt,
                        response=response['output'],
                        request_start_time=request_start_time,
                        request_end_time=request_end_time,
                        latency=latency,
                        error=None,
                        perplexity=float(perplexity),
                        output_token_count=output_token_count,
                    )
                except Exception as e:
                    print(f"\\n[ERROR] Prompt: {prompt[:30]}... Exception: {e}")
                    traceback.print_exc()
                    return Request(
                        prompt=prompt,
                        response=None,
                        request_start_time=0,
                        request_end_time=0,
                        latency=0,
                        error=str(e),
                        perplexity=None,
                    )

    start_time = time.time()
    tasks_to_run = [worker(p) for p in work_items]
    
    print(f"Executing {len(work_items)} requests with concurrency {concurrency}...")
    for f in tqdm(asyncio.as_completed(tasks_to_run), total=len(tasks_to_run)):
        results.append(await f)
        
    total_time = time.time() - start_time
    passed = sum(1 for r in results if r.error is None)
    failed = len(results) - passed
    latencies = [r.latency for r in results if r.error is None]
    perplexities = [r.perplexity for r in results if r.perplexity is not None]
    result = Result(
        total_requests=len(results),
        passed=passed,
        failed=failed,
        pass_rate=passed / len(results),
        fail_rate=failed / len(results),
        throughput=len(results) / total_time,
        mean_latency=float(np.mean(latencies)) if latencies else None,
        p50_latency=float(np.percentile(latencies, 50)) if latencies else None,
        p99_latency=float(np.percentile(latencies, 99)) if latencies else None,
        average_perplexity=float(np.mean(perplexities)) if perplexities else None,
    )

    result_dir = create_result_dir(experiment_name)
    with open(os.path.join(result_dir, "result.json"), "w") as f:
        json.dump(asdict(result), f, indent=2)
    with open(os.path.join(result_dir, "requests.jsonl"), "w") as f:
        for request in results:
            f.write(json.dumps(asdict(request)) + "\n")
    
    print(f"\nPerformance Metrics:")
    print(f"  Throughput: {result.throughput:.2f} req/s")
    print(f"  Passed: {result.passed}")
    print(f"  Failed: {result.failed}")
    if result.mean_latency is not None:
        print(f"  Avg Latency: {result.mean_latency:.4f}s")
        print(f"  P50 Latency: {result.p50_latency:.4f}s")
        print(f"  P99 Latency: {result.p99_latency:.4f}s")
    if result.average_perplexity is not None:
        print(f"  Avg Perplexity: {result.average_perplexity:.4f}")
    print(f"  Results saved to: {result_dir}")

if __name__ == '__main__':
    main()
