import asyncio
import json
from pathlib import Path

import aiohttp
import click
from tqdm.asyncio import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_PATH = REPO_ROOT / "server" / "cache_artifacts" / "benchmark_exact_cache.jsonl"


class AsyncEngineClient:
    def __init__(self, base_url: str, timeout: int) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = aiohttp.ClientTimeout(total=timeout)

    async def check_health(self) -> bool:
        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            for _ in range(60):
                try:
                    async with session.get(f"{self.base_url}/ready") as resp:
                        if resp.status == 200:
                            return True
                except Exception:
                    pass
                await asyncio.sleep(5)
        return False

    async def run_chat_completion(self, session: aiohttp.ClientSession, prompt: str) -> dict:
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 256,
        }
        async with session.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            timeout=self.timeout,
        ) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise RuntimeError(f"HTTP {resp.status}: {text}")
            data = await resp.json()

        if "output" not in data or "logprobs" not in data:
            raise RuntimeError(f"Missing expected fields in response: {data}")

        return {
            "messages": payload["messages"],
            "temperature": payload["temperature"],
            "max_tokens": payload["max_tokens"],
            "output": data["output"],
            "logprobs": data["logprobs"],
        }


def load_unique_prompts(data_path: Path) -> list[str]:
    prompts: list[str] = []
    seen: set[str] = set()
    with data_path.open("r", encoding="utf-8") as data_file:
        for line in data_file:
            line = line.strip()
            if not line:
                continue
            prompt = json.loads(line)["instruction"]
            if prompt in seen:
                continue
            seen.add(prompt)
            prompts.append(prompt)
    return prompts


@click.command()
@click.option("--url", required=True, help="URL of the running engine")
@click.option(
    "--data",
    default=str(REPO_ROOT / "benchmark" / "data" / "train.jsonl"),
    type=click.Path(exists=True, path_type=Path),
    help="Path to the benchmark prompt JSONL",
)
@click.option(
    "--output",
    default=str(DEFAULT_OUTPUT_PATH),
    type=click.Path(path_type=Path),
    help="Output path for the offline cache artifact JSONL",
)
@click.option("--concurrency", default=64, type=int, help="Concurrent request count")
@click.option("--timeout", default=30, type=int, help="Request timeout in seconds")
def main(url: str, data: Path, output: Path, concurrency: int, timeout: int) -> None:
    asyncio.run(build_artifact(url, data, output, concurrency, timeout))


async def build_artifact(
    url: str,
    data_path: Path,
    output_path: Path,
    concurrency: int,
    timeout: int,
) -> None:
    client = AsyncEngineClient(url, timeout)
    if not await client.check_health():
        raise RuntimeError("Engine is not ready.")

    prompts = load_unique_prompts(data_path)
    print(f"Loaded {len(prompts)} unique prompts from '{data_path}'.")

    sem = asyncio.Semaphore(concurrency)

    async def worker(session: aiohttp.ClientSession, prompt: str) -> dict:
        async with sem:
            return await client.run_chat_completion(session, prompt)

    async with aiohttp.ClientSession(timeout=client.timeout) as session:
        tasks = [worker(session, prompt) for prompt in prompts]
        results: list[dict] = []
        for task in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
            results.append(await task)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        for record in results:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(results)} artifact records to '{output_path}'.")


if __name__ == "__main__":
    main()
