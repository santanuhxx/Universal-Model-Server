import asyncio
import argparse
from benchmark.runner import BenchmarkRunner


async def main(args):
    runner = BenchmarkRunner(base_url=args.url)
    await runner.run(
        model_name=args.model,
        inputs={"float_input": [[1.0, 2.0]]},
        concurrency=args.concurrency,
        duration_seconds=args.duration,
        priority=args.priority,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Universal Model Server — Benchmark CLI"
    )
    parser.add_argument("--url",         default="http://localhost:8000")
    parser.add_argument("--model",       default="echo_model")
    parser.add_argument("--concurrency", type=int,   default=10)
    parser.add_argument("--duration",    type=float, default=30.0)
    parser.add_argument("--priority",    default="normal",
                        choices=["urgent", "normal", "batch"])
    args = parser.parse_args()
    asyncio.run(main(args))