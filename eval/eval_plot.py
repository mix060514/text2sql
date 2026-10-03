#!/usr/bin/env python3
"""Compatibility entry point: evaluate real query results, without mock rows."""
import asyncio
import argparse
from eval_e2e import run_evaluation

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real end-to-end plot evaluation")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    asyncio.run(run_evaluation(limit=args.limit, plot_only=True))
