#!/usr/bin/env python3
"""Generate a synthetic golden + guardrail test set with the local model.

uv run python scripts/generate_synthetic.py            # all seeds in data/seeds/
uv run python scripts/generate_synthetic.py --variants 1 --seeds data/seeds/golden_seeds.jsonl
"""
import argparse
import sys
from pathlib import Path

project = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project / "src"))

from text2sql.datagen.llm import LocalJSONModel  # noqa: E402
from text2sql.datagen.pipeline import run  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", nargs="+", type=Path,
                        default=sorted((project / "data" / "seeds").glob("*.jsonl")))
    parser.add_argument("--variants", type=int, default=2, help="accepted rewrites per seed")
    parser.add_argument("--max-rounds", type=int, default=3, help="rewrite rounds including regenerations")
    parser.add_argument("--out", type=Path, default=project / "data" / "synthetic" / "runs")
    args = parser.parse_args()
    run_dir = run(args.seeds, LocalJSONModel(), args.out, variants=args.variants,
                  max_rounds=args.max_rounds, log=lambda m: print(m, flush=True))
    print(f"run directory: {run_dir}")
    print((run_dir / "report.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
