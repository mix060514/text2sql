#!/usr/bin/env python3
"""Build SQL-first complex golden seeds: each SQL is checked against an independent pandas
implementation on the source CSV before its question is accepted as a seed.

uv run python scripts/build_complex_seeds.py
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

project = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project / "src"))

from text2sql.datagen.complex import build_complex_seeds  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=project / "data" / "seeds" / "complex_seeds.jsonl")
    args = parser.parse_args()
    seeds, problems = build_complex_seeds()
    print(f"complex seeds: {len(seeds)}; SQL vs pandas mismatches: {len(problems)}")
    for problem in problems:
        print(f"  {problem['seed_id']}: {problem['problem']}")
    if problems:
        print("refusing to write seeds while the two oracles disagree")
        return 1
    with args.out.open("w", encoding="utf-8") as handle:
        for seed in seeds:
            handle.write(json.dumps(seed, ensure_ascii=False) + "\n")
    print(f"wrote {args.out}")
    print("eval types:", dict(Counter(s["eval_type"] for s in seeds)),
          "| metric:", dict(Counter(str(s["metric"]) for s in seeds)),
          "| charts:", sum(1 for s in seeds if s["expected_plot"]))
    for seed in seeds:
        print(f"  {seed['seed_id']:<45} {seed['question']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
