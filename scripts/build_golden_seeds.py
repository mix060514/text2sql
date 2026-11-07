#!/usr/bin/env python3
"""Build golden seeds from the legacy templates, checking SQL against legacy pandas answers.

uv run python scripts/build_golden_seeds.py                  # 1 seed per template (default)
uv run python scripts/build_golden_seeds.py --per-template 2
uv run python scripts/build_golden_seeds.py --all            # every entity instance
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

project = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project / "src"))

from text2sql.datagen.templates import build_instances, select_seeds  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-template", type=int, default=1)
    parser.add_argument("--all", action="store_true", help="keep every entity instance")
    parser.add_argument("--out", type=Path, default=project / "data" / "seeds" / "golden_seeds.jsonl")
    args = parser.parse_args()

    seeds, problems = build_instances()
    print(f"template instances: {len(seeds)} from {len({s['template'] for s in seeds})} templates")
    print(f"SQL vs pandas mismatches: {len(problems)}")
    for problem in problems:
        print(f"  {problem['seed_id']}: {problem['problem']}")
    for seed in seeds:
        if "exclude" in seed:
            print(f"excluded {seed['seed_id']}: {seed['exclude']}")
    if problems:
        print("refusing to write seeds while the two oracles disagree")
        return 1
    chosen = select_seeds(seeds, None if args.all else args.per_template)
    with args.out.open("w", encoding="utf-8") as handle:
        for seed in chosen:
            handle.write(json.dumps(seed, ensure_ascii=False) + "\n")
    entities = Counter(g["any_of"][0] for s in chosen for g in s["keep"]
                       if g["type"] == "entity" and not g["name"].startswith("年份"))
    plots = Counter("unspecified" if s["expected_plot"] is None else
                    (s["expected_plot"].get("chart_type") if s["expected_plot"].get("should_plot") else "no chart")
                    for s in chosen)
    print(f"wrote {len(chosen)} seeds to {args.out}")
    print("entity coverage:", dict(sorted(entities.items())))
    print("chart types:", dict(plots), "| eval types:", dict(Counter(s["eval_type"] for s in chosen)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
