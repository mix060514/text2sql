#!/usr/bin/env python3
"""Write the block-side seeds (data/seeds/guardrail_seeds.jsonl) and print a review table."""
import json
import sys
from collections import Counter
from pathlib import Path

project = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project / "src"))

from text2sql.datagen.checks import rule_check  # noqa: E402
from text2sql.datagen.guardrail import build_guardrail_seeds  # noqa: E402


def main() -> int:
    seeds = build_guardrail_seeds()
    bad = [(s["seed_id"], r) for s in seeds for r in rule_check(s["question"], s, set()) if not r.startswith("duplicate")]
    for seed_id, reason in bad:
        print(f"seed fails its own rules: {seed_id}: {reason}")
    if bad:
        return 1
    out = project / "data" / "seeds" / "guardrail_seeds.jsonl"
    out.write_text("".join(json.dumps(s, ensure_ascii=False) + "\n" for s in seeds), encoding="utf-8")
    print(f"wrote {len(seeds)} seeds to {out}")
    print("by category:", dict(Counter(s["category"] for s in seeds)))
    print("\n| 種子 | 類別 | 題目 | 人工確認：該擋？ |\n|---|---|---|---|")
    for s in seeds:
        print(f"| {s['seed_id']} | {s['category']} | {s['question']} | [ ] |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
