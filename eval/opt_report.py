"""Per-question-type effort and outcome stats for the optimization page.

Reads eval result files (template e2e runs and synthetic runs) and reports, per question
type: success rate, timeout rate, first-attempt solve rate (passed with one SQL execution)
and mean seconds. Numbers come only from result files; nothing is typed in by hand.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL_SET = ROOT / "data" / "eval_set_v3.jsonl"
SYNTHETIC_RUNS = ROOT / "data" / "synthetic" / "runs"


def template_type(tags) -> str:
    tags = list(tags or [])
    t = set(tags)
    if "trend" in t and "monthly" in t:
        return "跨年逐月趨勢"
    if "trend" in t and "quarterly" in t:
        return "跨年逐季趨勢"
    if "customer_segmentation" in t:
        return "前 N 大客戶的類別占比"
    if "customer_churn" in t:
        return "客戶跨年變動"
    if "yoy" in t or "comparison" in t:
        return "年增率／同期比較"
    if "share" in t:
        return "占比"
    if "quarterly" in t:
        return "季度彙總"
    if "trend" in t:
        return "年度總計"
    if tags and tags[0] in ("top_products", "top_customer", "top", "regional_preference"):
        return "排名／最高"
    return "彙總查詢"


def _template_items():
    items = {}
    for line in EVAL_SET.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            items[item["question"]] = item
    return items


def _seed_questions():
    out = {}
    for path in sorted(SYNTHETIC_RUNS.glob("*/dataset.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                if item.get("source") == "seed":
                    out[item["seed_id"]] = item["question"]
    return out


def load_template(path) -> list[dict]:
    """eval_e2e.py e2e_results.json -> normalized rows."""
    items = _template_items()
    rows = []
    for r in json.loads(Path(path).read_text(encoding="utf-8")):
        reason = str(r.get("sql_reasoning") or "")
        rows.append({
            "question": r["question"],
            "type": template_type(items.get(r["question"], {}).get("tags")),
            "passed": bool(r.get("e2e_success")),
            "timeout": r.get("sql_query") == "TIMEOUT" or "Timeout" in reason,
            "seconds": r.get("seconds") or None,
            "attempts": r.get("sql_attempts") or None,
        })
    return rows


def _synthetic_items():
    out = {}
    for path in sorted(SYNTHETIC_RUNS.glob("*/dataset.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                out[item["question"]] = item
    return out


def load_synthetic(path, subset, review=None) -> list[dict]:
    """eval_e2e.py --synthetic synthetic_results.json -> normalized rows of one subset.

    With a review mapping, "reviewed" is True for passes and for strict-oracle failures
    that review_passes() verifies; "passed" stays the strict oracle result.
    """
    seeds = _seed_questions()
    items = _synthetic_items() if review else {}
    rows = []
    for r in json.loads(Path(path).read_text(encoding="utf-8")):
        is_block = r.get("label") == "block"
        sub = "block" if is_block else ("complex" if str(r.get("seed_id", "")).startswith("X-") else "template")
        if sub != subset:
            continue
        text = json.dumps(r.get("error") or "", ensure_ascii=False) + str(r.get("reason") or "")
        rows.append({
            "question": r["question"],
            "type": r.get("category") if is_block else seeds.get(r.get("seed_id"), r.get("seed_id")),
            "seed_id": r.get("seed_id"),
            "passed": bool(r.get("passed")),
            "reviewed": bool(r.get("passed")) or bool(
                review and r["question"] in items and review_passes(r, items[r["question"]], review)),
            "timeout": "Timeout" in text,
            "seconds": r.get("seconds") or None,
            "attempts": r.get("sql_attempts") or None,
        })
    return rows


def stats(rows) -> dict:
    n = len(rows)
    secs = [r["seconds"] for r in rows if r["seconds"]]
    tried = [r for r in rows if r["attempts"] is not None]
    return {
        "n": n,
        "success": sum(r["passed"] for r in rows) / n if n else None,
        "timeout": sum(r["timeout"] for r in rows) / n if n else None,
        "first_try": (sum(r["passed"] and r["attempts"] == 1 for r in tried) / len(tried)) if tried else None,
        "avg_seconds": round(sum(secs) / len(secs), 1) if secs else None,
    }


def by_type(rows) -> dict:
    groups = {}
    for r in rows:
        groups.setdefault(r["type"], []).append(r)
    return {k: stats(v) for k, v in groups.items()}


def hard_types(rows, max_seconds=120.0) -> list[str]:
    """Not reliably solved in one go: any failure, any retry, or slow on average."""
    out = []
    for name, s in by_type(rows).items():
        if s["success"] < 1 or (s["first_try"] is not None and s["first_try"] < 1) \
                or (s["avg_seconds"] or 0) > max_seconds:
            out.append(name)
    return out


def _normalize(value, rules):
    if isinstance(value, str):
        if "month" in rules and len(value) == 7 and value[4] == "-" and value[:4].isdigit():
            value = value[5:]
        if "weekday" in rules:
            value = value.replace("週", "星期")
        return value
    if isinstance(value, (int, float)) and "abs" in rules:
        return abs(value)
    return value


def review_passes(record: dict, item: dict, review: dict) -> bool:
    """A strict-oracle failure counts as correct only if the declared transformations make
    the actual rows equal the oracle rows (same row count, values, membership)."""
    import synthetic  # eval/ is on sys.path for both the builder and the tests

    entry = review.get(record.get("seed_id"))
    if not entry or record.get("workflow_status") != "completed":
        return False
    gt = item["ground_truth"]
    keep = [j for j, c in enumerate(gt["columns"]) if c not in entry["drop"]]
    rules = entry["normalize"]
    expected = [[_normalize(row[j], rules) for j in keep] for row in gt["rows"]]
    actual = [[_normalize(v, rules) for v in row] for row in record.get("rows") or []]
    if len(actual) != (record.get("row_count") or 0):
        return False  # stored rows were truncated; cannot verify
    return synthetic.rows_match(expected, actual)
