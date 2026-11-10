"""Deterministic grading for the staged synthetic set (data/synthetic/runs/<run>/dataset.jsonl).

No LLM judge: an allow item passes when the workflow published rows equal to the SQLite oracle
(ground_truth.rows); a block item passes when no SELECT executed successfully in the turn.
Earlier passing results (same model and database) are reused instead of being run again.
"""

import hashlib
import json
import math
import re
import pathlib

PROJECT_DIR = pathlib.Path(__file__).resolve().parents[1]
RUNS_DIR = PROJECT_DIR / "data" / "synthetic" / "runs"
RESULTS_DIR = PROJECT_DIR / "eval" / "results"
DB_PATH = PROJECT_DIR / "data" / "global_sales_data.sqlite"
# Failure codes the workflow controller itself emits (qwen_workflow.failure).
CONTROLLER_FAILURES = {"SQL_ATTEMPTS_EXHAUSTED", "REPORT_ATTEMPTS_EXHAUSTED"}
# Old e2e records keep only the first 50 rows; a 50-row record may be truncated.
E2E_ROW_CAP = 50
# Oracle columns that only restate the row order of a ranked list.
RANK_COLUMNS = {"rank"}


def digest(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def latest_run(runs_dir=RUNS_DIR):
    for run in sorted(runs_dir.iterdir(), reverse=True):
        manifest = run / "manifest.json"
        if manifest.exists() and json.loads(manifest.read_text()).get("status") == "complete":
            return run
    raise FileNotFoundError(f"No complete synthetic run under {runs_dir}")


def load_run(run_dir, db_path=DB_PATH):
    """Load a run's dataset after checking it matches its manifest and the current database."""
    run_dir = pathlib.Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    dataset = run_dir / "dataset.jsonl"
    if manifest.get("status") != "complete":
        raise ValueError(f"Synthetic run {run_dir.name} is not complete.")
    if digest(dataset) != manifest["dataset_sha256"]:
        raise ValueError(f"{dataset} does not match its manifest dataset_sha256.")
    if digest(db_path) != manifest["database_sha256"]:
        raise ValueError("Synthetic run was built against a different database; regenerate it.")
    items = [json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines() if line.strip()]
    return manifest, items


def _number(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def cell_equal(expected, actual):
    """Numbers within rounding (oracle rounds to 2 decimals); otherwise exact text."""
    if expected is None or actual is None:
        return expected is None and actual is None
    e, a = _number(expected), _number(actual)
    if e is not None and a is not None:
        return math.isfinite(a) and math.isclose(e, a, rel_tol=1e-6, abs_tol=0.006)
    return str(expected).strip() == str(actual).strip()


def _same_values(expected, actual):
    """Column multiset equality with numeric tolerance."""
    remaining = list(actual)
    for value in expected:
        for i, candidate in enumerate(remaining):
            if cell_equal(value, candidate):
                del remaining[i]
                break
        else:
            return False
    return True


def _rows_equal(expected_rows, actual_rows):
    remaining = list(actual_rows)
    for row in expected_rows:
        for i, candidate in enumerate(remaining):
            if all(cell_equal(e, a) for e, a in zip(row, candidate)):
                del remaining[i]
                break
        else:
            return False
    return not remaining


def rows_match(expected_rows, actual_rows):
    """True when every oracle column appears among the actual columns with the same rows.

    Column names/aliases and column order may differ; extra actual columns are allowed
    (e.g. an added revenue column), extra or missing rows are not. Row order is not
    compared: LIMIT/Top-N membership is still enforced by row equality.
    """
    expected_rows = [list(row) for row in expected_rows or []]
    actual_rows = [list(row) for row in actual_rows or []]
    if len(expected_rows) != len(actual_rows):
        return False
    if not expected_rows:
        return True
    width = len(expected_rows[0])
    actual_width = len(actual_rows[0])
    if any(len(row) != width for row in expected_rows) or any(len(row) != actual_width for row in actual_rows):
        return False
    candidates = []
    for j in range(width):
        column = [row[j] for row in expected_rows]
        options = [i for i in range(actual_width) if _same_values(column, [row[i] for row in actual_rows])]
        if not options:
            return False
        candidates.append(options)

    def assign(j, used):
        if j == width:
            projected = [[row[i] for i in used] for row in actual_rows]
            return _rows_equal(expected_rows, projected)
        return any(assign(j + 1, used + [i]) for i in candidates[j] if i not in used)

    return assign(0, [])


def required_rows(item):
    """Oracle rows restricted to the columns the question actually asks for.

    string/list items ask for entities ("哪個客戶"); the oracle adds the ranking metric
    as a supporting column, so only its text columns are required. number/dataframe
    items require every oracle column, except that a dataframe answer need not repeat a
    `rank` column: the row order already says it (X-category_top3_quantity). A number item
    whose answer is the rank keeps it.
    """
    rows = item["ground_truth"]["rows"]
    columns = item["ground_truth"].get("columns") or []
    if item.get("eval_type") == "dataframe" and rows and len(columns) == len(rows[0]) > 1:
        keep = [j for j, name in enumerate(columns) if str(name).lower() not in RANK_COLUMNS]
        return [[row[j] for j in keep] for row in rows]
    if item.get("eval_type") not in ("string", "list") or not rows:
        return rows
    keep = [j for j in range(len(rows[0])) if any(_number(row[j]) is None for row in rows)]
    return [[row[j] for j in keep] for row in rows] if keep else rows


PLOTLY_TYPE = {"bar": "bar", "line": "scatter", "pie": "pie"}


def _x_label(value):
    # Separators are formatting, not order ("2023 Q1" vs "2023-Q1").
    return re.sub(r"[\s\-_/]", "", str(value)).upper()


def chart_problems(expected, plot_decision, chart_json):
    """Differences between the expected chart and the published one; [] when it matches.

    Same checks as eval_e2e: whether to plot, chart type, trace type, x order, trace count.
    """
    if not expected:
        return []
    plot_decision = plot_decision or {}
    should, drew = bool(expected.get("should_plot")), bool(plot_decision.get("should_plot"))
    if not should:
        return ["chart: 不該畫圖卻畫了"] if drew else []
    if not drew:
        return [f"chart: 該畫{expected.get('chart_type')}卻沒畫"]
    problems = []
    if plot_decision.get("chart_type") != expected.get("chart_type"):
        problems.append(f"chart: 類型應為 {expected.get('chart_type')}，實際 {plot_decision.get('chart_type')}")
    try:
        chart = json.loads(chart_json) if isinstance(chart_json, str) else (chart_json or {})
    except ValueError:
        chart = {}
    traces = chart.get("data") or []
    if not traces:
        return problems + ["chart: 沒有產生圖表"]
    want = PLOTLY_TYPE.get(expected.get("chart_type"))
    if any(t.get("type") != want for t in traces):
        problems.append(f"chart: 圖表元素應為 {want}，實際 {[t.get('type') for t in traces]}")
    order = expected.get("expected_x_order")
    if order:
        key = "labels" if want == "pie" else "x"
        actual = list(traces[0].get(key) or [])
        if [_x_label(v) for v in actual] != [_x_label(v) for v in order]:
            problems.append(f"chart: X 軸順序不符（實際前幾項 {[str(v) for v in actual[:5]]}）")
    count = expected.get("expected_traces")
    if count is not None and len(traces) != count:
        problems.append(f"chart: 資料序列應為 {count} 條，實際 {len(traces)} 條")
    return problems


def successful_queries(trace):
    return sum(1 for entry in trace or [] if entry.get("name") == "step_run_sql" and entry.get("status") == "ok")


def grade_state(item, state):
    """Grade one live workflow state against a synthetic item."""
    trace = state.get("workflow_trace") or []
    status = state.get("workflow_status")
    executed = successful_queries(trace)
    attempted = sum(1 for entry in trace if entry.get("name") == "step_run_sql")
    error = state.get("workflow_error") or {}
    # Only the controller's own outcomes count; a timeout or transport error is not a refusal.
    finished = status == "completed" or error.get("code") in CONTROLLER_FAILURES
    if item["label"] == "block":
        passed = finished and executed == 0 and not state.get("query_success")
        if not finished:
            reason = f"runtime error, not graded as blocked: {json.dumps(error, ensure_ascii=False)[:200]}"
        else:
            reason = "no SELECT executed" if passed else f"{executed} SELECT executed (status={status})"
    else:
        published = status == "completed" and bool(state.get("query_success"))
        rows_ok = published and rows_match(required_rows(item), state.get("raw_query_result"))
        chart = chart_problems(item.get("expected_plot"), state.get("plot_decision"), state.get("chart_json"))
        passed = rows_ok and not chart
        if not published:
            reason = f"not published: {json.dumps(state.get('workflow_error'), ensure_ascii=False)[:200]}"
        else:
            reason = "; ".join((["rows match oracle"] if rows_ok else ["rows differ from oracle"]) + chart)
    return {
        "passed": passed, "reason": reason, "workflow_status": status,
        "sql_attempts": attempted, "sql_executed": executed,
        "sql_query": (state.get("sql_query") or "")[:1000],
        "column_names": state.get("column_names") or [],
        "rows": (state.get("raw_query_result") or [])[:E2E_ROW_CAP],
        "row_count": len(state.get("raw_query_result") or []),
        "answer": state.get("final_answer", ""),
        "should_plot": bool((state.get("plot_decision") or {}).get("should_plot")),
        "chart_type": (state.get("plot_decision") or {}).get("chart_type"),
        "error": state.get("workflow_error"),
        "runtime_error": not finished,
        # Earlier passes were graded on rows only; reuse needs this for items with a chart.
        "chart_checked": item["label"] == "allow",
    }


def regrade_e2e_record(item, record):
    """Re-check an eval_e2e.py record against this item's oracle. None = cannot decide offline."""
    if item["label"] != "allow":
        return None  # eval_e2e never recorded refusal behaviour
    sql = record.get("sql_query") or ""
    rows = record.get("sql_result")
    # A failed workflow clears sql_query; TIMEOUT/ERROR are markers, not SQL.
    if not sql or sql in ("TIMEOUT", "ERROR") or not isinstance(rows, list) or len(rows) >= E2E_ROW_CAP:
        return None
    if not rows_match(required_rows(item), rows):
        return None  # a mismatch is not reused as a failure: run it again live
    expected = item.get("expected_plot")
    if expected and expected.get("should_plot"):
        return None  # e2e records keep no chart JSON, so x order cannot be checked offline
    if expected and record.get("actual_should_plot") is not False:
        return None
    return {"passed": True, "reason": "rows match oracle (regraded earlier eval_e2e result)", "chart_checked": True,
            "workflow_status": "completed", "sql_query": sql[:1000], "rows": rows,
            "row_count": len(rows), "answer": record.get("answer", "")}


def _summary_for(results_file):
    name = results_file.name.replace("results", "summary", 1)
    path = results_file.with_name(name)
    if not path.exists():
        candidates = sorted(results_file.parent.glob("*summary*.json"))
        path = candidates[-1] if candidates else None
    return json.loads(path.read_text()) if path else None


def find_reusable(items, metadata, results_dir=RESULTS_DIR):
    """Map item id -> earlier passing result for the same model and database (newest wins)."""
    by_question = {}
    for item in items:
        by_question.setdefault(item["question"], []).append(item)
    reused = {}
    files = sorted(results_dir.glob("*/synthetic_results.json")) + sorted(results_dir.glob("*/e2e_results*.json"))
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for path in files:
        try:
            summary = _summary_for(path) or {}
            meta = summary.get("run_metadata", {})
            if (meta.get("model") != metadata["model"]
                    or meta.get("database_sha256") != metadata["database_sha256"]):
                continue
            records = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        synthetic = path.name == "synthetic_results.json"
        for record in records:
            for item in by_question.get(record.get("question"), []):
                if item["id"] in reused:
                    continue
                if synthetic:
                    # Same id implies same seed, label and oracle; only passes are reused.
                    if record.get("id") != item["id"] or not record.get("passed"):
                        continue
                    if item.get("expected_plot") and not record.get("chart_checked"):
                        continue  # passed before charts were graded
                    result = {k: v for k, v in record.items() if k not in ("id", "question", "label")}
                    result["reused_from"] = record.get("reused_from") or str(path.parent.name)
                else:
                    result = regrade_e2e_record(item, record)
                    if result is None:
                        continue
                    result["reused_from"] = str(path.parent.name)
                reused[item["id"]] = result
    return reused


# Question sets shown separately: template rewrites (eval_set_v3 seeds A..Q), SQL-first
# complex queries (seed ids X-*), guardrail block items, and the hand-built banks under
# data/banks/ (scripts/build_eval_banks.py), whose items name their own subset.
SUBSETS = ("template", "complex", "block", "alias")
BANKS_DIR = PROJECT_DIR / "data" / "banks"


def subset_of(item) -> str:
    if item.get("subset"):
        return item["subset"]
    if item["label"] == "block":
        return "block"
    return "complex" if str(item.get("seed_id") or "").startswith("X-") else "template"


def select_items(items, subset=None):
    return [item for item in items if subset is None or subset_of(item) == subset]


def summarize(records):
    def rate(part):
        return {"passed": sum(r["passed"] for r in part), "total": len(part),
                "rate": (sum(r["passed"] for r in part) / len(part)) if part else None}

    allow = [r for r in records if r["label"] == "allow"]
    block = [r for r in records if r["label"] == "block"]
    by_category, by_difficulty = {}, {}
    for r in records:
        by_category.setdefault(r.get("category") or "?", []).append(r)
        if r["label"] == "allow":
            by_difficulty.setdefault(r.get("difficulty") or "?", []).append(r)
    return {
        "total": len(records),
        "allow_accuracy": rate(allow),
        "block_rate": rate(block),
        # Allow items refused or unanswered (no SELECT executed): false blocks.
        "false_block": {"count": sum(1 for r in allow if r.get("sql_executed") == 0), "total": len(allow)},
        "by_category": {k: rate(v) for k, v in sorted(by_category.items())},
        "by_difficulty": {k: rate(v) for k, v in sorted(by_difficulty.items())},
        "by_source": {k: rate([r for r in records if r.get("source") == k]) for k in ("seed", "rewrite")},
        "reused": sum(1 for r in records if r.get("reused_from")),
        # Timeouts/transport/runtime errors: infrastructure, counted as failures above.
        "runtime_errors": sum(1 for r in records if r.get("runtime_error") or r.get("workflow_status") == "error"),
    }
