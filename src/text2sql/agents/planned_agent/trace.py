"""Per-turn execution trace: what each agent/tool/step received and returned.

Names follow the project convention: agent_* = LLM, tool_* = model-called tool,
step_* = deterministic code the controller always runs. The trace is display
evidence only; it never feeds back into prompts or publication decisions.
"""

import json
import time

PREVIEW_ROWS = 20
MAX_TEXT = 4000


def _clip(value):
    """Keep the trace JSON-safe and small enough for session state."""
    if isinstance(value, str):
        return value if len(value) <= MAX_TEXT else value[:MAX_TEXT] + "…（已截斷）"
    if isinstance(value, dict):
        return {str(key): _clip(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clip(item) for item in value]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return str(value)


def rows_preview(columns, rows) -> dict:
    rows = rows or []
    return {"columns": list(columns or []), "rows": rows[:PREVIEW_ROWS], "row_count": len(rows),
            "rows_shown": min(len(rows), PREVIEW_ROWS)}


def kind_of(name: str) -> str:
    return name.split("_", 1)[0] if name.split("_", 1)[0] in ("agent", "tool", "step") else "step"


class Trace:
    def __init__(self):
        self.entries: list[dict] = []

    def start(self, name: str, attempt: int, inputs: dict) -> dict:
        entry = {"name": name, "kind": kind_of(name), "attempt": attempt, "status": "running",
                 "input": _clip(inputs), "output": None, "ms": None, "_t": time.monotonic()}
        self.entries.append(entry)
        return entry

    def finish(self, entry: dict, status: str, output) -> None:
        entry["status"] = status  # ok | rejected | error | skipped
        entry["output"] = _clip(output)
        entry["ms"] = int((time.monotonic() - entry.pop("_t")) * 1000)

    def add(self, name: str, attempt: int, status: str, inputs: dict, output) -> None:
        """Record a step that already ran inside another step (no own timing)."""
        self.entries.append({"name": name, "kind": kind_of(name), "attempt": attempt, "status": status,
                             "input": _clip(inputs), "output": _clip(output), "ms": None})

    def snapshot(self) -> list[dict]:
        """A JSON-serializable copy for the ADK state delta."""
        return json.loads(json.dumps(
            [{key: value for key, value in entry.items() if key != "_t"} for entry in self.entries],
            ensure_ascii=False,
        ))


def record_chart(trace: Trace, attempt: int, spec, state) -> None:
    """step_draw_chart runs inside step_publish; summarize what it actually drew."""
    inputs = {"chart_type": spec.chart_type, "x_column": spec.x_column,
              "y_column": spec.y_column, "color_column": spec.color_column}
    if not state.get("chart_json"):
        trace.add("step_draw_chart", attempt, "skipped", inputs, {"note": "chart_type=none，不畫圖"})
        return
    figure = json.loads(state["chart_json"])
    traces = []
    for item in figure.get("data", []):
        labels = item.get("x") if item.get("x") is not None else item.get("labels")
        traces.append({"type": item.get("type"), "name": item.get("name"),
                       "x": labels[:PREVIEW_ROWS] if isinstance(labels, list) else []})
    trace.add("step_draw_chart", attempt, "ok", inputs, {"plotly_traces": traces})
