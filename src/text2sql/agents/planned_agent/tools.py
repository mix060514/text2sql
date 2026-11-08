"""Deterministic query execution, chart validation, and answer publication."""

import json
import math
import sqlite3
import time
from numbers import Real
from pathlib import Path

import sqlparse
from pydantic import ValidationError

from text2sql.config import MAX_QUERY_ROWS, MAX_RESULT_BYTES, QUERY_TIMEOUT_SECONDS
from .contracts import AnswerSubmission, SQLPlan, SemanticReview

DB_PATH = Path(__file__).resolve().parents[4] / "data" / "global_sales_data.sqlite"

# Authorization is enforced by SQLite, independently of model instructions.
SQL_FUNCTIONS = {
    "abs", "avg", "cast", "ceil", "ceiling", "coalesce", "count", "date",
    "datetime", "dense_rank", "floor", "ifnull", "instr", "julianday", "lag",
    "lead", "length", "lower", "ltrim", "max", "min", "nullif", "printf",
    "rank", "replace", "round", "row_number", "rtrim", "strftime", "substr",
    "substring", "sum", "time", "total", "trim", "unixepoch", "upper",
}


class QueryError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def read_sales(sql: str, db_path: Path = DB_PATH) -> dict:
    """Execute one bounded SELECT; SQLite validates real syntax and access."""
    try:
        sql = SQLPlan(sql=sql).sql.strip()
    except ValidationError as exc:
        raise QueryError("INVALID_ARGUMENTS", str(exc)) from exc
    statements = sqlparse.parse(sql)
    if len(statements) != 1 or statements[0].get_type() != "SELECT":
        raise QueryError("READ_ONLY", "Provide exactly one SELECT (WITH ... SELECT is allowed).")

    read_business_table = False

    def authorize(action, table, column, database, source):
        nonlocal read_business_table
        if action in (sqlite3.SQLITE_SELECT, sqlite3.SQLITE_RECURSIVE):
            return sqlite3.SQLITE_OK
        # SQLite's optimized COUNT(*) emits READ with a null database name.
        if action == sqlite3.SQLITE_READ and database in ("main", None) and table == "sales_data":
            read_business_table = True
            return sqlite3.SQLITE_OK
        if action == sqlite3.SQLITE_FUNCTION and (column or "").lower() in SQL_FUNCTIONS:
            return sqlite3.SQLITE_OK
        return sqlite3.SQLITE_DENY

    connection = None
    try:
        connection = sqlite3.connect(Path(db_path).resolve().as_uri() + "?mode=ro", uri=True)
        connection.execute("PRAGMA query_only = ON")
        connection.execute("PRAGMA trusted_schema = OFF")
        connection.set_authorizer(authorize)
        deadline = time.monotonic() + QUERY_TIMEOUT_SECONDS
        connection.set_progress_handler(lambda: int(time.monotonic() > deadline), 1000)
        cursor = connection.execute(sql)
        if not read_business_table:
            raise QueryError("NO_SALES_DATA", "The query must read sales_data; constants are not sales results.")
        columns = [item[0] for item in cursor.description]
        if len(columns) != len(set(columns)):
            raise QueryError("DUPLICATE_COLUMNS", "Give each result column a unique alias.")
        rows = [list(row) for row in cursor.fetchmany(MAX_QUERY_ROWS + 1)]
        if len(rows) > MAX_QUERY_ROWS:
            raise QueryError("ROW_LIMIT", f"Result exceeds {MAX_QUERY_ROWS} rows. Aggregate or narrow the query; no partial result was accepted.")
        result = {"success": True, "sql": sql, "columns": columns, "rows": rows, "row_count": len(rows)}
        try:
            size = len(json.dumps(result, ensure_ascii=False, allow_nan=False).encode("utf-8"))
        except (TypeError, ValueError) as exc:
            raise QueryError("RESULT_TYPE", "Result must contain finite, JSON-compatible values.") from exc
        if size > MAX_RESULT_BYTES:
            raise QueryError("RESULT_SIZE", f"Result exceeds {MAX_RESULT_BYTES} bytes. Aggregate or narrow the query.")
        return result
    except sqlite3.Error as exc:
        code = "QUERY_TIMEOUT" if str(exc) == "interrupted" else "SQL_ERROR"
        raise QueryError(code, str(exc)) from exc
    finally:
        if connection is not None:
            connection.close()


def clear_query_state(state) -> None:
    """Invalidate the previous result, answer, and chart before a new attempt."""
    state.update({
        "semantic_review": None, "reviewed_query": None,
        "query_success": False, "sql_query": "", "raw_query_result": [],
        "column_names": [], "query_result": {}, "query_error": None,
        "plot_decision": {"should_plot": False, "chart_type": None},
        "chart_json": None, "chart_data": None, "answer": "", "final_answer": "",
        "workflow_status": "running", "workflow_error": None,
    })


def step_run_sql(sql: str, state) -> dict:
    """Query directly; expose structured errors to whichever model/harness is used."""
    clear_query_state(state)
    state["sql_query"] = sql if isinstance(sql, str) else ""
    try:
        result = read_sales(sql)
    except QueryError as exc:
        error = {"code": exc.code, "message": str(exc)}
        state["query_error"] = error
        return {"success": False, "error": error}
    state.update({
        "query_success": True, "sql_query": result["sql"], "raw_query_result": result["rows"],
        "column_names": result["columns"], "query_result": result,
    })
    return result


def numeric_columns(columns: list[str], rows: list[list]) -> list[str]:
    return [column for i, column in enumerate(columns) if rows and all(
        isinstance(row[i], Real) and not isinstance(row[i], bool) and math.isfinite(row[i])
        for row in rows
    )]


def step_draw_chart(spec: AnswerSubmission, columns: list[str], rows: list[list]) -> str:
    """Validate against the full actual result, then return Plotly JSON."""
    import pandas as pd
    import plotly.express as px

    if len(rows) < 2:
        raise ValueError("Charts require at least two rows.")
    for column in (spec.x_column, spec.y_column, spec.color_column):
        if column and column not in columns:
            raise ValueError(f"Unknown chart column {column!r}. Available: {columns}")
    if spec.x_column == spec.y_column:
        raise ValueError("Choose distinct x and y columns.")
    if spec.y_column not in numeric_columns(columns, rows):
        raise ValueError("The y column must contain finite numeric values in every row.")
    data = pd.DataFrame(rows, columns=columns)
    if spec.chart_type == "pie":
        if (data[spec.y_column] < 0).any() or data[spec.y_column].sum() <= 0:
            raise ValueError("Pie values must be nonnegative with a positive total.")
        figure = px.pie(data, names=spec.x_column, values=spec.y_column, hole=0.3)
    else:
        draw = px.line if spec.chart_type == "line" else px.bar
        options = {"markers": True} if spec.chart_type == "line" else {"barmode": "group"}
        figure = draw(data, x=spec.x_column, y=spec.y_column, color=spec.color_column or None, **options)
        for trace in figure.data:
            trace.x = [str(value) for value in trace.x]
        labels = list(dict.fromkeys(str(value) for value in data[spec.x_column]))
        figure.update_xaxes(type="category", categoryorder="array", categoryarray=labels)
    figure.update_layout(template="plotly_white", margin={"l": 20, "r": 20, "t": 40, "b": 20})
    return figure.to_json()


def result_table(spec: AnswerSubmission, columns: list[str], rows: list[list]) -> str:
    """Present every accepted row without asking a model to copy the data."""
    from tabulate import tabulate

    if not rows:
        return "查詢結果：0 筆符合條件的資料。"
    labels = {spec.x_column, spec.color_column}

    def display(column, value):
        if value is None:
            text = "—"
        elif column in labels:
            text = str(value)
        elif isinstance(value, int):
            text = f"{value:,}"
        elif isinstance(value, float):
            text = f"{value:,.2f}"
        else:
            text = str(value)
        return text.replace("|", "\\|").replace("\n", " ")

    cells = [[display(column, value) for column, value in zip(columns, row)] for row in rows]
    headers = [column.replace("|", "\\|").replace("\n", " ") for column in columns]
    return tabulate(cells, headers=headers, tablefmt="pipe", disable_numparse=True)


def validate_review(state) -> None:
    """Qwen path: the published result must be the one the reviewer accepted."""
    review = SemanticReview.model_validate(state.get("semantic_review"))
    result = state.get("query_result")
    if not review.accepted or not result or state.get("reviewed_query") != result:
        raise ValueError("Explicit semantic acceptance of this executed result is required.")
    if (result.get("sql") != state.get("sql_query") or
            result.get("columns") != state.get("column_names") or
            result.get("rows") != state.get("raw_query_result")):
        raise ValueError("Published data must match the reviewed execution.")


def step_publish(spec: AnswerSubmission, state) -> dict:
    """Hard success gate: only executed data and a validated chart can be published.

    It also requires an explicit semantic acceptance bound to this execution.
    """
    if not state.get("query_success"):
        raise ValueError("A successful query in this turn is required before publishing.")
    validate_review(state)
    columns, rows = state["column_names"], state["raw_query_result"]
    if spec.chart_type == "none" and len(rows) >= 2 and len(columns) >= 2 and numeric_columns(columns, rows):
        raise ValueError("Multiple rows with a numeric measure require a chart.")
    chart_json = step_draw_chart(spec, columns, rows) if spec.chart_type != "none" else None
    final_answer = spec.answer + "\n\n" + result_table(spec, columns, rows)
    state.update({
        "answer": final_answer, "final_answer": final_answer, "workflow_status": "completed",
        "workflow_error": None,
        "plot_decision": {"should_plot": chart_json is not None,
                          "chart_type": None if chart_json is None else spec.chart_type,
                          "x_column": spec.x_column or None, "y_column": spec.y_column or None,
                          "color_column": spec.color_column or None},
        "chart_json": chart_json,
        "chart_data": {"chart_type": spec.chart_type, "x_column": spec.x_column,
                       "y_column": spec.y_column, "color_column": spec.color_column or None,
                       "data_rows": len(rows)} if chart_json else None,
    })
    return {"success": True, "answer": final_answer, "chart_generated": chart_json is not None}
