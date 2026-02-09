import sqlite3
import sqlparse
import logging

from pathlib import Path
from google.adk.tools import ToolContext

# Resolve project root relative to this file:
# .../src/text2sql/agents/planned_agent/tools.py -> .../text2sql (project root)
# parents[0]=planned_agent, [1]=agents, [2]=text2sql, [3]=src, [4]=project_root
PROJECT_ROOT = Path(__file__).resolve().parents[4]
DB_PATH = str(PROJECT_ROOT / "data" / "global_sales_data.sqlite")

logger = logging.getLogger(__name__)


def check_sql_syntax(tool_context: ToolContext) -> str:
    """
    Checks if the SQL syntax is valid using sqlparse.
    Returns "Valid" if no errors are detected, otherwise returns an error message.
    """
    try:
        sql_query = tool_context.state["sql_query"]
        parsed = sqlparse.parse(sql_query)
        if not parsed:
            return "Invalid SQL: Empty query"
        # minimal check: verify we have at least one statement
        if len(parsed) == 0:
            return "Invalid SQL: No statements found"

        return "Valid"
    except Exception as e:
        return f"Syntax Error: {str(e)}"


def _clean_sql(raw: str) -> str:
    """Strip markdown code fences and leading/trailing whitespace from SQL."""
    import re

    # Remove ```sql ... ``` or ``` ... ```
    cleaned = re.sub(r"^```(?:sql)?\s*\n?", "", raw.strip())
    cleaned = re.sub(r"\n?```\s*$", "", cleaned)
    return cleaned.strip()


def execute_sql(tool_context: ToolContext) -> str:
    """
    Executes the SQL query stored in state['sql_query'] against the SQLite database.
    Reads SQL directly from state to avoid LLM modifying the query during tool call.

    Returns:
        The query results formatted as a markdown table, or an error message.
    """
    import pandas as pd

    sql_query = _clean_sql(tool_context.state.get("sql_query", ""))
    if not sql_query:
        return "Error: No SQL query found in state."

    try:
        with sqlite3.connect(DB_PATH) as con:
            df = pd.read_sql_query(sql_query, con)

        # Capture column names
        column_names = list(df.columns)

        # Store raw results (list of lists/tuples) and column names in state
        if hasattr(tool_context, "state"):
            # Convert dataframe to list of values for consistency with previous state usage
            # best to convert to list of lists
            tool_context.state["raw_query_result"] = df.values.tolist()
            tool_context.state["column_names"] = column_names

        # Limit to 50 rows to save context
        if len(df) > 50:
            df = df.head(50)

        return df.to_markdown(index=False)

    except Exception as e:
        if hasattr(tool_context, "state"):
            tool_context.state["raw_query_result"] = []
            if "column_names" not in tool_context.state:
                tool_context.state["column_names"] = []
        return f"Execution Error: {str(e)}"


def render_chart(
    chart_type: str,
    data: list[dict],
    x_col: str,
    y_col: str,
    color_col: str = None,
    title: str = "",
) -> dict:
    """
    Renders a chart using Plotly Express and returns base64 encoded PNG image.

    Args:
        chart_type: Type of chart - "bar", "line", or "pie"
        data: List of dictionaries with column names as keys
              e.g., [{"Region": "North America", "Sales": 1000}, ...]
        x_col: Column name for X-axis (or names for Pie)
        y_col: Column name for Y-axis (or values for Pie)
        color_col: Optional column name for grouping/color
        title: Optional chart title

    Returns:
        dict with "success" boolean and "image" (base64 string) or "error" message
    """
    import pandas as pd
    import plotly.express as px
    import base64

    try:
        if not data:
            return {"success": False, "error": "No data provided for charting."}

        df = pd.DataFrame(data)

        # Basic validation
        if x_col not in df.columns or y_col not in df.columns:
            return {
                "success": False,
                "error": f"Columns {x_col} or {y_col} not found in data. Available: {list(df.columns)}",
            }

        fig = None

        if chart_type == "bar":
            # Bar chart with optional color grouping
            fig = px.bar(
                df,
                x=x_col,
                y=y_col,
                color=color_col if color_col and color_col in df.columns else None,
                barmode="group",
                text_auto=True,
                template="plotly_white",
            )
            fig.update_traces(textposition="outside")

        elif chart_type == "line":
            # Line chart with optional color series
            fig = px.line(
                df,
                x=x_col,
                y=y_col,
                color=color_col if color_col and color_col in df.columns else None,
                markers=True,
                template="plotly_white",
            )

        elif chart_type == "pie":
            # Pie chart (ignores color_col usually, uses x_col for slices)
            fig = px.pie(
                df,
                names=x_col,
                values=y_col,
                hole=0.3,  # Donut style
                template="plotly_white",
            )
            fig.update_traces(textposition="inside", textinfo="percent+label")

        else:
            return {"success": False, "error": f"Unknown chart type: {chart_type}"}

        if title:
            fig.update_layout(title_text=title, title_x=0.5)

        # Adjust layout for better readability
        fig.update_layout(
            margin=dict(l=20, r=20, t=40, b=20),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )

        # Convert to Plotly JSON
        try:
            chart_json = fig.to_json()
            return {"success": True, "chart_json": chart_json}
        except Exception as img_err:
            return {
                "success": False,
                "error": f"Chart serialization failed: {str(img_err)}",
            }

    except Exception as e:
        logger.error(f"Chart rendering error: {e}")
        return {"success": False, "error": str(e)}
