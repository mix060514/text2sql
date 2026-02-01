import sqlite3
import sqlparse
import logging

from pathlib import Path

# Resolve project root relative to this file:
# .../src/text2sql/agents/planned_agent/tools.py -> .../text2sql (project root)
# parents[0]=planned_agent, [1]=agents, [2]=text2sql, [3]=src, [4]=project_root
PROJECT_ROOT = Path(__file__).resolve().parents[4]
DB_PATH = str(PROJECT_ROOT / "data" / "global_sales_data.sqlite")

logger = logging.getLogger(__name__)


def check_sql_syntax(sql_query: str) -> str:
    """
    Checks if the SQL syntax is valid using sqlparse.
    Returns "Valid" if no errors are detected, otherwise returns an error message.
    """
    try:
        parsed = sqlparse.parse(sql_query)
        if not parsed:
            return "Invalid SQL: Empty query"
        # minimal check: verify we have at least one statement
        if len(parsed) == 0:
            return "Invalid SQL: No statements found"

        return "Valid"
    except Exception as e:
        return f"Syntax Error: {str(e)}"


def execute_sql(sql_query: str) -> str:
    """
    Executes the SQL query against the SQLite database.
    Returns the results as a string or an error message.
    """
    try:
        con = sqlite3.connect(DB_PATH)
        cur = con.cursor()
        cur.execute(sql_query)
        results = cur.fetchall()
        con.close()
        return str(results)
    except Exception as e:
        return f"Execution Error: {str(e)}"
