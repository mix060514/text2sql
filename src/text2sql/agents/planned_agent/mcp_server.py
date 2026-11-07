"""A per-request read-only business tool, with an application-owned receipt."""

import json
import os
from pathlib import Path
import threading

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from text2sql.config import MAX_QUERY_CALLS
from .tools import clear_query_state, step_run_sql


class QueryRecorder:
    def __init__(self, receipt: Path):
        self.receipt = receipt
        self.calls = 0
        self.lock = threading.Lock()

    def query(self, sql: str) -> dict:
        with self.lock:
            self.calls += 1
            state = {}
            if self.calls > MAX_QUERY_CALLS:
                clear_query_state(state)
                result = {"success": False, "error": {"code": "QUERY_BUDGET", "message": "Sales query budget exhausted."}}
                state["query_error"] = result["error"]
            else:
                result = step_run_sql(sql, state)
            self.receipt.write_text(json.dumps({"state": state, "query_calls": self.calls}, ensure_ascii=False))
            return result


def main():
    recorder = QueryRecorder(Path(os.environ["TEXT2SQL_RECEIPT"]))
    server = FastMCP("sales")

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=False))
    def tool_query_sales(sql: str) -> dict:
        """Execute one read-only SQLite SELECT against sales_data.

        Returns actual columns and all accepted rows, or a structured error.
        Empty rows with success=true is a valid empty result. Results are bounded;
        narrow or aggregate the query if the result is too large. Never writes data.
        """
        return recorder.query(sql)

    server.run(transport="stdio")


if __name__ == "__main__":
    main()
