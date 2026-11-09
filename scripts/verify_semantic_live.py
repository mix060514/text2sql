"""Live Qwen semantic gate checks against unchanged SQLite, including fault injection."""
import asyncio
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import InMemoryRunner
from google.genai import types
from text2sql.agents.planned_agent.agent import root_agent, model
from text2sql.agents.planned_agent.tools import DB_PATH, read_sales
from text2sql.config import ANALYSIS_MODEL
from text2sql.runtime import run_question, stream_turn

QUANTITY = 'SELECT SUM("Quantity") AS Units FROM sales_data'
ORDERS = 'SELECT COUNT(DISTINCT "Order ID") AS Orders FROM sales_data'
WRONG = 'SELECT COUNT(DISTINCT "Order ID") AS Units FROM sales_data'


class InjectFirstSQL(BaseLlm):
    delegate: BaseLlm
    calls: int = 0

    async def generate_content_async(self, llm_request, stream=False):
        self.calls += 1
        if self.calls == 1:
            yield LlmResponse(content=types.Content(role="model", parts=[types.Part(text=json.dumps({"sql": WRONG}))]))
        else:
            async for response in self.delegate.generate_content_async(llm_request, stream=stream):
                yield response


def record(name, state, oracle_sql, started):
    expected = read_sales(oracle_sql)["rows"]
    passed = (state.get("workflow_status") == "completed" and
              state.get("raw_query_result") == expected and
              state.get("semantic_review", {}).get("accepted") is True)
    result = {"case": name, "passed": passed, "seconds": round(time.monotonic() - started, 2),
              "oracle_sql": oracle_sql, "expected_rows": expected}
    for key in ("user_question", "previous_query_context", "workflow_status", "workflow_error",
                "sql_query", "column_names", "raw_query_result", "semantic_review",
                "sql_feedback", "sql_attempts", "report_attempts", "final_answer"):
        result[key] = state.get(key)
    return result


async def main():
    results = []
    for name, question, oracle in (("quantity", "總銷售量是多少？", QUANTITY),
                                    ("distinct orders", "不重複的訂單總數是多少？", ORDERS),
                                    ("wrong metric injection", "總銷售量是多少？", QUANTITY)):
        agent = root_agent.clone()
        if name == "wrong metric injection":
            agent.sub_agents[0].model = InjectFirstSQL(model="inject_once_then_qwen", delegate=model())
        runner = InMemoryRunner(agent=agent, app_name="semantic_live")
        started = time.monotonic()
        state = await run_question(runner, question)
        result = record(name, state, oracle, started)
        if name == "wrong metric injection":
            result["injected_sql"] = WRONG
            result["injected_rows"] = read_sales(WRONG)["rows"]
            result["passed"] = (result["passed"] and state["sql_attempts"] >= 2 and
                                "Semantic review rejected SQL:" in state.get("sql_feedback", ""))
        results.append(result)
        print(f"{name}: {result['passed']}", flush=True)
    runner = InMemoryRunner(agent=root_agent.clone(), app_name="semantic_followup")
    await runner.session_service.create_session(app_name=runner.app_name, user_id="live", session_id="followup")
    for question, oracle in (("南美總銷售量是多少？", QUANTITY + " WHERE Region = 'LATAM'"),
                             ("那北美呢？", QUANTITY + " WHERE Region = 'North America'")):
        started = time.monotonic()
        async for _ in stream_turn(runner, user_id="live", session_id="followup",
                                   new_message=types.Content(role="user", parts=[types.Part(text=question)])):
            pass
        session = await runner.session_service.get_session(app_name=runner.app_name, user_id="live", session_id="followup")
        result = record(question, session.state, oracle, started)
        results.append(result)
        print(f"{question}: {result['passed']}", flush=True)
    report = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "model": ANALYSIS_MODEL,
              "database_sha256": hashlib.sha256(DB_PATH.read_bytes()).hexdigest(),
              "scope": "Live Qwen planner/reviewer/report except explicitly injected first wrong SQL; SQLite value oracle, no LLM judge.",
              "passed": sum(r["passed"] for r in results), "total": len(results), "results": results}
    (PROJECT / "docs/semantic-qwen27b.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return 0 if all(r["passed"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
