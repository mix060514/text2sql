"""Small-model path: fixed stages with bounded, error-driven repair."""

from google.adk.agents import BaseAgent, LlmAgent
from google.adk.events import Event, EventActions
from google.adk.models.lite_llm import LiteLlm
from google.genai import types
from pydantic import ValidationError

from text2sql.config import (
    LLAMA_CPP_MODEL, LLAMA_CPP_API_BASE, LLAMA_CPP_API_KEY,
    REPORT_ATTEMPTS, SQL_ATTEMPTS,
)
from .contracts import AnswerSubmission, SQLPlan
from .prompts import DOMAIN, REPORT_RULES
from .tools import complete_answer, execute_query
from .context import start_turn


def model():
    return LiteLlm(model=LLAMA_CPP_MODEL, api_base=LLAMA_CPP_API_BASE,
                   api_key=LLAMA_CPP_API_KEY, timeout=120)


sql_planner = LlmAgent(
    name="sql_planner", model=model(), output_schema=SQLPlan, output_key="sql_plan",
    disallow_transfer_to_parent=True, disallow_transfer_to_peers=True,
    include_contents="none",
    instruction=DOMAIN + """
你只負責產生一個 SQLite SELECT，不負責執行或回答。
不要呼叫工具，不要用 Markdown 或加上說明，依輸出 schema 提交 SQL。
重試只能修正原問題的 SQL；不得把問題換成訂單總數或其他未要求的統計。
使用者問題：{user_question}
前一個成功回合的問題與 SQL（僅用於理解追問，必須重新查詢）：{previous_query_context}
上一個錯誤（若有）：{sql_feedback}
""",
    generate_content_config=types.GenerateContentConfig(temperature=0.1, max_output_tokens=2048),
)

report_writer = LlmAgent(
    name="report_writer", model=model(), output_schema=AnswerSubmission, output_key="report",
    disallow_transfer_to_parent=True, disallow_transfer_to_peers=True,
    include_contents="none",
    instruction=REPORT_RULES + """
只根據以下查詢結果生成 answer 與圖表規格，依輸出 schema 提交。
使用者問題：{user_question}
追問上下文：{previous_query_context}
實際 SQL：{sql_query}
實際欄位：{column_names}
實際資料：{raw_query_result}
上一個輸出錯誤（若有）：{report_feedback}
""",
    generate_content_config=types.GenerateContentConfig(temperature=0.1, max_output_tokens=4096),
)


class Text2SQLWorkflow(BaseAgent):
    """The controller owns ordering and success; models own SQL and explanation."""

    def update(self, ctx, delta, text=None):
        return Event(
            invocation_id=ctx.invocation_id, author=self.name,
            actions=EventActions(state_delta=delta),
            content=types.Content(role="model", parts=[types.Part(text=text)]) if text else None,
        )

    async def _run_async_impl(self, ctx):
        feedback = ""
        for attempt in range(SQL_ATTEMPTS):
            yield self.update(ctx, {"sql_plan": None, "sql_feedback": feedback, "sql_attempts": attempt + 1})
            try:
                async for event in self.sub_agents[0].run_async(ctx):
                    yield event
                plan = SQLPlan.model_validate(ctx.session.state.get("sql_plan"))
                state = {}
                result = execute_query(plan.sql, state)  # mandatory execution, direct call
                yield self.update(ctx, state)
                if result["success"]:
                    break
                feedback = str(result["error"])
            except ValidationError as exc:
                feedback = f"Invalid SQLPlan: {exc}"
        else:
            yield self.failure(ctx, "SQL_ATTEMPTS_EXHAUSTED", feedback)
            return

        feedback = ""
        for attempt in range(REPORT_ATTEMPTS):
            yield self.update(ctx, {"report": None, "report_feedback": feedback, "report_attempts": attempt + 1})
            try:
                async for event in self.sub_agents[1].run_async(ctx):
                    yield event
                spec = AnswerSubmission.model_validate(ctx.session.state.get("report"))
                state = dict(ctx.session.state)
                complete_answer(spec, state)  # validates and renders directly
                yield self.update(ctx, state, state["final_answer"])
                return
            except (ValidationError, ValueError, TypeError) as exc:
                feedback = f"Invalid report or chart: {exc}"
        yield self.failure(ctx, "REPORT_ATTEMPTS_EXHAUSTED", feedback)

    def failure(self, ctx, code, detail):
        message = "這次未能取得可確認的分析結果，請重新提問。"
        return self.update(ctx, {
            "workflow_status": "failed", "workflow_error": {"code": code, "message": detail[:500]},
            "answer": message, "final_answer": message, "chart_json": None, "chart_data": None,
            "plot_decision": {"should_plot": False, "chart_type": None},
        }, message)


root_agent = Text2SQLWorkflow(
    name="root_agent", description="Fixed SQL → execute → report workflow for Qwen 27B.",
    sub_agents=[sql_planner, report_writer], before_agent_callback=start_turn,
)
