"""Strong-model path: one Codex Harness invocation, with hard application gates."""

from google.adk.agents import BaseAgent
from google.adk.events import Event, EventActions
from google.genai import types

from .context import start_turn
from .codex_backend import run_codex
from .tools import step_publish
from .trace import Trace, record_chart, rows_preview


class CodexSalesAgent(BaseAgent):
    """ADK is only the UI/session adapter; it does not orchestrate reasoning."""

    async def _run_async_impl(self, ctx):
        trace = Trace()
        question = ctx.session.state["user_question"]
        entry = trace.start("agent_codex", 1, {
            "user_question": question,
            "previous_query_context": ctx.session.state["previous_query_context"],
        })
        try:
            spec, evidence, stats = await run_codex(question, ctx.session.state["previous_query_context"])
        except Exception as exc:
            trace.finish(entry, "error", {"error": str(exc)})
            yield self.traced(ctx, trace)
            raise
        trace.finish(entry, "ok", {"report": spec.model_dump(), "harness_stats": stats})
        # The MCP receipt holds only the final call; earlier calls are counted in query_calls.
        query = evidence.get("query_result") or {}
        trace.add("tool_query_sales", stats.get("query_calls") or 1,
                  "ok" if evidence.get("query_success") else "error",
                  {"sql": evidence.get("sql_query", ""), "query_calls": stats.get("query_calls")},
                  rows_preview(query.get("columns"), query.get("rows")) if evidence.get("query_success")
                  else {"error": evidence.get("query_error")})
        state = dict(ctx.session.state)
        state.update(evidence)
        state["harness_stats"] = stats
        state["workflow_trace"] = trace.snapshot()
        yield Event(invocation_id=ctx.invocation_id, author=self.name,
                    actions=EventActions(state_delta=state))
        entry = trace.start("step_publish", 1, spec.model_dump())
        try:
            step_publish(spec, state, require_review=False)  # receipt is the evidence; no reviewer on this path
        except Exception as exc:
            trace.finish(entry, "error", {"error": str(exc)})
            yield self.traced(ctx, trace)
            raise
        trace.finish(entry, "ok", {"final_answer": state["final_answer"],
                                   "chart_generated": state["chart_json"] is not None})
        record_chart(trace, 1, spec, state)
        state["workflow_trace"] = trace.snapshot()
        yield Event(
            invocation_id=ctx.invocation_id, author=self.name,
            actions=EventActions(state_delta=state),
            content=types.Content(role="model", parts=[types.Part(text=state["final_answer"])]),
        )


    def traced(self, ctx, trace):
        return Event(invocation_id=ctx.invocation_id, author=self.name,
                     actions=EventActions(state_delta={"workflow_trace": trace.snapshot()}))


root_agent = CodexSalesAgent(
    name="root_agent", description="One Codex reasoning agent with bounded sales-data access.",
    before_agent_callback=start_turn,
)
