"""Carry intent across turns while invalidating all previous result artifacts."""

import json
import unicodedata
from google.genai import types
from .tools import clear_query_state


GREETINGS = {"hello", "hi", "hey", "你好", "您好", "嗨", "哈囉", "哈啰", "早安", "午安", "晚安"}
GREETING_REPLY = "你好！我可以幫你查詢銷售資料。你想了解哪個地區、期間或產品的銷售情況？"


def start_turn(callback_context):
    state = callback_context.state
    previous = {}
    if state.get("workflow_status") == "completed" and state.get("query_success"):
        previous = {"question": state.get("user_question", ""), "sql": state.get("sql_query", "")}
    elif state.get("interaction_kind") in ("greeting", "refusal"):
        previous = json.loads(state.get("previous_query_context", "{}"))
    clear_query_state(state)
    parts = callback_context.user_content.parts if callback_context.user_content else []
    state["user_question"] = "\n".join(part.text for part in (parts or []) if part.text)
    state["previous_query_context"] = json.dumps(previous, ensure_ascii=False)
    state["sql_plan"] = None
    state["report"] = None
    state["sql_attempts"] = 0
    state["report_attempts"] = 0
    state["harness_stats"] = {}
    state["workflow_trace"] = []
    state["interaction_kind"] = "query"
    # Match the whole message, not a substring: "hello，請查營收" is a query.
    normalized = unicodedata.normalize("NFKC", state["user_question"]).casefold().strip(" \t\r\n.!?，。！？~～")
    if normalized in GREETINGS:
        state.update({"interaction_kind": "greeting", "workflow_status": "completed",
                      "answer": GREETING_REPLY, "final_answer": GREETING_REPLY})
        # ADK's callback return skips the agent body: no LLM or database call.
        return types.Content(role="model", parts=[types.Part(text=GREETING_REPLY)])
