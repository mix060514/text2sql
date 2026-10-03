"""Carry intent across turns while invalidating all previous result artifacts."""

import json
from .tools import clear_query_state


def start_turn(callback_context):
    state = callback_context.state
    previous = {}
    if state.get("workflow_status") == "completed":
        previous = {"question": state.get("user_question", ""), "sql": state.get("sql_query", "")}
    clear_query_state(state)
    parts = callback_context.user_content.parts if callback_context.user_content else []
    state["user_question"] = "\n".join(part.text for part in (parts or []) if part.text)
    state["previous_query_context"] = json.dumps(previous, ensure_ascii=False)
    state["sql_plan"] = None
    state["report"] = None
