import streamlit as st
import sys
import os
import pathlib
import json
import asyncio
import uuid
import time
from typing import List, Dict

# Setup path to import local src
# Assuming web_ui/pages/1_Chat.py -> web_ui/pages -> web_ui -> project_root -> src
project_root = pathlib.Path(__file__).parent.parent.parent.resolve()
sys.path.append(str(project_root / "src"))
sys.path.append(str(project_root / "eval"))

try:
    from google.adk.runners import InMemoryRunner
    from google.genai import types
    from text2sql.agents.planned_agent.agent import root_agent
except ImportError as e:
    st.error(f"Failed to import agent or dependencies: {e}")
    st.stop()

st.set_page_config(page_title="Chat with Agent", page_icon="💬", layout="wide")

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = (
        f"web_chat_{int(time.time()*1000)}_{uuid.uuid4().hex[:6]}"
    )

if "runner" not in st.session_state:
    st.session_state.runner = InMemoryRunner(agent=root_agent, app_name="text2sql_chat")

    # Initialize session in ADK
    async def init_session():
        await st.session_state.runner.session_service.create_session(
            app_name=st.session_state.runner.app_name,
            user_id="web_user",
            session_id=st.session_state.session_id,
        )

    asyncio.run(init_session())

# Sidebar settings
st.sidebar.title("Settings")
show_subagent = st.sidebar.checkbox("Show Sub-agent Output", value=True)


# Load Ground Truth for Suggestions
@st.cache_data
def load_suggestions():
    data_path = project_root / "data" / "eval_set_v2.jsonl"
    questions = []
    if data_path.exists():
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if "question" in data:
                        questions.append(data["question"])
                except:
                    pass
    return questions


suggestions = load_suggestions()
selected_suggestion = st.sidebar.selectbox(
    "Suggested Questions", [""] + suggestions[:20]
)  # Limit to 20 for UI
send_suggestion = st.sidebar.button("Send Suggestion")

# Display Messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "subagent_output" in msg and show_subagent:
            with st.expander("Sub-agent Details"):
                st.text(msg["subagent_output"])

# Chat Input
chat_prompt = st.chat_input("Ask a question...")

prompt = None
if chat_prompt:
    prompt = chat_prompt
elif send_suggestion and selected_suggestion:
    prompt = selected_suggestion

if prompt:

    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process Agent Response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        subagent_placeholder = st.empty()

        full_response = ""
        subagent_logs = []

        def format_event(event) -> str:
            """Formats an ADK event into a readable markdown string."""
            log_lines = []

            # Extract basic info
            author = getattr(event, "author", "unknown")
            # header = f"**[{author}]**"
            # Simplified header to avoid noise, use author only if meaningful

            content = getattr(event, "content", None)
            if content and hasattr(content, "parts"):
                for part in content.parts:
                    # Text Content
                    if hasattr(part, "text") and part.text:
                        log_lines.append(f"**[{author}]** {part.text}")

                    # Function Call
                    if hasattr(part, "function_call") and part.function_call:
                        fc = part.function_call
                        args = getattr(fc, "args", {})
                        log_lines.append(f"**[{author}]** 🛠️ **Tool Call**: `{fc.name}`")
                        log_lines.append(
                            f"```json\n{json.dumps(args, ensure_ascii=False, indent=2)}\n```"
                        )

                    # Function Response
                    if hasattr(part, "function_response") and part.function_response:
                        fr = part.function_response
                        response = getattr(fr, "response", {})
                        log_lines.append(
                            f"**[{author}]** 📋 **Tool Result**: `{fr.name}`"
                        )
                        # Truncate
                        res_str = json.dumps(response, ensure_ascii=False, indent=2)
                        if len(res_str) > 500:
                            res_str = res_str[:500] + "... (truncated)"
                        log_lines.append(f"```json\n{res_str}\n```")

            return "\n\n".join(log_lines)

        async def run_chat():
            response_text = ""
            current_logs = []
            content = types.Content(role="user", parts=[types.Part(text=prompt)])

            if show_subagent:
                subagent_status = subagent_placeholder.expander(
                    "Processing...", expanded=True
                )
                live_log_container = subagent_status.empty()

            async for event in st.session_state.runner.run_async(
                user_id="web_user",
                session_id=st.session_state.session_id,
                new_message=content,
            ):
                # Format event
                formatted_log = format_event(event)
                if formatted_log:
                    current_logs.append(formatted_log)

                    if show_subagent:
                        live_log_container.markdown("\n---\n".join(current_logs))

            # Get final answer from session state
            session = await st.session_state.runner.session_service.get_session(
                app_name="text2sql_chat",
                user_id="web_user",
                session_id=st.session_state.session_id,
            )

            if session and session.state.get("final_answer"):
                response_text = session.state.get("final_answer")
            else:
                response_text = "Done. (Check 'final_answer' in state if empty)"

            return response_text, current_logs

        full_response, subagent_logs = asyncio.run(run_chat())

        message_placeholder.markdown(full_response)
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": full_response,
                "subagent_output": "\n".join(subagent_logs),
            }
        )
