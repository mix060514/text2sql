import streamlit as st
import sys
import os
import pathlib
import json
import asyncio
import uuid
import time
from typing import List, Dict
import plotly.io as pio

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


SUGGESTED_QUESTIONS = [
    "EMEA 的總銷售額是多少？",
    "請列出 Electronics 類別在各年份的銷售額總計。",
    "請列出 North America 區域在 2023 至 2024 年每個月的銷售趨勢數據。",
    "各區域的銷售額佔比是多少？",
]

selected_suggestion = st.sidebar.selectbox(
    "Suggested Questions", [""] + SUGGESTED_QUESTIONS
)
send_suggestion = st.sidebar.button("Send Suggestion")

# Display Messages
st.info(
    "👋 歡迎使用 *資料庫查詢助手*！請在下方輸入您的問題，我會幫您查詢資料庫回答您問題。"
)

for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # Display chart if available
        if "chart_json" in msg and msg["chart_json"]:
            try:
                fig = pio.from_json(msg["chart_json"])
                st.plotly_chart(fig, use_container_width=True, key=f"history_chart_{i}")
            except Exception as e:
                st.error(f"Failed to render chart: {e}")
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

            chart_json = None
            if session:
                response_text = (
                    session.state.get("final_answer")
                    or session.state.get("answer")
                    or "No answer generated."
                )
                chart_json = session.state.get("chart_json")
            else:
                response_text = "No session found."

            return response_text, current_logs, chart_json

        full_response, subagent_logs, chart_json = asyncio.run(run_chat())

        message_placeholder.markdown(full_response)

        # Display chart if available
        # Display chart if available
        if chart_json:
            try:
                fig = pio.from_json(chart_json)
                st.plotly_chart(fig, use_container_width=True, key="latest_chart")
            except Exception as e:
                st.error(f"Failed to render chart: {e}")

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": full_response,
                "subagent_output": "\n".join(subagent_logs),
                "chart_json": chart_json,
            }
        )
