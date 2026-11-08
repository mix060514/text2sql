import streamlit as st
import sys
import os
import pathlib
import asyncio
import uuid
import time
from typing import List, Dict
import plotly.io as pio

# Setup path to import local src
# Assuming web_ui/pages/1_Chat.py -> web_ui/pages -> web_ui -> project_root -> src
project_root = pathlib.Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(project_root / "src"))
sys.path.append(str(project_root / "eval"))
sys.path.insert(0, str(project_root / "web_ui"))

try:
    from google.adk.runners import InMemoryRunner
    from google.genai import types
    from text2sql.agents.planned_agent.agent import root_agent
    from text2sql.runtime import stream_turn
    from text2sql.config import BACKEND
    from flow_view import render_flow
except ImportError as e:
    st.error("服務暫時無法啟動，請稍後再試。")
    print(f"chat page import failed: {e}", file=sys.stderr)
    st.stop()

st.set_page_config(page_title="對話查詢", page_icon="💬", layout="wide")

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

FLOW_TITLE = "查看執行流程（點步驟看輸入／輸出）"


# Only questions the default model (Swift 1.5) answered correctly in the 2026-11-07
# 132-question eval, with SQL results matching data/eval_set_v3.jsonl ground truth.
# Region-level monthly trends are left out: most of them hit the 120 s timeout.
SUGGESTED_QUESTIONS = [
    # Single value
    "EMEA 的總銷售額是多少？",
    "2024 年的總銷售額是多少？",
    "哪一個產品的平均單價 (ASP) 最高？",
    "Germany 訂單數量最多的客戶是誰？",
    "產品類別 Electronics 的最大購買客戶是誰？",
    "2024 年哪一個季度的總銷售額表現最好？",
    "Electronics 類別在 2024 年相對於 2023 年的銷售額年增率 (YoY) 是多少？(請以百分比表示)",
    "EMEA 區域 2024 年對比 2023 年的銷售額年增率是多少？",
    # Table
    "各個區域 (Region) 最偏好（銷售額最高）的產品類別分別為何？",
    # Bar chart
    "APAC 銷售量最高的前三個產品是什麼？",
    "請列出 Electronics 類別在各年份的銷售額總計。",
    "請比較各區域的總銷售額。",
    "銷售額前五名的產品分別是哪些？請用圖表呈現。",
    "2024 年各季度的銷售額分別是多少？",
    # Line chart (monthly trend, slower: about 1-2 minutes)
    "請呈現 Electronics 類別在 2023 至 2024 年的月銷售額趨勢。",
    # Pie chart
    "各區域的銷售額占比分別是多少？請用圓餅圖呈現。",
    "2024 年各產品類別的銷售額占比分別是多少？",
]

selected_suggestion = st.sidebar.selectbox(
    "範例問題", [""] + SUGGESTED_QUESTIONS
)
send_suggestion = st.sidebar.button("送出範例問題")

# Display Messages
st.info(
    "👋 用中文詢問 2023–2024 年的全球銷售資料（地區、國家、產品、客戶、銷售額）。"
    "可在下方輸入問題，或從左側選一題範例。月趨勢等較複雜的問題約需 1–2 分鐘。"
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
                st.error("圖表無法顯示。")
        if msg.get("workflow_trace"):
            with st.expander(FLOW_TITLE):
                render_flow(msg["workflow_trace"], msg.get("backend", BACKEND))

# Chat Input
chat_prompt = st.chat_input("輸入問題，例如：EMEA 的總銷售額是多少？")

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
        chart_placeholder = st.empty()
        flow_placeholder = st.empty()

        def show_flow(trace, expanded):
            # The workflow streams a trace snapshot when each step starts and ends;
            # redraw the diagram in place so the user sees which step is running.
            with flow_placeholder.container():
                with st.expander(FLOW_TITLE, expanded=expanded):
                    render_flow(trace, BACKEND)

        async def run_chat():
            content = types.Content(role="user", parts=[types.Part(text=prompt)])
            message_placeholder.markdown("⏳ 分析中…")
            async for event in stream_turn(
                st.session_state.runner,
                user_id="web_user",
                session_id=st.session_state.session_id,
                new_message=content,
            ):
                delta = event.actions.state_delta if event.actions else None
                if delta and delta.get("workflow_trace"):
                    show_flow(delta["workflow_trace"], expanded=True)

            session = await st.session_state.runner.session_service.get_session(
                app_name="text2sql_chat",
                user_id="web_user",
                session_id=st.session_state.session_id,
            )
            if not session:
                return "這次分析未能完成，請稍後重試。", None, []
            response_text = (
                session.state.get("final_answer")
                or session.state.get("answer")
                or "這次分析未能完成，請稍後重試。"
            )
            return (response_text, session.state.get("chart_json"),
                    session.state.get("workflow_trace") or [])

        full_response, chart_json, trace = asyncio.run(run_chat())

        message_placeholder.markdown(full_response)

        if chart_json:
            try:
                fig = pio.from_json(chart_json)
                chart_placeholder.plotly_chart(fig, use_container_width=True, key="latest_chart")
            except Exception as e:
                chart_placeholder.error("圖表無法顯示。")

        if trace:
            show_flow(trace, expanded=True)
        else:
            flow_placeholder.empty()

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": full_response,
                "chart_json": chart_json,
                "workflow_trace": trace,
                "backend": BACKEND,
            }
        )
