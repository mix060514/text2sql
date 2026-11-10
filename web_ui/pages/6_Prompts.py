"""Prompt 模板: the instruction each model step receives, read from the running agents.

Nothing here is a copy: the text, settings and output schemas come from the LlmAgent
objects in qwen_workflow, so the page always matches what the service sends.
"""

import json
import pathlib
import re
import sys

import streamlit as st

project_root = pathlib.Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(project_root / "src"))

st.set_page_config(page_title="Prompt 模板", page_icon="📝", layout="wide")
st.title("📝 Prompt 模板")

try:
    from text2sql.agents.planned_agent import qwen_workflow as wf
    from text2sql.config import (REPORT_ATTEMPTS, REPORT_THINKING, REVIEW_THINKING_BUDGET, SQL_ATTEMPTS,
                                 SQL_FEW_SHOT, SQL_THINKING_BUDGET)
except ImportError as e:
    st.error("無法載入問答系統設定。")
    print(f"prompts page import failed: {e}", file=sys.stderr)
    st.stop()

PLACEHOLDER = re.compile(r"\{(\w+)\}")

# Where each placeholder's value comes from at run time (session state written by code).
SOURCES = {
    "user_question": "使用者這次的問題",
    "previous_query_context": "前一個成功回合的問題與 SQL（JSON），只用來理解追問",
    "sql_feedback": "上一次 SQL 的錯誤訊息（執行錯誤或審查退回的理由），第一次為空",
    "sql_query": "這次實際執行的 SQL",
    "column_names": "這次查詢結果的欄位名稱",
    "raw_query_result": "這次查詢結果的資料列（有列數與位元組上限）",
    "report_feedback": "上一次回答未通過程式檢查的原因，第一次為空",
}

STAGES = [
    {
        "agent": wf.agent_intent, "title": "① 意圖判斷",
        "role": "寫 SQL 前先判斷訊息是不是銷售資料查詢。不是的話，程式直接回固定拒答文字，"
                "模型寫的 reason 只進執行紀錄，不會出現在回答裡。",
        "parts": "DOMAIN + INTENT_TASK",
        "thinking": "關閉",
        "max_tries": "1 次", "attempts": "1 次；輸出無法解析時放行到 SQL 步驟，SQL 仍受唯讀限制",
    },
    {
        "agent": wf.agent_sql, "title": "② SQL 規劃",
        "role": "只產生一個 SQLite SELECT，由程式執行；執行失敗或審查退回時，錯誤訊息會填進 {sql_feedback} 重寫。",
        "parts": "DOMAIN + " + ("SQL 範例（few-shot）+ " if SQL_FEW_SHOT else "") + "SQL_TASK",
        "thinking": f"預算 {SQL_THINKING_BUDGET:,}",
        "max_tries": f"{SQL_ATTEMPTS} 次", "attempts": f"共 {SQL_ATTEMPTS} 次",
    },
    {
        "agent": wf.agent_review, "title": "③ 語意審查",
        "role": "SQL 執行成功後，核對 SQL 與結果是否真的回答原問題；不通過時 reason 會成為下一次 SQL 的錯誤回饋。",
        "parts": "DOMAIN + 審查規則",
        "thinking": f"預算 {REVIEW_THINKING_BUDGET:,}",
        "max_tries": f"{SQL_ATTEMPTS} 次", "attempts": f"每次 SQL 執行成功後一次（與 SQL 共用 {SQL_ATTEMPTS} 次額度）",
    },
    {
        "agent": wf.agent_report, "title": "④ 回答撰寫",
        "role": "依實際查詢結果寫文字回答與圖表規格；圖表與資料表由程式繪製，欄位與數值由程式檢查。",
        "parts": "REPORT_RULES + 查詢結果",
        "thinking": "開啟" if REPORT_THINKING else "關閉",
        "max_tries": f"{REPORT_ATTEMPTS} 次", "attempts": f"共 {REPORT_ATTEMPTS} 次",
    },
]

st.markdown("""
問答系統有四個步驟由模型負責，每一步都是同一個本機模型，收到各自的指令（prompt）。
以下內容直接從執行中的程式讀出，與服務實際送出的文字一致。
`{名稱}` 是模板欄位，每次提問時由程式從這一回合的狀態填入；模型看不到對話歷史，只看到這份填好的指令。
""")

tabs = st.tabs([s["title"] for s in STAGES])
for tab, stage in zip(tabs, STAGES):
    agent = stage["agent"]
    config = agent.generate_content_config
    text = agent.instruction
    with tab:
        st.markdown(stage["role"])
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("temperature", f"{config.temperature:g}")
        c2.metric("輸出上限", f"{config.max_output_tokens:,} tokens")
        c3.metric("思考（tokens）", stage["thinking"])
        c4.metric("最多嘗試", stage["max_tries"])
        st.caption(f"組成：`{stage['parts']}`　·　嘗試：{stage['attempts']}　·　"
                   f"指令長度 {len(text):,} 字　·　agent 名稱 `{agent.name}`")

        fields = list(dict.fromkeys(PLACEHOLDER.findall(text)))
        if fields:
            st.markdown("**模板欄位**")
            st.dataframe(hide_index=True, width="stretch", data=[{"欄位": f"{{{f}}}", "執行時填入": SOURCES.get(f, "—")} for f in fields])

        st.markdown("**指令全文**")
        st.code(text.strip(), language="markdown", wrap_lines=True)

        if agent.output_schema is not None:
            with st.expander(f"輸出格式：{agent.output_schema.__name__}（模型必須回這個 JSON schema）"):
                st.code(json.dumps(agent.output_schema.model_json_schema(), ensure_ascii=False, indent=2),
                        language="json")

        if agent is wf.agent_intent:
            st.markdown("**非查詢類別的固定拒答文字**（由程式輸出，不經過模型）")
            st.markdown("\n".join(f"- `{k}`：{v}" for k, v in wf.REFUSALS.items()))
