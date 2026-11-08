import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from text2sql.config import ANALYSIS_MODEL

st.set_page_config(page_title="Text2SQL Agent", page_icon="🤖", layout="wide",
                   initial_sidebar_state="expanded")
st.title("🤖 Text2SQL Agent")
st.markdown("""
以自然語言查詢銷售資料，查看實際查詢結果與圖表。
請從左側選擇 Chat、Eval、System Monitor 或 Data Dashboard。
""")
st.caption(f"分析模型：{ANALYSIS_MODEL}")
st.markdown("分析流程：SQL 規劃 → 唯讀查詢 → 語意審查 → 回答與圖表。")
st.sidebar.success("從左側選擇頁面開始。")
