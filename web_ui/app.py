import base64
import streamlit as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHITECTURE_SVG = ROOT / "docs" / "architecture.svg"
sys.path.insert(0, str(ROOT / "src"))
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

st.divider()
st.subheader("系統架構")
st.markdown("""
| 編號 | 模組 | 做什麼 | 產出 |
|---|---|---|---|
| M1 | 出題程式 | 題型填空，用銷售資料算標準答案 | 模板考卷（132 題） |
| M2 | 合成程式 | 種子交給 Qwen 改寫，再三層檢查 | 合成考卷（354 題，含攔截題） |
| M3 | 評估器 | 拿考卷考 M4 並評分 | 成績單 |
| M4 | 問答系統 | 寫 SQL、唯讀查詢、檢查、回答 | 答案（表格 + 圖 + 說明） |
| M5 | 網頁服務 | 本網站，使用者從這裡提問 | 網頁畫面 |
""")
if ARCHITECTURE_SVG.exists():
    svg = base64.b64encode(ARCHITECTURE_SVG.read_bytes()).decode()
    st.markdown(
        f'<img src="data:image/svg+xml;base64,{svg}" alt="text2sql 架構圖" '
        'style="width:100%;max-width:680px">',
        unsafe_allow_html=True,
    )
st.caption("名詞定義與已知缺口見 docs/MODULES.md。")
