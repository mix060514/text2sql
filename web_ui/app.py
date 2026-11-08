"""Streamlit entrypoint: declares the sidebar pages with Traditional Chinese labels."""
import streamlit as st

st.set_page_config(page_title="Text2SQL Agent", page_icon="🤖", layout="wide",
                   initial_sidebar_state="expanded")

st.navigation([
    st.Page("home.py", title="首頁", icon="🏠", default=True),
    st.Page("pages/1_Chat.py", title="對話查詢", icon="💬"),
    st.Page("pages/2_Eval.py", title="評估結果", icon="📊"),
    st.Page("pages/3_System_Monitor.py", title="系統監控", icon="🖥️"),
    st.Page("pages/4_Data_Dashboard.py", title="資料儀表板", icon="📈"),
]).run()
