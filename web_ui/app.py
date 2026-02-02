import streamlit as st

st.set_page_config(
    page_title="Text2SQL Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🤖 Text2SQL Agent demo APP", anchor='top')

st.markdown(
    """
歡迎來到 Text2SQL Agent Demo的示範APP。

本應用展示了一個多代理系統，該系統能夠將自然語言查詢轉換為SQL查詢，並從銷售資料庫中查詢資料，返回合理描述給使用者。
額外包含一個多模態語言模型，可以處理文字和圖片輸入。

### 請用左邊側邊條(sidebar)選擇展示的頁面。
- **Chat**: 用正常語言詢問資料問題，讓 Agent 查詢資料庫獲得最新資料。可以查看sub Agent的調用過程，包含範例問題集。
- **Eval**: 查看開發此agent應用時評估的資料集和評分的詳細的評判。
- **System Monitor**: 監控系統狀態和日誌，目前本應用部署在地端（NB RTX3080 16GB vram），部署模型為Qwen3-4b-instruct-2507以及qwen3-4b-vl。
- **Data Dashboard**: 展示實際的銷售資料，用來做chat的對比。
- **Image Read**: 展示多模態語言模型的圖片理解能力，可以上傳圖片並詢問相關問題。

從左側邊欄選擇一個頁面開始。

---
"""
)


# Mermaid Diagram helper
def mermaid(code: str, height: int=600):
    import streamlit.components.v1 as components

    components.html(
        f"""
        <pre class="mermaid">
            {code}
        </pre>
        <script type="module">
            import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
            mermaid.initialize({{ startOnLoad: true }});
        </script>
        """,
        height=height,
    )



# st.markdown("### Agent 互動序列", anchor="agent-互動序列")
st.subheader("Agent 交互時序圖", anchor="agent-時序圖")

mermaid(
    """
%%{init: { 'theme': 'base', 'themeVariables': { 
    'loopBkg': '#E1F5FE', 
    'loopBorder': '#0277BD', 
    'altBkg': '#FFF9C4', 
    'altBorder': '#FBC02D' 
} } }%%
sequenceDiagram
    autonumber
    participant User as 用戶
    participant Root as root Agent
    participant QA as 查詢&回答 Agent
    participant GetData as 獲取數據 Agent (Loop)
    participant Region as 區域檢查 Agent
    participant SqlGen as SQL生成 Agent
    participant Check as 檢查SQL Agent
    participant Exec as 執行SQL Agent
    participant Critic as 批評家 Agent
    participant Ans as 回答 Agent

    User->>Root: 提出數據相關問題
    Root->>QA: 委派任務
    
    Note over QA, GetData: 開始數據獲取迴圈 (最多重試 3 次)
    QA->>GetData: 啟動流程

    loop 數據獲取與修正迴圈
        %% 第一階段：生成 SQL
        GetData->>Region: 檢查問題中的國家/地區
        Region-->>GetData: 回傳地區上下文 (region_country)
        GetData->>SqlGen: 根據 Schema 生成 SQL
        SqlGen-->>GetData: 回傳 SQL 語句 (sql_query)

        %% 第二階段：驗證與執行
        GetData->>Check: 檢查 SQL 語法
        Check-->>GetData: 語法確認無誤
        GetData->>Exec: 執行 SQL 查詢
        Exec-->>GetData: 回傳查詢結果 (query_result)
        GetData->>Critic: 審查結果是否回答問題
        
        alt 結果正確
            Critic-->>GetData: 呼叫工具: exit_loop (跳出迴圈)
        else 結果錯誤
            Critic-->>GetData: 回傳錯誤反饋 (觸發重試)
        end
    end

    GetData-->>QA: 回傳最終查詢結果
    
    QA->>Ans: 生成商業回答
    Ans-->>QA: 回傳最終文本 (繁中+英文產品名)
    
    QA-->>Root: 任務完成
    Root-->>User: 回傳最終答案
""",
height=900
)

st.markdown("---")
# st.markdown("### 數據範例")
st.subheader("資料範例", anchor="sample-data")
st.markdown("系統使用的全球銷售數據前 5 行：")

sample_data = {
    "Order ID": ["ORD-202309986", "ORD-202325336", "ORD-202315895", "ORD-202328380", "ORD-202318954"],
    "Order Date": ["2023-01-01", "2023-01-01", "2023-01-01", "2023-01-01", "2023-01-01"],
    "Region": ["North America", "LATAM", "North America", "EMEA", "APAC"],
    "Country": ["United States", "Mexico", "United States", "United Kingdom", "Singapore"],
    "Customer Name": ["Elite Elite Systems", "Apex Elite Systems", "Apex Apex Ltd.", "Next Beta Ltd.", "Next Delta Ltd."],
    "Product Category": ["Electronics", "Electronics", "Electronics", "Software", "Office Supplies"],
    "Product Name": ["Docking Station", "4K Monitor 27\"", "Pro Smartphone 15", "Team Collaboration Tool", "Ergonomic Chair"],
    "ASP": [184.04, 345.53, 1002.65, 208.97, 382.53],
    "Quantity": [6, 11, 3, 5, 2],
    "Total Revenue": [1104.24, 3800.83, 3007.95, 1044.85, 765.06]
}

import pandas as pd
df = pd.DataFrame(sample_data)
st.dataframe(df, use_container_width=True)

st.sidebar.success("請在上方選擇一個頁面。")

st.sidebar.markdown("目錄")
st.sidebar.markdown("* [回頂部](#top)")
st.sidebar.markdown("* [Agent 交互時序圖](#agent-時序圖)")
st.sidebar.markdown("* [資料範例](#sample-data)")
數據範例
