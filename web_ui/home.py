import base64
import streamlit as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHITECTURE_SVG = ROOT / "docs" / "architecture.svg"
sys.path.insert(0, str(ROOT / "src"))
from text2sql.config import ANALYSIS_MODEL

REPO = "https://github.com/mix060514/text2sql"
BRANCH = "codex/text2sql-unified"  # master is the old implementation
REPO_URL = f"{REPO}/tree/{BRANCH}"
DOCS_URL = f"{REPO}/blob/{BRANCH}/docs"

st.title("🤖 Text2SQL：用中文問銷售資料")
st.markdown(f"""
輸入中文問題，系統自動寫 SQL、在唯讀資料庫上執行、檢查結果是否對題，再回覆說明與圖表。
模型在本機 GPU 上執行，SQL 執行、安全限制與圖表繪製都由程式負責，不交給模型。

**建議從左側「對話查詢」開始**：側邊欄有幾題範例問題，回答下方可展開執行流程，
逐步查看模型寫的 SQL、查詢結果與審查結果。程式碼：[GitHub]({REPO_URL})
""")
# "openai/" is only LiteLLM's provider prefix for an OpenAI-compatible endpoint, not the model.
MODEL_ALIAS = ANALYSIS_MODEL.split("/", 1)[1] if ANALYSIS_MODEL.startswith("openai/") else ANALYSIS_MODEL

st.subheader("使用的模型")
st.markdown(f"""
目前分析模型：`{MODEL_ALIAS}`

- **Swift 1.5**：UkisAI 以阿里巴巴開源的 **Qwen3.8-27B** 做後訓練的版本，思考 token 較少、回應較快。
- 量化格式為 GSQ-RCO IQ3_XXS GGUF（約 10.4 GB，含 MTP 推測解碼），
  由 **本機 llama.cpp** 伺服器執行，模型推論不呼叫任何外部 API。
- 同一個模型分別擔任三個角色：SQL 規劃、語意審查、回答撰寫；SQL 執行與檢查由程式負責。
- 模型來源：[Swift-1.5-Qwen3.8-27B-GSQ-RCO-GGUF](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27B-GSQ-RCO-GGUF)
""")

st.divider()
st.subheader("系統架構")
st.markdown("""
| 編號 | 模組 | 做什麼 | 產出 |
|---|---|---|---|
| M1 | 出題程式 | 把地區、產品等填進 17 個題型，用 pandas 從原始資料算出標準答案 | 模板考卷（132 題） |
| M2 | 合成程式 | 由模型改寫種子題的問法，再經規則、標籤、語意三層檢查 | 合成考卷（354 題，含 126 題攔截題） |
| M3 | 評估器 | 用兩份考卷測試 M4 並評分（模板考卷由 LLM 評審，合成考卷直接比對查詢結果） | 成績單（見「評估結果」頁） |
| M4 | 問答系統 | 寫 SQL、唯讀查詢、語意審查、回答與圖表 | 答案（表格＋圖表＋說明） |
| M5 | 網頁服務 | 本網站，使用者從這裡提問 | 網頁畫面 |

名詞：**考卷**＝評估集（題目＋標準答案）；**攔截題**＝應該拒答的題目，例如要求修改資料或與資料無關的問題。
""")
if ARCHITECTURE_SVG.exists():
    svg = base64.b64encode(ARCHITECTURE_SVG.read_bytes()).decode()
    st.markdown(
        f'<img src="data:image/svg+xml;base64,{svg}" alt="text2sql 架構圖" '
        'style="width:100%;max-width:680px">',
        unsafe_allow_html=True,
    )
st.caption(f"完整名詞定義見 [docs/MODULES.md]({DOCS_URL}/MODULES.md)，"
           f"問答流程細節見 [docs/ARCHITECTURE.md]({DOCS_URL}/ARCHITECTURE.md)。")
