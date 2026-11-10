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

# GitHub mark (Octicons mark-github, MIT), inlined so the page loads no external image.
GITHUB_SVG = base64.b64encode(
    b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="#8c959f"><path d="M8 0c4.42 0 8 '
    b'3.58 8 8a8.013 8.013 0 0 1-5.45 7.59c-.4.08-.55-.17-.55-.38 0-.27.01-1.13.01-2.2 0-.75-.25-1.23-.54-1.48 '
    b'1.78-.2 3.65-.88 3.65-3.95 0-.88-.31-1.59-.82-2.15.08-.2.36-1.02-.08-2.12 0 0-.67-.22-2.2.82-.64-.18-1.32-.27'
    b'-2-.27-.68 0-1.36.09-2 .27-1.53-1.03-2.2-.82-2.2-.82-.44 1.1-.16 1.92-.08 2.12-.51.56-.82 1.28-.82 2.15 0 '
    b'3.06 1.86 3.75 3.64 3.95-.23.2-.44.55-.51 1.07-.46.21-1.61.55-2.33-.66-.15-.24-.6-.83-1.23-.82-.67.01-.27'
    b'.38.01.53.34.19.73.9.82 1.13.16.45.68 1.31 2.69.94 0 .67.01 1.3.01 1.49 0 .21-.15.45-.55.38A7.995 7.995 0 0 1 '
    b'0 8c0-4.42 3.58-8 8-8Z"/></svg>').decode()


def github_link(label: str, url: str) -> str:
    icon = (f'<img src="data:image/svg+xml;base64,{GITHUB_SVG}" alt="GitHub" '
            'style="width:1em;height:1em;vertical-align:-0.125em;margin-right:0.25em">')
    return f'<a href="{url}" target="_blank">{icon}{label}</a>'


st.title("🤖 Text2SQL：用中文問銷售資料")
st.markdown(f"""
輸入中文問題，系統自動寫 SQL、在唯讀資料庫上執行、檢查結果是否對題，再回覆說明與圖表。
模型在本機 GPU 上執行，SQL 執行、安全限制與圖表繪製都由程式負責，不交給模型。

**建議從左側「對話查詢」開始**：側邊欄有幾題範例問題，回答下方可展開執行流程，
逐步查看模型寫的 SQL、查詢結果與審查結果。程式碼：{github_link('GitHub', REPO_URL)}
""", unsafe_allow_html=True)
# "openai/" is only LiteLLM's provider prefix for an OpenAI-compatible endpoint, not the model.
MODEL_ALIAS = ANALYSIS_MODEL.split("/", 1)[1] if ANALYSIS_MODEL.startswith("openai/") else ANALYSIS_MODEL

st.subheader("使用的模型")
st.markdown(f"""
目前分析模型：`{MODEL_ALIAS}`

- **Swift 1.5**：UkisAI 以阿里巴巴開源的 **Qwen3.8-27B** 做後訓練的版本，思考 token 較少、回應較快。
- 量化格式為 GSQ-RCO IQ3_XXS GGUF（約 10.4 GB，含 MTP 推測解碼），
  由 **本機 llama.cpp** 伺服器執行，模型推論不呼叫任何外部 API。
- 同一個模型分別擔任四個角色：意圖判斷、SQL 規劃、語意審查、回答撰寫；SQL 執行與檢查由程式負責。
- **意圖判斷**（2026-11-09 加入）：寫 SQL 前先判斷問題是不是銷售資料查詢；要求改資料、資料庫沒有的資料
  （預測、毛利）、閒聊、提示注入這四類由程式直接用固定文字拒答，不寫也不執行 SQL。
  曾比較 Jev 類的開源判斷模型 Laya、提示注入分類器 PIGuard 與 ProtectAI（皆在 CPU 上執行、不佔顯存），
  但它們在中文四類上分不清：攔得多的誤擋也多（Laya 攔 86%、誤擋 59%），誤擋少的攔不到（ProtectAI 22%、
  PIGuard 0%），用本專案種子微調 Laya 後未見過的題目仍只攔 67–75%、誤擋 5–12%。改用已載入的同一個模型、不思考的短判斷，
  不新增任何模型或顯存，攔截題由 7.9% 提升到 99.2%。比較數字見「🛠️ 優化成果」頁。
- 模型來源：🤗 [Swift-1.5-Qwen3.8-27B-GSQ-RCO-GGUF](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27B-GSQ-RCO-GGUF)
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
st.caption(f"完整名詞定義見 {github_link('docs/MODULES.md', f'{DOCS_URL}/MODULES.md')}，"
           f"問答流程細節見 {github_link('docs/ARCHITECTURE.md', f'{DOCS_URL}/ARCHITECTURE.md')}。",
           unsafe_allow_html=True)
