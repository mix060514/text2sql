# 架構比較與閱讀指南

基準：`fd66447`。兩版都保留原本的 Streamlit / ADK session 介面，
但推理與控制流程的責任不同。

| | 強模型版 | 27B 版 |
|---|---|---|
| 分支 | codex/text2sql-strong | codex/text2sql-qwen27b |
| 分析模型 | Codex CLI，gpt-6-sol | Pi 使用的 qwen3.8-27b-iq3-xxs |
| 推理單位 | 一個 Codex Agent | SQL planner、semantic reviewer、report writer |
| 工具 / 推理迴圈 | Codex Harness | 固定程式流程，模型不帶 tools |
| SQL 檢查與執行 | 同一個唯讀 MCP 查詢工具內直接執行 | controller 直接 execute_query |
| 應用內模型 retry | 無 | SQLPlan/SQL/語義審查錯誤合計最多 3 次；report/圖表錯誤最多 2 次 |
| 圖表 | 驗證最終規格後直接 render_chart | 同左 |
| 格式控制 | Codex output-schema + Pydantic gate | ADK output_schema + Pydantic gate |
| 最終成功條件 | 本回合查詢成功、規格有效 | 本次查詢成功、strict review 接受且綁定執行結果、規格有效 |
| context | Harness 管本次工具對話；應用只傳上回合意圖 | 每個步驟使用明確的問題、資料、錯誤欄位 |

## 強模型路線

```text
UI / API
└─ CodexSalesAgent（ADK 介面轉接）
   └─ 一次 Codex Harness invocation
      ├─ 規劃、SQL 選擇、工具錯誤修正
      └─ sales.query_sales_data(sql)
         └─ deterministic SELECT 驗證 + SQLite 執行
   → 應用檢查本回合執行 receipt + AnswerSubmission
   → deterministic render_chart
   → deterministic result_table（完整資料）+ answer / chart
```

`agent.py` 只接橋接函式與完成結果。沒有 root delegation prompt，
沒有為每一步設一個 Agent，也沒有自己寫一個 loop until success。
`codex_backend.py` 啟動一次非互動 CLI，使用 JSON Schema 接收最終規格。
`mcp_server.py` 提供實際查詢，並把最後一次查詢的執行證據寫入本回合的暫存 receipt。

receipt 由應用的 MCP server 產生，不能把模型自己說「查詢成功」視作證據。
每次 invocation 使用獨立暫存目錄；不讀上回合的 receipt。
CLI 不繼承個人 connectors，shell 關閉、sandbox 是 read-only；
MCP 的 annotation 是提示，真正的 DB 權限仍由 SQLite 執行。
本次 CLI 的期限由應用限制；取消時停止整個程序群組。
這是本機示範接法，沒有把 ChatGPT 登入憑證放進 repo 或提供給模型。

官方依據：
[非互動與 output-schema](https://learn.chatgpt.com/docs/non-interactive-mode)、
[MCP 設定](https://learn.chatgpt.com/docs/extend/mcp?surface=cli)。

## 小模型路線

```text
UI / API
└─ Text2SQLWorkflow（deterministic controller）
   ├─ SQL planner → SQLPlan
   │  └─ schema / SQL error：回饋 planner
   ├─ execute_query(sql)：必定由程式直接執行
   ├─ semantic reviewer → SemanticReview（accepted: bool、reason）
   │  └─ 拒絕 / review schema error：回饋 planner，查詢合計最多 3 次嘗試
   ├─ report writer → AnswerSubmission
   │  └─ schema / 圖表 error：修正 writer，最多 2 次總嘗試
   └─ complete_answer：hard gate + render + 完整資料表 + commit
```

27B 負責 SQL、意圖審查與解釋的不確定性，不決定固定步驟是否發生。
三個模型步驟都沒有 tools，不存在 prompt 要求另一個 Agent 務必呼叫 executor。
retry 只處理具體錯誤，重跑出錯的步驟，不能用無限 critic loop 取代成功條件。
reviewer 明確取得原問題、追問上下文、DOMAIN 業務定義及實際 SQL / columns / rows。
controller 只在本次 query_success 且 strict review 明確接受後進 report，
發布時比對 reviewed_query 與當前執行結果；拒絕、解析失敗及耗盡均不發布分析。
每次嘗試、回合開始與失敗清除 review、結果及圖表，不能沿用前次接受。
查詢修正與報告修正分開，後者不重查或重審；維持 8 calls / 300 秒總限制。
正常路徑 3 calls，最壞 3 × (planner + reviewer) + 2 report = 8。
語義 reviewer 仍是同一 Qwen 的判斷，可能與 planner 有共同盲點，並非一般 SQL 正確性的證明。

## 兩版共同保留的程式責任

- 單一 SELECT、read-only connection、只授權 sales_data 與允許的分析函式。
- 由 SQLite 準備、執行來驗證真正語法；不以 sqlparse.parse 的非空結果判定正確。
- 5 秒 SQL 時間、200 列、20,000 bytes；超限不接受不完整成功資料。
- 空結果與執行失敗分開；失敗後沒有成功回答。
- 圖表欄位必須存在、y 必須有限數值、pie 不允許負值或零總額。
- 多列數值結果需要圖表；已知該產生圖表時，由程式直接呼叫繪圖。
- 每回合清除資料與圖表，只攜帶前一個成功問題與 SQL 供追問參考。
- 直接回傳已提交文字，完整消費 event stream，不再用 LLM echo。
- 完整列出實際資料由 result_table 保證；模型只寫解釋，不需重抄所有數字。
- eval 配對 DB 的 SHA-256，不能把資料版本或 fixture 問題算成模型錯誤。

## 三類問題的處理位置

A：指標理解、region mapping、SQL 分組與圖表類型選擇仍是模型的推理問題。
較強模型是否減少這些錯誤，需要實測，不能從架構圖推出結論。

B：輸出欄位型別與 JSON/schema，由框架/CLI 格式約束與硬驗證處理。
小模型版有有限的錯誤修正；強模型版不增加 application repair loop。

C：不能寫 DB、失敗不能繼續、資料量與時間上限、狀態隔離，是程式規則。
強模型也不能取代這些條件。強模型成功 gate 驗證執行與結構；27B 另要求本次語義審查接受。
兩者都不保證 answer 所有敘述都正確。

沒有因為「Agent 不可靠」就加入 scheduler、權限審批工作流或永久 checkpoint。
目前仍是 InMemoryRunner；若未來需求包含恢復或多使用者服務，才另加相應的儲存與隔離。
