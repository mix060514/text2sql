# 架構比較與閱讀指南

> 2026-11-08：強模型（Codex CLI）路線已從 `codex/text2sql-unified` 移除，
> 目前只有下方「小模型路線」。比較表保留作歷史紀錄；移除前的程式在 commit `96f9615`。

基準：`fd66447`。兩版都保留原本的 Streamlit / ADK session 介面，
但推理與控制流程的責任不同。

| | 強模型版 | 27B 版 |
|---|---|---|
| 分支 | codex/text2sql-strong | codex/text2sql-qwen27b |
| 分析模型 | Codex CLI，gpt-6-sol | Swift 1.5（Qwen3.8-27B 後訓練、思考較短），`swift15-27b-iq3-xxs`；2026-11-07 前為原版 `qwen3.8-27b-iq3-xxs` |
| 推理單位 | 一個 Codex Agent（agent_codex） | agent_sql、agent_review、agent_report |
| 工具 / 推理迴圈 | Codex Harness | 固定程式流程，模型不帶 tools |
| SQL 檢查與執行 | 同一個唯讀 MCP 查詢工具內直接執行 | controller 直接 step_run_sql |
| 應用內模型 retry | 無 | SQLPlan/SQL/語義審查錯誤合計最多 3 次；report/圖表錯誤最多 2 次 |
| 圖表 | 驗證最終規格後直接 step_draw_chart | 同左 |
| 格式控制 | Codex output-schema + Pydantic gate | ADK output_schema + Pydantic gate |
| 最終成功條件 | 本回合查詢成功、規格有效 | 本次查詢成功、strict review 接受且綁定執行結果、規格有效 |
| context | Harness 管本次工具對話；應用只傳上回合意圖 | 每個步驟使用明確的問題、資料、錯誤欄位 |

## 小模型路線

```text
UI / API
└─ Text2SQLWorkflow（deterministic controller）
   ├─ agent_intent → IntentDecision（不思考；2026-11-09 加入）
   │  └─ write_request / not_in_data / chitchat / prompt_injection：程式用固定文字拒答，
   │     不寫、不執行 SQL，清除結果；輸出不合法時放行（仍受唯讀查詢限制）
   ├─ agent_sql → SQLPlan
   │  └─ schema / SQL error：回饋 planner
   ├─ step_run_sql(sql)：必定由程式直接執行
   ├─ agent_review → SemanticReview（accepted: bool、reason）
   │  └─ 拒絕 / review schema error：回饋 planner，查詢合計最多 3 次嘗試
   ├─ agent_report → AnswerSubmission
   │  └─ schema / 圖表 error：修正 writer，最多 2 次總嘗試
   └─ step_publish：hard gate + render + 完整資料表 + commit
```

27B 負責 SQL、意圖審查與解釋的不確定性，不決定固定步驟是否發生。
三個模型步驟都沒有 tools，不存在 prompt 要求另一個 Agent 務必呼叫 executor。
retry 只處理具體錯誤，重跑出錯的步驟，不能用無限 critic loop 取代成功條件。
reviewer 明確取得原問題、追問上下文、DOMAIN 業務定義及實際 SQL / columns / rows。
controller 只在本次 query_success 且 strict review 明確接受後進 report，
發布時比對 reviewed_query 與當前執行結果；拒絕、解析失敗及耗盡均不發布分析。
每次嘗試、回合開始與失敗清除 review、結果及圖表，不能沿用前次接受。
查詢修正與報告修正分開，後者不重查或重審；每題 9 calls / 600 秒總限制。
正常路徑 4 calls（意圖 1 + SQL + 審查 + 報告），最壞 1 + 3 × (planner + reviewer) + 2 report = 9；拒答路徑 1 call。
意圖判斷的理由只寫入 trace，公開回答一律是程式的固定拒答文字，注入內容不會被回顯。
意圖步驟的工具選擇（CPU 分類器與本機模型的比較）見 docs/optimization/intent_tools.json。
語義 reviewer 仍是同一 Qwen 的判斷，可能與 planner 有共同盲點，並非一般 SQL 正確性的證明。

命名規則：`agent_*` = LLM、`tool_*` = 模型自己呼叫的工具、`step_*` = controller 一定執行的程式。

## 執行紀錄與流程圖

controller 每一步記錄 `{name, kind, attempt, status, input, output, ms}` 到 state `workflow_trace`
（`agents/planned_agent/trace.py`），每回合由 start_turn 清空，失敗路徑也保留。
資料列只存前 20 列預覽；紀錄只供顯示，不回饋給 prompt 或發布判斷。
`step_draw_chart` 在 `step_publish` 內執行，紀錄的是實際 Plotly trace，沒有自己的耗時。
Chat 頁（`web_ui/flow_view.py`）把它畫成可點擊的 SVG 流程圖，點步驟看每次嘗試的輸入、輸出。

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
