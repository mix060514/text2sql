# 交接紀錄（2026-11-07）

新 session 先讀這份，再讀 README.md 與 CLAUDE.md。

## 現況

| 項目 | 狀態 |
|---|---|
| 目前分支 | `codex/text2sql-unified`（commit `19821ae` 起），已 push 到 `origin` |
| GitHub master | 仍停在 2026-02-06 `4761ae8`，尚未合併 unified，也沒開 PR |
| 路線切換 | `TEXT2SQL_BACKEND=qwen27b`（預設）或 `codex` |
| Codex 路線 | 使用者的 Codex 未續費，只有離線測試，沒有實際呼叫 |
| 舊分支 | `codex/text2sql-qwen27b`、`codex/text2sql-strong` 保留不動，只在本機 |
| 強模型 worktree | `../text2sql-strong`（分支 `codex/text2sql-strong`），合併後已不需要；尚未移除 |

## 合併做了什麼

- `planned_agent/agent.py` 只依 `config.BACKEND` 選路線。
- 27B：`qwen_workflow.py`。planner → 程式唯讀執行 SQL → semantic reviewer → report writer → 發布 gate → 程式畫圖與完整資料表。
- Codex：`codex_workflow.py` + `codex_backend.py` + `mcp_server.py`。一次 Codex CLI 呼叫，模型透過唯讀 MCP `tool_query_sales` 查詢；應用核對 MCP 寫的 receipt。
- 發布 gate `tools.step_publish(..., require_review=True)`：27B 要求 reviewer 接受且資料與被接受的執行一致；Codex 傳 `require_review=False`，改以 receipt 為證據。
- 兩條路線共用 `tools.py`（唯讀、上限、畫圖、資料表）、`contracts.py`、`prompts.py`、`context.py`。

## 驗證證據

- 離線 `uv run python scripts/verify_all.py`：54/54（27B 43、Codex 9、路線切換 2）。
- 27B 實測 `南美 的總銷售額是多少？` → 4,872,116.29，約 35 秒，與 SQLite 直接計算相同。
- push 前掃描 `origin/master..unified` 19 個 commit：無金鑰／密碼；`.env` 只含 `localhost` 與假金鑰 `aaa`，且本次是刪除它們。

## 未解決

1. **27B 月份趨勢題失敗**（8 個 smoke 案例中 7/8）。planner 額度 2,048 時用光輸出；調到 4,096 後 reviewer 請求 120 秒超時。細節見 `docs/VALIDATION.md` 末段。
2. **完整 132 題 eval 從未在目前兩條路線上跑過**（`eval/eval_e2e.py`、`eval/eval.py`）。
3. Session 仍用 InMemoryRunner，沒有永久儲存。
4. 是否開 PR 把 unified 合併進 master：使用者尚未決定。

## 既有測試資料

| 檔案 | 筆數 | 說明 |
|---|---|---|
| `data/eval_set_v3.jsonl` | 132 | 正式評估集。eval_type：dataframe 59、number 33、string 23、list 17；difficulty：medium 59、hard 43、simple 30 |
| `data/eval_set_v2.jsonl` | 82 | 舊 DB 用，不再使用 |
| `data/plot_eval_set.jsonl` | 19 | 圖表決策題（mock columns） |
| `data/eval_manifest.json` | — | DB 與 dataset 的 SHA-256 |

產生方式：`scripts/generate_ground_truth.py` 以地區／國家別名表加題目模板展開問法，ground truth 由 pandas 從原始 CSV 計算，不經模型。

## 下一個任務：照演講方法產生新測試資料

使用者想用一場演講（台灣 IBM 講者，「用合成資料測試 AI Guardrails」）的方法產生新資料，舊 132 題當種子與對照組。演講要點：

1. 先定義測試標的與政策：哪些該擋、哪些該放行（演講例子：證券助手，擋投資建議／保證收益，放行公開資訊查詢；目標 300 筆，擋 150、放行 150）。
2. 準備種子資料；改寫時保留關鍵欄位（例如公司、年份），不得改變原意或標籤。
3. 生成流程分階段，不要一個 prompt 做完；每階段可 rollback／regen。
4. 品質檢查分三層，由便宜到昂貴：
   - 規則：JSON 結構、長度、語言，用程式檢查。
   - 標籤／意圖：產出是否仍屬同一類別、同一標籤。
   - 語意：LLM judge 檢查一致性、相關性。
5. 不通過時，把失敗原因放回 prompt 重新生成。
6. 不要照單全收 judge：注意位置偏差、長度偏差；演講的失敗例子是改寫後丟掉了「退休」背景，judge 仍給過。
7. 評估看兩個指標：該擋的有沒有擋、有沒有漏。

套到本專案的草案（尚未確認）：

| 演講步驟 | text2sql |
|---|---|
| 政策 | 該回答的銷售查詢 vs. 不該查資料的輸入（寫入請求、資料庫外問題、閒聊） |
| 種子 | `eval_set_v3.jsonl` 132 題 |
| 生成 | 本機 Qwen 改寫問法，地區、指標、時間不得改變 |
| 規則層 | JSON、長度、繁體中文，程式判斷 |
| 標籤層 | 標準答案由 SQLite 重新計算，不靠模型 |
| 語意層 | judge 檢查改寫是否偏離原意 |
| 輸出 | 新評估集 + 每層刷掉的題數，再分別測兩條路線 |

**使用者已選 B（2026-11-07）**：加入不該查資料的題目，攔截與放行各半。
演講完整重點與套用草案見 `docs/SYNTHETIC_DATA_METHOD.md`。
生成方式：程式管線（本機 Qwen 改寫 + 三層檢查），做成可重跑、可展示的小系統；Claude 寫程式與種子。
已完成第一個檢查點（10 題跑通，2026-11-07）：
- 指令：`uv run python scripts/generate_synthetic.py`（`--variants N --max-rounds N --seeds ...`）。
- 程式：`src/text2sql/datagen/`（`checks.py` 規則層、`pipeline.py` 分階段管線與報告、`llm.py` 本機模型 JSON 呼叫）。
- 種子：`data/seeds/golden_seeds.jsonl`（G01–G05，含 SQL 與預期圖表）、`data/seeds/guardrail_seeds.jsonl`
  （B01–B05：寫入、資料庫沒有×2、閒聊、越權）。保留項分 entity（程式逐字檢查）與 concept（交給標籤／語意層）。
- 每次執行輸出到 `data/synthetic/runs/<時間>/`：`round*/1_rewrite..4_semantic.jsonl`、`dataset.jsonl`、`manifest.json`、`report.md`。
- 最新一次 `20261107-071720`：30 題（放行 15、攔截 15），25 候選通過 20；規則層刷 5 題（皆模型混入簡體字）。
- 已知：標籤層與語意層在實跑中刷掉 0 題，攔截能力未被證明；B02 漏「第一季」由人工抽查發現（judge 放過），已改成 entity。
- 下一步候選：放入刻意偏離的改寫測試標籤／語意層；擴大種子；把 dataset 接到 eval 分別測兩條路線（攔截率、誤擋、答對率）。
- 注意：生成會用本機模型，不可與其他 session 的 smoke／eval 同時執行。

舊模板與新系統已合併（2026-11-07）：
- 黃金種子改由 `uv run python scripts/build_golden_seeds.py` 從舊模板產生（`src/text2sql/datagen/templates.py`）。
  43 個題型、81 個實體版本，每個都有 SQL；答案先用 SQL 算，再與舊腳本 pandas 從 CSV 算的答案比對，81/81 一致（含圖表 X 軸順序）。
- 舊 132 題中 51 題只是換地區／國家別名，不再當種子；別名改為實體的可用寫法，讓改寫自然使用。
- 預設每個題型 1 筆（`--per-template N` 或 `--all` 可調），挑選時輪流使用不同實體：42 筆，涵蓋 4 區域、3 類別。
- 排除 I 題：題目說「在 North America 區域」，舊答案卻把這些客戶在其他地區的購買也算入（487/494 位客戶跨地區）。
- 手寫的 G01–G05 已由模板種子取代（保留在 git 歷史與測試 fixture）。
- 跨年季度標籤依專案慣例用 `2023-Q1`；舊資料是 `2023 Q1`，比對時視為相同。
- 實跑 `20261107-082751`（47 種子，883 秒，364 次呼叫）：141 題（放行 126、攔截 15），110 候選通過 94；
  規則層刷 9（簡體字），語意層刷 7（其中 5 次正反順序判斷不一致）。
- 人工抽查發現：M「每個區域**銷量**第一的品類」把銷售額換成數量，judge 放過；E「哪一季賣得最多」、K「買貨最多」
  在金額與數量間有歧義。已改為程式檢查：種子帶 `metric`（amount／quantity／count），規則層擋錯誤指標用詞與
  未寫明指標的「賣得最多」類說法，改寫 prompt 也先說明指標（`checks.metric_check`）。
- 指標檢查後實跑 `20261107-084843`（995 秒，377 次呼叫）：116 候選通過 94；指標規則實跑刷 0（prompt 已先避開，
  規則是保底，由離線測試證明能擋）；標籤層首次刷 1（「每個產品各賺了多少錢」被判為利潤＝資料庫沒有，正確）；
  語意層刷 10，部分過嚴（「營收增加」「3C」被判意圖改變），代價是多樣性而非正確性。
- 放行與攔截比例 126:15，攔截種子需要擴充才能接近演講的各半。

## 方向與優先順序（2026-11-07 討論）

主軸：**先讓合成測試集可信，再拿它評估模型。** 依序：

1. **平衡攔截／放行**：目前種子 42 放行（黃金）對 5 攔截（B01–B05），攔截率無意義。
   補攔截種子（寫入、資料庫沒有、閒聊、越權）到接近放行數。只改種子檔，不需模型。
2. **證明檢查層有效**：實跑中標籤層、語意層刷掉 0 題，judge 放過 B02 漏「第一季」。
   刻意放入偏離原意的改寫，確認會被刷掉；刷不掉就修檢查層。
3. **接上 eval 並取得分數**：dataset 接到 eval，用 Swift 1.5 跑 27B 路線，
   看攔截率、誤擋率、放行題答對率；同時在 Swift 上重跑完整 132 題 eval 與 8 案例 smoke。

暫不做：Codex 路線（未續費）、開 PR 合併 master（等第 3 步數字出來，作為 PR 證據）、
永久 session 儲存（與目前目標無關）。
`20261107-082751` 這輪生成（42+5 種子）讓它跑完，當補種子前的對照。

### 介面：維持 Streamlit

- 瓶頸在測試集與評估，不在介面；Streamlit 足以支撐單人展示問答、表格、Plotly。
- 後端仍是 InMemoryRunner，只換前端解決不了掉對話與多人問題。
- 重新考慮的條件（任一成立）：多人同時使用／部署給別人；需要登入與保留對話紀錄；
  需要即時顯示 planner → SQL → reviewer 每一步且 Streamlit 明顯卡頓。屆時主要工作是後端
  （API 層 + 永久 session），前端選型是次要。
- 可做的改進（取代原本「可展開區塊顯示 SQL」的想法）：**可點擊的執行流程圖**，見下節。

### 介面改進：可點擊的執行流程圖（2026-11-07 已實作，見 docs/ARCHITECTURE.md「執行紀錄與流程圖」）

目標讀者是使用者與面試官，他們最想知道：
1. 哪裡是 LLM、哪裡是程式（本專案賣點：模型只負責想，程式負責做）。
2. 每一步實際看到什麼、交出什麼（證明答案不是模型編的）。
3. 出錯怎麼處理（reviewer 拒絕、SQL 重試、報告重寫）。
4. 每一步花多少時間（本機 27B 的瓶頸在哪）。

呈現：
```
[問題] → (agent_sql) → [step_run_sql] → (agent_review) → (agent_report) → [step_publish] → [step_draw_chart]
            ↑________ 被拒絕／SQL 錯誤時重試 ________|
```
- 圓角框 = LLM（`agent_`），方框 = 程式（`step_`）；通過綠色、拒絕／失敗紅色，重試標 ×2。
- 點任一框，下方顯示該步的輸入、輸出、耗時。
- 每則回答下方有「查看執行流程」可展開，歷史題目也看得到。

做法：
- Chat 頁已逐一讀 `stream_turn` 的 event（`web_ui/pages/1_Chat.py`），每個 event 有 author 與
  state_delta；據此記錄每步的輸入、輸出、時間成「執行紀錄」，跟回答一起存進訊息歷史。
- 單一自含 SVG + 少量 JS 元件嵌入 Streamlit，點擊在瀏覽器內處理，不需額外套件、不觸發整頁重跑。
- 執行紀錄先用假 event 離線測試；真實跑一題需本機模型空閒時驗證。
- 實作結果：controller 直接記錄每步（比從 event 反推可靠），`tests/test_trace.py` 6 個測試；
  Streamlit AppTest 以假模型驅動 Chat 頁，確認流程圖出現；亮色、暗色、380px 手機截圖無 JS 錯誤。
- **尚未驗證**：用本機模型真實跑一題看流程圖（合成資料生成佔用模型時不可做）。
- Codex 路線只有「Codex → tool_query_sales → step_publish」，流程圖依 BACKEND 切換。

### 命名規則：名字要看得出是 Agent、Tool 還是程式步驟（2026-11-07 已實作，agent 與函式全改）

一律用小寫開頭前綴（全大寫保留給常數，例如 `MAX_QUERY_ROWS`）：

| 類型 | 意思 | 前綴 | 現在 → 改成 |
|---|---|---|---|
| Agent（LLM） | 由模型判斷、產生內容 | `agent_` | `sql_planner` → `agent_sql`；`semantic_reviewer` → `agent_review`；`report_writer` → `agent_report` |
| Tool（模型呼叫） | 模型自己決定要不要呼叫 | `tool_` | Codex MCP `query_sales_data` → `tool_query_sales` |
| 步驟（程式，一定執行） | controller 固定呼叫，不經模型判斷 | `step_` | `execute_query` → `step_run_sql`；`complete_answer` → `step_publish`；`render_chart` → `step_draw_chart` |

- 範圍：agent、MCP tool、step 函式、tests、CLAUDE.md、README、ARCHITECTURE 全部改名；離線測試通過。
- MCP tool 改名會改到 Codex 看到的名稱；Codex 目前無法實測（只有離線測試）。
- `docs/semantic-planner-truncation.json` 等歷史紀錄保留舊名 `sql_planner`，不改。

### 待補：ASP「平均單價」定義

`DOMAIN` 只寫 ASP 是單價，沒定義平均單價用 `AVG("ASP")`（簡單平均）還是
`SUM("Total Revenue")/SUM("Quantity")`（加權平均），reviewer 也無規則判斷。
先查 eval ground truth 用哪一種，再補進 `prompts.py` 的 `DOMAIN`。

## 使用者偏好（本專案相關）

- 回覆用繁體中文；一次只問一個問題（使用者常用語音回答）。
- 不問工程細節，只問目標、檢查點、完成標準。
- 宣稱完成要附證據（指令輸出、測試結果）。
- 本機模型一次只能處理一個請求：模型相關工作不可平行執行。
- 端點：`http://127.0.0.1:18080/v1`，模型 `qwen3.8-27b-iq3-xxs`。

## 複習資料

`docs/review/` 有 5 頁 60 分鐘複習頁面（用 answer-me-with-html 的 am CLI 產生，未安裝該 skill）：
1. 專案全貌　2. 27B 路線　3. Codex 路線　4. 驗證方式　5. 自我測驗與現況。
頁面內容描述的是合併前兩個分支的狀態。
