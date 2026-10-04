# 驗證結果

下方既有比較與 smoke 檔案是語義 reviewer 加入前的歷史紀錄；
不能當成目前三階段 27B workflow 的驗證結果。最新修復結果見文末。

兩版都已可在這台主機執行。原工作區保留 27B 版，強模型版在相鄰
`text2sql-strong` worktree。兩版資料庫未修改，皆與 eval_manifest 的 SHA-256 相符。

| 驗證 | 強模型版 | 27B 版 |
|---|---:|---:|
| 最終離線測試 | 32/32 | 33/33 |
| 同一組 8 個代表案例 | 8/8 | 8/8 |
| 完整資料表變更後，再測月份趨勢與空結果 | 2/2 | 2/2 |
| eval_e2e 介面整合（單一 EMEA 案例） | 1/1 | 1/1 |
| 分析模型 | gpt-6-sol，medium | qwen3.8-27b-iq3-xxs |
| 格式 / 資料版本 | Codex CLI 0.157.1、ADK 1.14.1、SQLite seed-43、eval_set_v3 | ADK 1.14.1、相同 DB / dataset |

8 個案例包含地區別名、銷售量排名、訂單數排名、年份比較、
單一年份季度標籤、跨年月份趨勢、區域佔比及真正的空結果。
直接比對 SQLite oracle 與實際圖表，不用 LLM judge 判斷這些 pass/fail。

離線測試包含實際 SQLite 權限拒絕、語法錯誤、CTE/window functions、
結果上限、查詢期限、錯誤後清除資料、schema 修正、重試耗盡、整體期限、
完整事件串流、跨回合圖表隔離與完整資料表。
強模型版另外驗證不做 application retry、MCP query budget、執行 receipt
與取消後停止程序。

## 實測紀錄

- [27B 完整案例](smoke-qwen27b.json)
- [強模型完整案例](smoke-strong.json)
- [27B 最終資料表案例](smoke-qwen27b-final.json)
- [強模型最終資料表案例](smoke-strong-final.json)
- [27B 初輪原始紀錄](smoke-qwen27b-initial.json)
- [27B eval_e2e 整合紀錄](eval-integration-qwen27b.json)
- [強模型 eval_e2e 整合紀錄](eval-integration-strong.json)

eval_e2e 的單案執行確認分析、獨立 judge 與結果輸出串接正常。
這個 judge 評分不作為整體模型準確率的依據。

完整 8 案例的一次觀測總時間：強模型 152.58 秒、27B 326.58 秒。
這不是受控的速度 benchmark；不同模型、硬體、服務負載與網路影響時間。
兩個最終資料表案例另外記錄，沒有混進上述時間。

8 案例這一輪：強模型每題使用一次 sales 查詢；
當時的 27B 每題使用一次 SQL planner 和一次 report writer，沒有觸發 retry；
目前正常路徑另有一次 semantic reviewer。
Codex 內部模型呼叫次數未量測，因此不能把「一次 Harness invocation」
或「一次查詢」當成「一次模型呼叫」。

## 找到與修正的問題

初輪記錄顯示 5/8，但不能把三個紅燈都當成模型失敗：

1. Canada 問「客戶是誰」，模型回傳正確名字，沒有多選訂單數。
   原 oracle 多要求一個不必要的欄位；修正比較規則，仍檢查 distinct orders 指標。
2. 月份趨勢除了正確營收，也回傳了真實訂單數與銷售量。
   改為驗證被畫出的維度/指標與順序，避免把額外正確欄位誤判為錯誤。
3. 年份總計比較被模型選成 line，是這次真正觀察到的決策錯誤。
   明確定義「年份總計比較用 bar、明確時間趨勢才用 line」，再次實測通過。

最後將完整資料表交給 deterministic result_table 產生。模型只寫解釋，
不再靠 prompt 保證每一筆資料都被重新抄寫。變更後再測 24 個月份與空結果，
兩版皆通過；整個 8 案例集的紀錄保留在資料表變更前的版本。

本機 config 的 gpt-6.1-sol 被 CLI 的 ChatGPT 登入拒絕。
gpt-6-sol 已實際測通，所以強模型版明確預設為它，沒有靜默模型 fallback。
模型可用性是帳號/客戶端的設定問題，不是 SQL 推理問題。

## 範圍與限制

這些案例證明兩條流程能執行、資料與圖表符合代表案例的 oracle，
不能證明所有自然語言敘述正確，也不能證明 27B 的整體推理能力等同強模型。
硬 gate 保證執行證據、唯讀限制、資料完整性與 schema；
目前 27B 另要求本次 semantic reviewer 明確接受，但 reviewer 仍可能判錯；
指標理解或不當 SQL 分組等語意問題仍需 SQLite oracle 評估。

沒有重跑全部 132 個案例，沒有部署或 push。
原本 Docker image 未包含 Codex CLI/登入；強模型版目前驗證的是主機執行方式。
Session 仍使用 InMemoryRunner，沒有加入永久 checkpoint 或 scheduler。


## 問候回合修正

使用者實測 `hello` 時，27B planner 先生成 `SELECT 'hello'`，被資料來源驗證拒絕後，
重試改查訂單總數。這是把「每個輸入都必須查資料」誤當成系統規則，
再加上重試偏離原意；SQL 成功不能證明回答符合問題。

兩版共用的 `context.start_turn` 現在只對整句明確問候直接回覆，
由 ADK callback 跳過 agent body，不呼叫 LLM 或 DB。
`hello，請查南美營收` 仍走查詢流程。問候會清除舊結果／圖表，
並保留最近成功查詢的問題與 SQL 作後續追問意圖，不冒充本回合資料。
27B planner 另加「修正原問題，不得改查未要求統計」的提示；
這項提示不是一般語意正確性的硬性驗證。

修正後離線測試：27B 36/36、強模型 35/35。
新增回歸測試證明問候跳過整個查詢 workflow、混合問候與查詢不被攔截、
連續問候不殘留圖表且不丟失追問意圖。兩版 CLI 的 `hello` 均回覆問候，SQL 為空。
此修正只涵蓋明確問候，並非完整的非查詢意圖分類器。

## 27B 語義審查修復（2026-10-04）

本次只修改 `codex/text2sql-qwen27b`，沒有修改強模型 worktree 的 Harness。
應用模型維持 `qwen3.8-27b-iq3-xxs`，endpoint 維持 `http://127.0.0.1:18080/v1`。

新流程是 planner → 實際唯讀執行 → semantic reviewer → report → 發布 gate。
reviewer 接收原問題、追問意圖、業務定義、SQL、完整 columns / rows；
strict `SemanticReview` 要求 boolean `accepted` 和非空 `reason`。
拒絕理由或 schema 錯誤回饋 planner，與 SQL 錯誤共用 3 次查詢嘗試；
報告 / 圖表另有 2 次嘗試，不重查資料。只有本次查詢成功且明確接受才寫報告；
發布再核對接受的 SQL / columns / rows。每次嘗試、回合與失敗清除舊接受及產物。
問候 callback fast path、完整事件消費、唯讀與資料限制均保留。

維持原 8 次模型呼叫 / 300 秒總期限：一般成功路徑由 2 次增加到 3 次，
最壞 3 組 planner＋reviewer 加 2 次 report 剛好 8 次；離線測試覆蓋此路徑。
較慢服務仍可能耗盡期限；期限與呼叫數例外也清除接受、結果及圖表並回報失敗。

離線：`uv run python scripts/verify_all.py`，**43/43 通過**。新增回歸覆蓋：
合法錯誤指標拒絕後執行正確指標、全部拒絕不呼叫 report / 不發布、
review JSON / boolean / reason 解析失敗不通過、跨回合接受隔離、
拒絕後遇到 schema / SQL 失敗不沿用結果、發布時拒絕未接受或不同執行結果、
8 calls 最壞路徑；原 SQL / 圖表修正、問候、完整表格與邊界斷言保持通過。

真實語義專項：`uv run python scripts/verify_semantic_live.py`，**5/5 通過**。
[完整紀錄](semantic-qwen27b.json)保留實際 SQL、rows、審查、回饋與時間：

- 總銷售量：SQLite `SUM(Quantity)` = 137,430。
- distinct orders：SQLite `COUNT(DISTINCT "Order ID")` = 30,000。
- 刻意注入可執行的錯誤 SQL `COUNT(DISTINCT "Order ID") AS Units`：
  真實 Qwen reviewer 拒絕，真實 Qwen planner 收到理由後重寫，第 2 次實際查得 137,430。
  只有第一次 planner 輸出是 fault injection，其餘 reviewer / planner / report 皆用真實 Qwen。
- 南美銷售量 7,020，接續「那北美呢？」重新查得 55,128，均符合獨立 SQLite oracle。

原有 8 案例初輪（planner 額度 2,048）：`uv run python scripts/smoke_workflow.py --output docs/smoke-qwen27b-semantic.json`，
**7/8 通過（exit 1）**，不以歷史 8/8 冒充本次結果。
[本次完整紀錄](smoke-qwen27b-semantic.json)：地區營收、銷售量排名、distinct orders 排名、
年份比較、季度趨勢、佔比及空結果通過，實際 rows / 圖表與原 SQLite oracle 比對。
季度題的首個可執行 SQL 把 3 月算成 Q2、12 月算成 Q5；真實 reviewer 指出季度公式錯誤，
回饋後第 2 次 SQL 改正，結果與 Q1–Q4 oracle 一致。此處沒有注入錯誤輸出。
月份趨勢首輪 3 次嘗試耗盡，最終未取得可解析 SQLPlan，`SQL_ATTEMPTS_EXHAUSTED`，
report_attempts = 0、SQL / rows / charts 清空，只回傳失敗訊息，未發布分析。
[單題診斷紀錄](semantic-planner-truncation.json)的 3 個 planner 事件皆顯示 `candidates_token_count = 2048`、`content.parts = []`：
輸出額度用滿而沒有 SQL 文字。因而將 planner `max_output_tokens` 從 2,048 提高到 4,096；
reviewer 仍是 2,048、report 仍是 4,096，模型與 endpoint 不變。
這只增加單次規劃輸出的空間，8 calls / 300 秒、每次模型請求 timeout 120 秒都維持。
若模型仍不提供有效輸出或超時，照樣失敗，不發布分析；沒有修改 expected。

提高 planner 額度後，只重跑相同月份案例：
`uv run python scripts/smoke_workflow.py --case 'Monthly trend' --output docs/smoke-qwen27b-monthly-repair.json`，
**0/1（exit 1，260.51 秒）**，[重測紀錄](smoke-qwen27b-monthly-repair.json)。
這次進入 review，但沒有有效 SemanticReview，下一個查詢嘗試遇到
`litellm.Timeout: Request timed out after 120 seconds`；report_attempts 仍為 0，
結果 / 圖表 / 接受狀態清空。增加 planner 額度並未解決這題的整體可靠性；
沒有再增加 reviewer 額度或總期限，也沒有宣稱月份 oracle 已通過。
最終保留 planner 4,096 額度以容納較長的規劃輸出，但模型仍可能無輸出或超時。
語義專項 5/5 與 8 案例初輪紀錄是在此單次 token 調整前取得；
調整後離線 43/43 通過，只重跑上述月份案例，沒有再重跑完整 8 題。


DB 與 dataset SHA-256 仍符合 `data/eval_manifest.json`，沒有重生或改 expected。
這次沒有重跑完整 132 案例或強模型版，也沒有 push、部署或啟停 Streamlit。
使用者需要重啟既有 Streamlit 才載入新 agent，並重新建立聊天 session。

reviewer 仍可能錯誤接受或錯誤拒絕，與 planner 使用同一模型也可能有共同盲點。
這是線上語義攔截與有限修復，不是一般 SQL 語義正確性或回答敘述的數學保證。
