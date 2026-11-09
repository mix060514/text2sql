# 交接紀錄（2026-11-08 更新；以下 11-07 段落為歷史）

新 session 先讀這份，再讀 README.md 與 CLAUDE.md。

## 2026-11-08 優化與定版評估

所有數字由 `scripts/build_optimization_report.py` 從 `docs/optimization/` 的證據檔計算，顯示在 Streamlit「🛠️ 優化成果」頁。

### 做了什麼（依序）

| 修改 | 原因 | commit |
|---|---|---|
| SQL／審查思考上限、報告不思考；請求改串流並在結束時關閉 | 思考無上限＋非串流請求逾時後 llama.cpp 仍生成（幽靈請求佔住唯一推論槽） | 11-07 晚至 11-08 早 |
| LIKE 加入 SQL 允許函數 | 被授權器擋下 | 同上 |
| 評審改用同一套串流模型；`eval_e2e.py --rejudge` | 評審連鎖失敗 20 題被記 0 分 | 同上 |
| 季度公式寫入 DOMAIN；X 軸比對忽略分隔符；評審提示圖表另由程式檢查 | Q94、Q101–105、Q110 | 同上 |
| 同題型、不同內容的 few-shot（5 個，`SQL_FEW_SHOT`） | 跨年逐月、前 N 大客戶占比、期間成長率、累計占比、條件占比難一次解決 | c5de570、0c1536a |
| 複雜題人工複核檔 `docs/optimization/complex_review.json` | 標準答案過嚴（多帶欄位、標籤寫法、正負號）；程式只套用宣告的轉換再比對 | 0c1536a |
| 拿掉每次請求 120 秒，只留每題時限；SQL 思考上限 2048→4096，輸出上限＝上限＋2048 | 使用者決定只要一種時間限制；整題到期仍會關閉串流（有測試） | 4fd14d6 |
| 每題時限 300→600 秒（app 與兩條 eval 共用 `AGENT_TIMEOUT_SECONDS`） | 使用者決定；4 題並發時 300 秒不夠 | b18696b |
| `eval_e2e.py --concurrency N` | 用上伺服器的 4 路並發，同批題目約快 1.6 倍 | 787c215 |
| 複核規則加英文星期、可省略年份欄；`regression_notes.json` 逐題人工原因 | 定版由對變錯的題目要附可查證的原因 | 42ffbf4、b32122e |
| 並發與休眠測試（`scripts/bench_concurrency.py`） | 評估能否多人同時使用 | 3fb7cf1–c3d667f |

### 評估結果

| 評估 | 成功率 | 逾時率 | 證據 |
|---|---|---|---|
| 模板題 2026-11-07（修正前） | 65.9% | 13.6% | `template_2026-11-07_original.json` |
| 模板題基準（基礎設施修正後） | 92.4%（122/132） | 3.0% | `template_baseline.json`（436a573） |
| 複雜題基準 | 嚴格 81.4%（83/102）、複核後 97.1% | 1.0% | `complex_baseline.json`（a9062d2） |
| 模板題定版 | 93.9%（124/132），一次解決 94.6% | 2.3% | `template_fewshot.json`（e2e_2026-11-09_01-53-40） |
| 複雜題定版 | 嚴格 79.4%（81/102）、複核後 95.1%（97/102） | 2.0% | `complex_fewshot.json`（synthetic_complex_2026-11-09_00-45-08） |
| 攔截題定版（第一次） | 7.9%（10/126） | 27.0% | `block_final.json`（synthetic_block_2026-11-09_03-09-44） |

定版設定：`SQL_FEW_SHOT=true`、每題 600 秒、SQL 思考 4096、`--concurrency 4`；跑在臨時伺服器 port 18090
（router 同參數＋4 路、`--ctx-checkpoints 4 --cache-ram 2048`），router 設定未改。基準是一次 1 題，
所以定版的平均耗時（模板 136.7 秒、複雜 151.7 秒）是並發下每題的時間，不可直接比較。

由對變錯（原因在 `docs/optimization/regression_notes.json`，頁面「逐題變化」）：
- 5 題（模板 3、複雜 2）是 4 題並發下跑滿 600 秒；單獨重跑 37–136 秒全部答對（`timeout_reruns_alone.json`）。
- 日本在區域的名次 ×2：真 SQL 錯誤，ROW_NUMBER 在 `WHERE Country='Japan'` 之後算，名次永遠是 1。
- 下半年比上半年「多賣」×1：模型用數量（數字正確），標準答案用營收；題意歧義。
- 2024 類別占比圓餅 ×1：數字對，切片按名稱排而非金額由大到小。
- 修好：模板 6 題（跨年逐月 4）、複雜題 APAC 月增率 3 題（few-shot 的 LAG 範例）。

攔截題結論：流程沒有意圖判斷，四類（改資料、資料沒有、閒聊、注入）都直接寫 SQL。寫入全被 READ_ONLY 擋下，
資料安全；但平均 418 秒、會把詩寫進 `SELECT '…' FROM sales_data`、把過去成長率當「預測」。
**下一步候選（需使用者決定）：在 agent_sql 前加意圖判斷，直接拒答這四類。**

### 並發與休眠（MTP、64k、kv-unified，臨時伺服器 port 18090 實測）

| 路數 | 總吞吐量 | 單一請求平均 | 顯存 |
|---|---|---|---|
| 1 | 25.0 tok/s | 17.7 s | 13.9 GB |
| 2 | 30.6 tok/s（1.22×） | 25.4 s | 14.5 GB |
| 4 | 50.8 tok/s（2.03×） | 31.5 s | 15.7 GB |
| 6 | 啟動時顯存不足（差 885 MiB） | | |
| DFlash2 1 | 37.9 tok/s（MTP 1 路的 1.52×） | 9.9 s | 14.8 GB |
| DFlash2 2 | 59.1 tok/s（2.36×，勝過 MTP 4 路） | 14.5 s | 15.8 GB |
| DFlash2 4 | 放不下：草稿 KV f16／q8_0／q4_0 各差 165／88／46 MiB；48k＋無 mmproj 能啟動但第一個請求就 CUDA OOM 崩潰 | | |

DFlash2（2026-11-09）：`--model-draft Qwen3.8-27B-DFlash2-Q4_K_M.gguf --spec-type draft-dflash --spec-draft-n-max 5`，
草稿 KV f16，主模型仍是 Swift 1.5；接受率約 50–75%。只量速度，沒跑答對率評估。
**建議（需使用者決定）**：router 改用 DFlash2＋2 路，延遲與吞吐量都比目前的 MTP＋4 路好。

- 每種只有 12 個請求，2 路與 4 路的倍數要保守看待。
- 閒置 30 秒休眠：顯存 13.9 GB → 211 MiB；休眠後第一個請求 3.7 秒（一般 0.56 秒）。
- 建議：多人使用設 4 路；只有一人（面試 demo）時 1 路單請求最快。
- prompt 快取已生效（`cached n_tokens = 1886/2000`），固定開頭不需重算；prompt 處理約 190 tok/s。

### router 設定

- 設定：`/mnt/data1/llama-config/acer-two-profiles.ini`，systemd 系統服務 `llama-server-qwen`（不是 Docker），重啟需使用者 sudo。
- 2026-11-08 備份：`acer-two-profiles.before-text2sql-sleep-20261108.ini`。
- 使用者決定把 `[swift15-27b-iq3-xxs]` 改成 64k、`sleep-idle-seconds = 600`、`parallel = 4`、`kv-unified = true`；
  已生效（ps／`/slots` 確認 4 路、每路 65,536）。
- **未解決：主記憶體上限尚未加入 ini。** 4 路跑約 15 分鐘後 checkpoint（每份約 150 MiB、每路最多 32 份）＋8 GB
  prompt 快取用到 9.9 GB，主機在 HDD swap 上卡死（95c2389）。需要使用者在該段加 `ctx-checkpoints = 4`、
  `cache-ram = 2048` 並 `sudo systemctl restart llama-server-qwen`、`sudo swapoff -a && sudo swapon -a`。
  加上限的臨時伺服器跑完 6 小時定版評估，伺服器 RSS 最高 5.2 GB，swap 未增加。
  在加上限之前，router 的 swift15 不要長時間 4 路滿載。

### 尚待

- DFlash2＋2 路若要上 router，先跑一次模板題確認答對率不變。
- 是否把 `SQL_FEW_SHOT` 預設改為 true（定版結果支持；預設仍是 false）。
- 意圖判斷（攔截題）與日本名次類的「先排名再篩選」範例。
- 並發 2／4 路加測到每種 36 個請求，數字較可信。

## 現況

| 項目 | 狀態 |
|---|---|
| 目前分支 | `codex/text2sql-unified`（commit `19821ae` 起），已 push 到 `origin` |
| GitHub master | 仍停在 2026-02-06 `4761ae8`，尚未合併 unified，也沒開 PR |
| 路線 | 只有 `qwen27b`；`TEXT2SQL_BACKEND` 設其他值會報錯 |
| Codex 路線 | 2026-11-08 移除（服務對外，不用 CLI agent 子程序）；移除前程式在 `96f9615` |
| 舊分支 | `codex/text2sql-qwen27b`、`codex/text2sql-strong` 保留不動，只在本機 |
| 本機模型 | 2026-11-07 改為 Swift 1.5（`swift15-27b-iq3-xxs`，Qwen3.8-27B 後訓練、思考較短）；原版 `qwen3.8-27b-iq3-xxs` 仍可切回 |
| 強模型 worktree | `../text2sql-strong`（分支 `codex/text2sql-strong`），合併後已不需要；尚未移除 |

## 合併做了什麼（歷史；Codex 部分已於 2026-11-08 移除）

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

1. **Swift 1.5 上 smoke 8/8 通過、132 題 eval 已跑**（2026-11-07，見 `docs/VALIDATION.md` 末段）。
   人工核對後正確 112/132（84.8%），與 2026-02 v1.0.0 的 84.1% 持平；主要問題是 **18 題 120 秒逾時**
   （12 題多年月份趨勢、4 題前五大客戶占比等），另有 2 題答錯。prd 升級前要先處理逾時。
2. eval 的 LLM judge 有 20 題沒回傳結果，被記為 0 分；其中 19 題對照 ground truth 是對的。原始分數會低估。
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
- 攔截種子擴充為 42 筆（`uv run python scripts/build_guardrail_seeds.py`，`src/text2sql/datagen/guardrail.py`）：
  寫入 10、資料庫沒有 16、閒聊 8、越權 8；`label_source` 標「Claude 起草，待人工確認」，建置時印出勾選表。
- 管線新增「種子自我分類」：分類器先判斷種子本身，不一致列入報告供人工確認。
- 實跑 `20261107-090845`（84 種子，1895 秒，711 次呼叫）：252 題，放行 126、攔截 126；193 候選通過 168。
  種子自我分類抓到 B23「業務員 Amy 的業績排名」被判為可查（Amy 可被當成客戶名稱）→ 種子有歧義，待修。
  標籤層刷 4：B39／B40 在 write_request、not_in_data、prompt_injection 之間分類搖擺（都是攔截，但類別界線模糊）。
- 新增 `--resume-from <run>`：種子未變（`seed_fingerprint`：題目、標籤、類別、SQL）就沿用上次通過的改寫，
  規則層一律重檢（不呼叫模型），被刷掉的才重生成；prompt 版本變更或 `--recheck-model` 才重跑標籤／語意層。
  驗證：沿用 090845 → `20261107-094108` 131 秒、84 次（補算舊紀錄沒有的種子分類）；再沿用 → `20261107-094328` 0 次、1 秒。
- 修正（2026-11-07）：B23 改為「每位業務員在 2024 年的業績排名是多少？」（不帶人名）；分類器類別定義寫明
  prompt_injection 含冒充管理員解除唯讀、存取 sales_data 以外的表，並先判斷 injection 再判斷 write。
  prompt 版本拆成各層（`PROMPT_VERSIONS`），沿用時只重跑版本變更的層。
- 目前資料集：`data/synthetic/runs/20261107-095120/dataset.jsonl`（252 題，放行 126、攔截 126）。沿用 094328、
  只重判標籤層：257 次、382 秒；166 題沿用全數通過，B23 重新生成；種子自我分類不一致 0 筆。
- SQL 先行的複雜查詢（2026-11-07，細節見 `docs/SYNTHETIC_DATA_METHOD.md`「SQL 先行」）：34 題型
  （`src/text2sql/datagen/complex.py`，`uv run python scripts/build_complex_seeds.py`），每題 SQL 與讀原始 CSV 的
  pandas 實作逐列一致才收；新增第 4 層「SQL 對題」：本機模型看 SQL、實際結果前幾列與題目，判斷 SQL 是否剛好回答，
  種子本身也要過。抓到真種子問題（月增率 1 月應比前一年 12 月）。逐種子存檔，中斷的 run 可續跑。
- 目前資料集：`data/synthetic/runs/20261107-162208/dataset.jsonl`（354 題：放行 228＝舊黃金 126＋複雜 102，攔截 126）。
- 已知限制：SQL 對題層抓不到「上半年相比下半年」這類比較方向對調；judge 放過的改寫靠人工抽查補規則（已補 4 種）。
- 尚待：攔截種子仍無人工確認（`label_source` 註明）；資料集尚未接到 eval 測兩條路線（複雜題的 eval_type 多為
  dataframe，答案在 `ground_truth.rows`，接 eval 時需逐列比對）。
- SQL 先行複雜查詢（2026-11-07）：34 個題型（CTE、視窗函數、世代分析、年增／月增／季增、門檻），SQL 與讀原始 CSV 的
  pandas 逐列一致才收（`scripts/build_complex_seeds.py`）；新增 SQL 對題層（模型判斷 SQL 是否剛好回答題目）。
  目前資料集：`data/synthetic/runs/20261107-155802/dataset.jsonl`（352 題，放行 226、攔截 126）。
  細節與已知限制（judge 看不懂 LAG、上下半年方向對調抓不到）見 `docs/SYNTHETIC_DATA_METHOD.md`。

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
- 端點：`http://127.0.0.1:18080/v1`，模型 `swift15-27b-iq3-xxs`（2026-11-07 前為 `qwen3.8-27b-iq3-xxs`）。
- 模型部署：router 設定 `/mnt/data1/llama-config/acer-two-profiles.ini`（`--models-max 1`，一次只載入一個模型），
  改前備份 `acer-two-profiles.before-swift15-20261107.ini`；模型檔在 `/mnt/data1/models/`。

## 複習資料

`docs/review/` 有 5 頁 60 分鐘複習頁面（用 answer-me-with-html 的 am CLI 產生，未安裝該 skill）：
1. 專案全貌　2. 27B 路線　3. Codex 路線　4. 驗證方式　5. 自我測驗與現況。
頁面內容描述的是合併前兩個分支的狀態。
