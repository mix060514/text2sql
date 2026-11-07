# Text-to-SQL：兩種 Agent 架構

這個分支（`codex/text2sql-unified`）合併了兩條路線，用環境變數切換：

| `TEXT2SQL_BACKEND` | 路線 | 檔案 |
|---|---|---|
| `qwen27b`（預設） | 本機 Swift 1.5（Qwen3.8-27B 後訓練版），程式控制 planner → 執行 → reviewer → report | `planned_agent/qwen_workflow.py` |
| `codex` | 一次 Codex CLI 呼叫，模型透過唯讀 sales MCP 工具查詢 | `planned_agent/codex_workflow.py`、`codex_backend.py`、`mcp_server.py` |

`planned_agent/agent.py` 只負責選擇路線。兩條路線共用 `tools.py`、`contracts.py`、`prompts.py`。
發布 gate `step_publish` 預設要求 reviewer 接受；Codex 路線沒有 reviewer，
改用 MCP receipt 作為執行證據，呼叫時傳 `require_review=False`。

```bash
export TEXT2SQL_BACKEND=codex   # 需要已登入且有額度的 Codex CLI
```

Codex 路線在此分支只有離線測試；Codex 帳號未續費，沒有實際呼叫。
舊版保存在基準 commit `fd66447`；原兩個分支 `codex/text2sql-qwen27b`、`codex/text2sql-strong` 保留不動。
以下內容描述 27B 路線。

兩版共用 SQLite 銷售資料、查詢限制、圖表驗證及相同的 live smoke cases。
設計差異見 [架構比較](docs/ARCHITECTURE.md)，實測紀錄見 [驗證結果](docs/VALIDATION.md)。

## 27B 版流程

```text
使用者問題
  → agent_sql（Qwen，SQLPlan schema）
  → step_run_sql（程式直接驗證、執行唯讀 SQL）
      └─ SQL 或 schema 錯誤：回饋 agent_sql
  → agent_review（同一 Qwen，strict accepted boolean / reason）
      └─ 意圖不符或 review 解析失敗：回饋 agent_sql；查詢合計最多 3 次嘗試
  → agent_report（Qwen，AnswerSubmission schema）
      └─ schema / 圖表參數錯誤：只重試 agent_report，最多 2 次總嘗試
  → step_publish（程式驗證成功條件、產生完整資料表）→ step_draw_chart（Plotly）、提交回答
```

只有 agent_sql、agent_review、agent_report 三個模型步驟（命名：agent_ = LLM、tool_ = 模型呼叫的工具、step_ = 程式一定執行）；Chat 頁可展開「查看執行流程」，點步驟看輸入／輸出；沒有 region/checker/executor/renderer/echo Agents。
模型不需要決定「要不要執行已生成的 SQL」；那是 controller 的固定下一步。
完整資料表由程式產生，模型只寫解釋。
SQL 執行成功且本次 reviewer 明確接受，才能產生報告與發布。
每次嘗試清除舊接受狀態與資料；發布再核對被接受的實際 SQL、columns、rows。
reviewer 仍可能判錯，接受不代表語意的數學保證，因此仍保留 SQLite oracle 評估。

## 執行

2026-11-07 起預設模型改為 **Swift 1.5**：UkisAI 以 Qwen3.8-27B 後訓練、思考 token 較少的版本，
本機使用 [GSQ-RCO IQ3_XXS-mtp GGUF](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27B-GSQ-RCO-GGUF)
（10.44GB，含 MTP 推測解碼），router alias 為 `swift15-27b-iq3-xxs`。
換模型的原因：原版思考很長，月份趨勢題的 planner 用光輸出額度、reviewer 超過 120 秒。
OpenAI-compatible API 為 `http://127.0.0.1:18080/v1`。原版 `qwen3.8-27b-iq3-xxs` 仍在 router 上，
用 `LLAMA_CPP_MODEL=openai/qwen3.8-27b-iq3-xxs` 即可切回。

```bash
uv sync --locked
uv run python scripts/run_planned_agent.py '南美 的總銷售額是多少？'
uv run streamlit run web_ui/app.py
```

若模型服務不同，使用環境變數覆寫：

```bash
export LLAMA_CPP_MODEL=openai/swift15-27b-iq3-xxs
export LLAMA_CPP_API_BASE=http://127.0.0.1:18080/v1
export LLAMA_CPP_API_KEY=aaa
```

LiteLLM 的 `openai/` 是 transport 前綴，實際模型 ID 不含該前綴。
Docker 中的 localhost 是容器自己；使用 Compose 時，必須將
`LLAMA_CPP_API_BASE` 設成容器能到達且有此模型的服務 URL。
Vision 頁面沿用獨立的 `LLAMA_CPP_VL_*` 設定。

## 驗證

```bash
# 離線：真實 SQLite、圖表與注入失敗的 workflow 測試，不呼叫模型
uv run python scripts/verify_all.py

# 線上：同一組 8 個問題，直接與現在的 DB 比對，不使用 LLM judge
uv run python scripts/smoke_workflow.py --output docs/smoke-qwen27b-semantic.json

# 真實 reviewer：銷售量 / 訂單數、錯誤 SQL 注入與追問隔離
uv run python scripts/verify_semantic_live.py

# 較廣的評估集；這兩個命令會呼叫模型及 LLM judge
uv run python eval/eval_e2e.py --limit 5
uv run python eval/eval.py
```

兩個 eval 都改用與目前 DB 配對的 `eval_set_v3.jsonl`。
`data/eval_manifest.json` 驗證 DB / dataset 的 SHA-256，避免舊 ground truth
被拿來評估新資料；歷史結果不能直接當成本次模型準確率。
smoke 驗證實際資料與圖表，不能保證自然語言回答中的每一個敘述。

## 閱讀順序

1. `src/text2sql/agents/planned_agent/agent.py`：固定順序與兩處有限重試。
2. `contracts.py`、`prompts.py`：模型輸入、輸出與業務定義。
3. `tools.py`：SQLite 權限、結果大小、圖表與發布條件。
4. `context.py`、`src/text2sql/runtime.py`：回合隔離與完整事件串流。

查詢最多 200 列、20,000 bytes、5 秒；超限即失敗，不接受截斷的成功結果。
每個 ADK invocation 最多 8 次模型呼叫、300 秒，維持原限制。
一般成功路徑使用 3 次呼叫；最壞為 3 組 planner＋reviewer 加 2 次 report。
planner 輸出額度為 4,096 token、reviewer 為 2,048；無有效輸出或超時則失敗。
目前月份 smoke 仍未通過，詳見 [實際驗證與限制](docs/VALIDATION.md)。
前一回合只攜帶問題與已執行 SQL 作追問上下文，本回合一定重新查詢。
目前使用 InMemoryRunner；沒有加入排程器或永久 session 儲存。

## 比較分支

```bash
git diff codex/text2sql-qwen27b..codex/text2sql-strong -- src/text2sql/agents/planned_agent
```

工作目錄已分開：27B 版在原 repo，強模型版在相鄰的 `text2sql-strong` worktree。
兩個分支皆為本機分支，尚未 push 或部署。

MIT License.
