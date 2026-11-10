# Repository guidance

This branch has one backend: the "qwen27b" deterministic controller with three Qwen 27B
LlmAgents (qwen_workflow.py; default model since 2026-11-07 is Swift 1.5 `swift15-27b-iq3-xxs`,
a shorter-thinking Qwen3.8-27B post-train). TEXT2SQL_BACKEND (config.BACKEND) accepts only
"qwen27b". The Codex CLI backend was removed on 2026-11-08 (the service is external-facing;
a CLI agent subprocess is not a suitable serving path). It remains in git at 96f9615.
Read docs/HANDOFF.md first (current state, open issues, next task), then README.md and
docs/ARCHITECTURE.md before changing the workflow. The six terms used in
discussion (資料庫、題目、標準答案、考卷、問答系統、成績單) and their old names (M1–M5, seeds, banks) are in docs/MODULES.md (diagram: docs/architecture.svg).

## Commands

- `uv sync --locked`
- `uv run streamlit run web_ui/app.py`
- `uv run python scripts/run_planned_agent.py 'EMEA 的總銷售額是多少？'`
- `uv run python scripts/verify_all.py` (offline, standard-library unittest)
- `uv run python scripts/smoke_workflow.py --output docs/smoke-qwen27b.json` (live model)
- `uv run python eval/eval_e2e.py --exam [--no-reuse] [--concurrency 4]` (live model; the exam in data/exam: 84 questions,
  each asked once, graded by code against the SQLite oracle and expected chart, no judge)
- `uv run python scripts/build_exam.py` (offline; rebuilds data/exam from the PICKS blueprint in that script)
- `uv run python eval/eval_e2e.py --limit 5` (historical eval_set_v3 run with the LLM judge; retired 2026-11-10)
- `uv run python scripts/build_golden_seeds.py` (offline; golden seeds from legacy templates, SQL checked against legacy pandas)
- `uv run python scripts/build_guardrail_seeds.py` (offline; 42 block seeds, prints a review table)
- `uv run python scripts/build_complex_seeds.py` (offline; 34 SQL-first complex seeds, SQL checked against pandas on the CSV)

## Boundaries

- Models generate SQL and explanation/chart intent; code executes mandatory actions.
- tools.read_sales enforces one SELECT, SQLite read-only/authorizer, query deadline,
  row/byte limits and typed results. sqlparse is not a syntax or security validator.
- SQL success permits a legitimate empty result. Failed SQL cannot reach report generation.
- contracts.AnswerSubmission validates output shape. step_publish validates real columns,
  numeric values and required charts, then renders and commits without an echo agent.
- SQL attempts: 3 total; report attempts: 2 total. Only explicit validation/execution
  errors retry. Transport/runtime errors stop the invocation.
- runtime.stream_turn owns the invocation deadline and ADK max_llm_calls; consume ALL events.
  An individual agent's is_final_response() does not indicate workflow completion.
- context.start_turn clears query/answer/chart artifacts; carries only previous successful
  question/SQL for follow-up intent. No stale result is accepted as fresh evidence.
- All public answers use state.final_answer after workflow_status == completed.

## Evaluation

The current evaluation is the exam (data/exam, scripts/build_exam.py): under 100 questions in five groups
(基本查詢、畫圖、多步驟查詢、不同叫法、該拒答), each question asked once and testing a different point, graded by
code (eval/synthetic.py): rows against the SQLite oracle, plus the expected chart where the item has one.
To test something new, add a row to PICKS; do not ask the same question again in other words.
The earlier sets (16 template, 27 alias, 354 synthetic with every question in three wordings) are retired.
M1 (generate_ground_truth.py) and data/eval_set_v3.jsonl are retired (2026-11-10); v3 stays only as the record
behind the optimization page. Do not compare a changed database against old ground truth, or treat an LLM judge as truth.
Smoke compares executed data/actual chart generation with deterministic SQLite oracles.
Top-N ties are ordered by metric DESC, name ASC. Time labels are Q1..Q4 for one-year
quarters, YYYY-Qn for multi-year quarters, YYYY-MM for months.

## History

The rewrite program ("M2": scripts/generate_synthetic.py, datagen/pipeline.py, datagen/llm.py) was removed on
2026-11-10; it is in git at c7c60db. Its runs under data/synthetic/runs stay as the evidence behind the optimization
page (eval_e2e.py --synthetic RUN_DIR can still replay one). New exam questions are written directly in natural wording.
The removed Codex backend (codex_workflow.py, codex_backend.py, mcp_server.py) is at 96f9615.
step_publish now always requires the semantic review. The source branches
codex/text2sql-qwen27b and codex/text2sql-strong are kept unchanged.
The legacy implementation is available at fd66447.
