# Repository guidance

This branch uses a deterministic Text2SQL controller with two Qwen 27B LlmAgents.
Read README.md and docs/ARCHITECTURE.md before changing the workflow.

## Commands

- `uv sync --locked`
- `uv run streamlit run web_ui/app.py`
- `uv run python scripts/run_planned_agent.py 'EMEA 的總銷售額是多少？'`
- `uv run python scripts/verify_all.py` (offline, standard-library unittest)
- `uv run python scripts/smoke_workflow.py --output docs/smoke-qwen27b.json` (live model)
- `uv run python eval/eval_e2e.py --limit 5` (live model and LLM judge)

## Boundaries

- Models generate SQL and explanation/chart intent; code executes mandatory actions.
- tools.read_sales enforces one SELECT, SQLite read-only/authorizer, query deadline,
  row/byte limits and typed results. sqlparse is not a syntax or security validator.
- SQL success permits a legitimate empty result. Failed SQL cannot reach report generation.
- contracts.AnswerSubmission validates output shape. complete_answer validates real columns,
  numeric values and required charts, then renders and commits without an echo agent.
- SQL attempts: 3 total; report attempts: 2 total. Only explicit validation/execution
  errors retry. Transport/runtime errors stop the invocation.
- runtime.stream_turn owns the invocation deadline and ADK max_llm_calls; consume ALL events.
  An individual agent's is_final_response() does not indicate workflow completion.
- context.start_turn clears query/answer/chart artifacts; carries only previous successful
  question/SQL for follow-up intent. No stale result is accepted as fresh evidence.
- All public answers use state.final_answer after workflow_status == completed.

## Evaluation

Both eval scripts use data/eval_set_v3.jsonl, matched to the current DB by eval_manifest.json.
Do not compare a changed database against v2 ground truth, or treat an LLM judge as truth.
Smoke compares executed data/actual chart generation with deterministic SQLite oracles.
Top-N ties are ordered by metric DESC, name ASC. Time labels are Q1..Q4 for one-year
quarters, YYYY-Qn for multi-year quarters, YYYY-MM for months.

## Other branches

codex/text2sql-strong delegates reasoning/tool loops to Codex CLI and exposes read_sales
through an isolated MCP tool. ADK there is a UI/session adapter, not the reasoning runtime.
The legacy implementation is available at fd66447.
