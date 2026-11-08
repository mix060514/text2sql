import os

# Pi Agent's verified model alias and local OpenAI-compatible endpoint.
LLAMA_CPP_MODEL = os.getenv("LLAMA_CPP_MODEL", "openai/swift15-27b-iq3-xxs")
LLAMA_CPP_API_KEY = os.getenv("LLAMA_CPP_API_KEY", "aaa")
LLAMA_CPP_API_BASE = os.getenv("LLAMA_CPP_API_BASE", "http://127.0.0.1:18080/v1")

# Analysis backend: only "qwen27b" (local model). The Codex CLI backend was removed
# on 2026-11-08 because a CLI agent is not suitable for an external-facing service.
BACKEND = os.getenv("TEXT2SQL_BACKEND", "qwen27b").strip().lower()
if BACKEND != "qwen27b":
    raise ValueError("TEXT2SQL_BACKEND must be 'qwen27b' (the codex backend was removed).")
WORKFLOW_VARIANT = "qwen27b"
ANALYSIS_MODEL = LLAMA_CPP_MODEL

LLAMA_CPP_VL_MODEL = os.getenv("LLAMA_CPP_VL_MODEL", "openai/Qwen3VL-4B-Instruct-Q4_K_M.gguf")
LLAMA_CPP_VL_API_KEY = os.getenv("LLAMA_CPP_VL_API_KEY", "aaa")
LLAMA_CPP_VL_API_BASE = os.getenv("LLAMA_CPP_VL_API_BASE", "http://localhost:8082")

MAX_LLM_CALLS = int(os.getenv("MAX_LLM_CALLS", "8"))
AGENT_TIMEOUT_SECONDS = float(os.getenv("AGENT_TIMEOUT_SECONDS", "300"))
QUERY_TIMEOUT_SECONDS = float(os.getenv("QUERY_TIMEOUT_SECONDS", "5"))
MAX_QUERY_ROWS = int(os.getenv("MAX_QUERY_ROWS", "200"))
MAX_RESULT_BYTES = int(os.getenv("MAX_RESULT_BYTES", "20000"))
SQL_ATTEMPTS = int(os.getenv("SQL_ATTEMPTS", "3"))
REPORT_ATTEMPTS = int(os.getenv("REPORT_ATTEMPTS", "2"))
# The report writer only phrases already-validated rows; thinking made it take 20-85 s and
# pushed requests past the 120 s timeout. Set to true to restore model thinking there.
REPORT_THINKING = os.getenv("REPORT_THINKING", "false").strip().lower() in ("1", "true", "yes")
# Max thinking tokens per request (llama.cpp thinking_budget_tokens; -1 = unlimited). When
# the budget is used up llama.cpp forces the end of thinking and the model answers at once.
# The only time limit is AGENT_TIMEOUT_SECONDS per question; there is no per-request limit.
SQL_THINKING_BUDGET = int(os.getenv("SQL_THINKING_BUDGET", "4096"))
REVIEW_THINKING_BUDGET = int(os.getenv("REVIEW_THINKING_BUDGET", "1024"))
JUDGE_THINKING_BUDGET = int(os.getenv("JUDGE_THINKING_BUDGET", "1024"))
# Few-shot SQL examples for hard question types (prompts.SQL_EXAMPLES).
SQL_FEW_SHOT = os.getenv("SQL_FEW_SHOT", "false").strip().lower() in ("1", "true", "yes")

if min(MAX_LLM_CALLS, AGENT_TIMEOUT_SECONDS, QUERY_TIMEOUT_SECONDS,
       MAX_QUERY_ROWS, MAX_RESULT_BYTES, SQL_ATTEMPTS, REPORT_ATTEMPTS) <= 0:
    raise ValueError("Agent and query limits must be positive.")
