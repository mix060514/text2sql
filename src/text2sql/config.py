import os

# Pi Agent's verified model alias and local OpenAI-compatible endpoint.
LLAMA_CPP_MODEL = os.getenv("LLAMA_CPP_MODEL", "openai/swift15-27b-iq3-xxs")
LLAMA_CPP_API_KEY = os.getenv("LLAMA_CPP_API_KEY", "aaa")
LLAMA_CPP_API_BASE = os.getenv("LLAMA_CPP_API_BASE", "http://127.0.0.1:18080/v1")
CODEX_BINARY = os.getenv("CODEX_BINARY", "codex")
CODEX_MODEL = os.getenv("CODEX_MODEL", "gpt-6-sol")
CODEX_REASONING_EFFORT = os.getenv("CODEX_REASONING_EFFORT", "medium")

# Analysis backend: "qwen27b" (local model, default) or "codex" (Codex CLI).
BACKEND = os.getenv("TEXT2SQL_BACKEND", "qwen27b").strip().lower()
if BACKEND not in ("qwen27b", "codex"):
    raise ValueError("TEXT2SQL_BACKEND must be 'qwen27b' or 'codex'.")
WORKFLOW_VARIANT = "strong" if BACKEND == "codex" else "qwen27b"
ANALYSIS_MODEL = CODEX_MODEL if BACKEND == "codex" else LLAMA_CPP_MODEL

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
MAX_QUERY_CALLS = int(os.getenv("MAX_QUERY_CALLS", "8"))

if min(MAX_LLM_CALLS, AGENT_TIMEOUT_SECONDS, QUERY_TIMEOUT_SECONDS,
       MAX_QUERY_ROWS, MAX_RESULT_BYTES, SQL_ATTEMPTS, REPORT_ATTEMPTS,
       MAX_QUERY_CALLS) <= 0:
    raise ValueError("Agent and query limits must be positive.")
