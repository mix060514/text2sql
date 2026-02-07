import os

LLAMA_CPP_MODEL = os.getenv("LLAMA_CPP_MODEL", "openai/qwen3-4b-instruct-2507")
LLAMA_CPP_API_KEY = os.getenv("LLAMA_CPP_API_KEY", "aaa")
LLAMA_CPP_API_BASE = os.getenv("LLAMA_CPP_API_BASE", "http://localhost:8081")

LLAMA_CPP_VL_MODEL = os.getenv(
    "LLAMA_CPP_VL_MODEL", "openai/Qwen3VL-4B-Instruct-Q4_K_M.gguf"
)
LLAMA_CPP_VL_API_KEY = os.getenv("LLAMA_CPP_VL_API_KEY", "aaa")
LLAMA_CPP_VL_API_BASE = os.getenv("LLAMA_CPP_VL_API_BASE", "http://localhost:8082")

MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
