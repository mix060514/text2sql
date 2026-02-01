import os
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
# The model name might need to match what LiteLLM expects or just be a placeholder
# if the server ignores it. Using 'openai/' prefix to ensure LiteLLM uses OpenAI-compatible endpoint.
LLAMA_CPP_VL_MODEL = "openai/Qwen3VL-4B-Instruct-Q4_K_M.gguf"
LLAMA_CPP_VL_API_KEY = "aaa"
# Default to port 8082 as requested by user
LLAMA_CPP_VL_API_BASE = os.getenv("LLAMA_CPP_VL_API_BASE", "http://localhost:8082")

# ------------------------------------------------------------------------------
# Root Agent
# ------------------------------------------------------------------------------
root_agent = LlmAgent(
    model=LiteLlm(
        model=LLAMA_CPP_VL_MODEL,
        api_key=LLAMA_CPP_VL_API_KEY,
        api_base=LLAMA_CPP_VL_API_BASE,
    ),
    name="pic_read_agent",
    description="A visual language agent capable of reading and understanding images.",
    instruction="""You are an intelligent agent capable of understanding images (Visual Language Model).
    When a user provides an image, describe it or answer questions about its content.
    If no image is provided, simply assist the user as a text-based assistant.
    """,
)
