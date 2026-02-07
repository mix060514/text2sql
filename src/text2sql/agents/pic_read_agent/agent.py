import os
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from text2sql.config import (
    LLAMA_CPP_VL_MODEL,
    LLAMA_CPP_VL_API_KEY,
    LLAMA_CPP_VL_API_BASE,
)

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
