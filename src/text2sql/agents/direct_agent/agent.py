from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm

root_agent = LlmAgent(
    model=LiteLlm(
        model="ollama_chat/qwen2.5:7b",
    ),
    name="root_agent",
    description="A helpful assistant for user questions.",
    instruction="Answer user questions to the best of your knowledge",
)
