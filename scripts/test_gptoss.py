import os
from litellm import completion

# Set verbose to see what's happening
os.environ["LITELLM_LOG"] = "INFO"

print("Attempting to connect to Ollama via LiteLLM with gpt-oss:20b...")
try:
    response = completion(
        model="openai/gpt-oss:20b",
        messages=[{"role": "user", "content": "Hello, what's your name?"}],
        api_key="aaa",
        api_base="http://localhost:8082",
    )
    print("Response received:")
    print(response.choices[0].message.content)
    print("SUCCESS")
except Exception as e:
    print(f"Error occurred: {e}")
