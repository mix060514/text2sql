import os
import base64
from openai import OpenAI

# Configuration
API_BASE = "http://localhost:8082/v1"
API_KEY = "sk-no-key-required"
MODEL_NAME = "Qwen3VL-4B-Instruct-Q4_K_M.gguf"
# Local image path
IMAGE_PATH = os.path.join(
    os.path.dirname(__file__), "Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
)


def encode_image(image_path):
    print(f"Reading image from {image_path}...")
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def test_vision():
    print(f"Connecting to {API_BASE} with model {MODEL_NAME}...")

    # Encode image before sending
    try:
        base64_image = encode_image(IMAGE_PATH)
        image_data = f"data:image/jpeg;base64,{base64_image}"
    except Exception as e:
        print(f"Failed to read/encode image: {e}")
        return

    client = OpenAI(
        base_url=API_BASE,
        api_key=API_KEY,
    )

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "What is in this image? Describe it in detail.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": image_data},
                        },
                    ],
                }
            ],
            max_tokens=300,
        )
        print("\nResponse:")
        print(response.choices[0].message.content)

    except Exception as e:
        print(f"\nError occurred: {e}")


if __name__ == "__main__":
    test_vision()
