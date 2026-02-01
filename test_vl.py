import asyncio
import os
import sys
import base64

# Ensure src is in path
sys.path.append("/app/src")

from text2sql.agents.pic_read_agent.agent import root_agent
from google.adk.runners import InMemoryRunner
from google.genai import types

# 1x1 Red Pixel PNG
SAMPLE_IMAGE_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="


async def main():
    print("Testing Pic Read Agent with Image...")
    print(f"LLAMA_CPP_VL_API_BASE: {os.getenv('LLAMA_CPP_VL_API_BASE')}")

    runner = InMemoryRunner(agent=root_agent, app_name="test_app_img")
    session_id = "test_session_img"
    await runner.session_service.create_session(
        app_name="test_app_img", user_id="test_user", session_id=session_id
    )

    # Create message with image
    image_blob = types.Blob(
        mime_type="image/png", data=base64.b64decode(SAMPLE_IMAGE_B64)
    )

    content = types.Content(
        role="user",
        parts=[
            types.Part(text="What color is this image?"),
            types.Part(inline_data=image_blob),
        ],
    )

    try:
        print("Sending request with image...")
        async for event in runner.run_async(
            user_id="test_user", session_id=session_id, new_message=content
        ):
            print(f"Event: {event}")

    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
