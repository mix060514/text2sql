import streamlit as st
import sys
import os
import pathlib
import json
import asyncio
import uuid
import time
import base64
from typing import List, Dict

# Setup path to import local src
project_root = pathlib.Path(__file__).parent.parent.parent.resolve()
sys.path.append(str(project_root / "src"))
sys.path.append(str(project_root / "eval"))

try:
    from google.adk.runners import InMemoryRunner
    from google.genai import types
    from text2sql.agents.pic_read_agent.agent import root_agent
except ImportError as e:
    st.error(f"Failed to import agent or dependencies: {e}")
    st.stop()

st.set_page_config(page_title="Image Read Chat", page_icon="🖼️", layout="wide")

# Initialize Session State
if "pic_messages" not in st.session_state:
    st.session_state.pic_messages = []

if "pic_session_id" not in st.session_state:
    st.session_state.pic_session_id = (
        f"web_pic_{int(time.time()*1000)}_{uuid.uuid4().hex[:6]}"
    )

if "pic_runner" not in st.session_state:
    st.session_state.pic_runner = InMemoryRunner(
        agent=root_agent, app_name="pic_read_chat"
    )

    # Initialize session in ADK
    async def init_session():
        await st.session_state.pic_runner.session_service.create_session(
            app_name=st.session_state.pic_runner.app_name,
            user_id="web_user",
            session_id=st.session_state.pic_session_id,
        )

    asyncio.run(init_session())

# Sidebar settings
st.sidebar.title("Image Upload")
uploaded_file = st.sidebar.file_uploader(
    "Choose an image...", type=["jpg", "jpeg", "png"]
)

current_image_base64 = None
current_image_mime = None

if uploaded_file is not None:
    # Display the uploaded image
    st.sidebar.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    # Process image for agent
    from PIL import Image
    import io

    image = Image.open(uploaded_file)

    # Qwen3-VL uses patch_size 16 and usually merge_size 2 => alignment 32
    # Ensure dimensions are multiples of 32 and max dimension is reasonable
    target_max = 1024  # Safe upper bound

    width, height = image.size
    scale = min(target_max / width, target_max / height, 1.0)  # Don't upscale

    new_width = int(width * scale)
    new_height = int(height * scale)

    # Align to multiple of 32
    def align_32(x):
        return (x // 32) * 32

    new_width = max(32, align_32(new_width))
    new_height = max(32, align_32(new_height))

    image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # Convert back to bytes for base64
    buffered = io.BytesIO()
    if image.mode in ("RGBA", "P"):
        image = image.convert("RGB")

    image.save(buffered, format="JPEG", quality=85)
    bytes_data = buffered.getvalue()
    current_image_mime = "image/jpeg"
    current_image_base64 = base64.b64encode(bytes_data).decode("utf-8")

# Display Messages
for msg in st.session_state.pic_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "image_base64" in msg and msg["image_base64"]:
            st.image(base64.b64decode(msg["image_base64"]), width=300)

# Chat Input
if prompt := st.chat_input("Ask about the image..."):

    # User message
    msg_content = prompt
    user_msg_data = {
        "role": "user",
        "content": msg_content,
    }

    parts = [types.Part(text=prompt)]

    if current_image_base64:
        user_msg_data["image_base64"] = current_image_base64

        try:
            # Construct Part with inline data (standard Gemini/ADK pattern)
            image_blob = types.Blob(
                mime_type=current_image_mime,
                data=base64.b64decode(current_image_base64),
            )
            parts.append(types.Part(inline_data=image_blob))
        except Exception as e:
            st.error(f"Error preparing image for agent: {e}")

    st.session_state.pic_messages.append(user_msg_data)
    with st.chat_message("user"):
        st.markdown(prompt)
        if "image_base64" in user_msg_data:
            st.image(base64.b64decode(user_msg_data["image_base64"]), width=300)

    # Process Agent Response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        async def run_chat():
            collected_text = []
            content = types.Content(role="user", parts=parts)

            async for event in st.session_state.pic_runner.run_async(
                user_id="web_user",
                session_id=st.session_state.pic_session_id,
                new_message=content,
            ):
                c = getattr(event, "content", None)
                if c and c.parts:
                    for p in c.parts:
                        if p.text:
                            collected_text.append(p.text)
                            # Optional: Stream the update to UI
                            # message_placeholder.markdown("".join(collected_text) + "▌")

            return "".join(collected_text)

        full_response = asyncio.run(run_chat())

        if not full_response:
            full_response = "Done (No text response captured, check logs if error)."

        message_placeholder.markdown(full_response)
        st.session_state.pic_messages.append(
            {
                "role": "assistant",
                "content": full_response,
            }
        )
