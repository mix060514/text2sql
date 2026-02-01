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
    bytes_data = uploaded_file.getvalue()
    current_image_mime = uploaded_file.type
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

    # Check if we should include image (only if specific to this turn or persistent?)
    # For now, let's include the image IF it's the first time or if user re-uploads.
    # Actually, simplistic approach: pass image if uploaded and currently selected.

    parts = [types.Part(text=prompt)]

    if current_image_base64:
        # Check if we already sent this image?
        # For simple VLM interaction, usually sending it every time or maintaining context is model dependent.
        # ADK/LiteLLM with caching might handle it, but safer to send it if user currently sees it.
        # But to avoid token waste, typically we send it once.
        # For this demo, let's attach it to the current message if it's there.

        # Attach image to UI message
        user_msg_data["image_base64"] = current_image_base64

        try:
            # Construct Part with inline data (standard Gemini/ADK pattern)
            image_blob = types.Blob(
                mime_type=current_image_mime,
                data=base64.b64decode(current_image_base64),
            )
            parts.append(types.Part(inline_data=image_blob))
        except Exception as e:
            # Fallback for LiteLLM if it expects specific format like image_url with data URI
            # But ADK types usually abstraction this. If standard Blob fails, we might need a different approach.
            # Let's try the Blob approach first as it's most likely for Google ADK.
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
            response_text = ""
            content = types.Content(role="user", parts=parts)

            async for event in st.session_state.pic_runner.run_async(
                user_id="web_user",
                session_id=st.session_state.pic_session_id,
                new_message=content,
            ):
                pass  # We just wait for completion for now, or could stream text part

            # Get final answer
            session = await st.session_state.pic_runner.session_service.get_session(
                app_name="pic_read_chat",
                user_id="web_user",
                session_id=st.session_state.pic_session_id,
            )

            # Since pic_read_agent is a simple LlmAgent, it puts result in last turn model response
            # Or formatted answer.
            # LlmAgent usually returns the model response text.

            # Let's check session events or state.
            # For simple LlmAgent, the runner usually yields the model response chunks.
            # But here we didn't capture them in loop.

            # Let's retrieve the last message from history
            if session and session.history:
                last_msg = session.history[-1]
                if last_msg.role == "model":
                    for part in last_msg.parts:
                        if part.text:
                            response_text += part.text

            return response_text

        full_response = asyncio.run(run_chat())

        if not full_response:
            full_response = "No response from model (check logs/connection)."

        message_placeholder.markdown(full_response)
        st.session_state.pic_messages.append(
            {
                "role": "assistant",
                "content": full_response,
            }
        )
