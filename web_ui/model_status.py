"""Is the local model free right now? Read-only llama.cpp router endpoints, no generation.

/v1/models says whether the model is loaded; /slots says whether a request is being
processed. Both return small JSON and do not touch the GPU, so polling every few
seconds is cheap. The router serves one request at a time, so "busy" means a new
question will queue behind someone else's.
"""

import json
from urllib.request import urlopen

LABELS = {
    "idle": ("🟢", "模型空閒，可以直接提問"),
    "busy": ("🟡", "模型忙碌中：有人正在查詢，你的問題會排隊等候"),
    "loading": ("🔵", "模型載入中，請稍候"),
    "not_loaded": ("⚪", "模型尚未載入，第一個問題會先等它載入"),
    "down": ("🔴", "目前連不到模型服務"),
}


def _get(url: str, timeout: float):
    with urlopen(url, timeout=timeout) as response:
        return json.loads(response.read())


def model_status(api_base: str, model: str, timeout: float = 2.0) -> tuple[str, str, str]:
    """Return (state, icon, text); state is one of LABELS."""
    root = api_base.rstrip("/").removesuffix("/v1")
    name = model.split("/", 1)[1] if model.startswith("openai/") else model
    try:
        listed = _get(f"{root}/v1/models", timeout).get("data", [])
        entry = next((m for m in listed if m.get("id") == name), None)
        loaded = (entry or {}).get("status", {}).get("value", "loaded")
        if loaded == "loading":
            state = "loading"
        elif loaded != "loaded":
            state = "not_loaded"
        else:
            slots = _get(f"{root}/slots?model={name}", timeout)
            state = "busy" if any(s.get("is_processing") for s in slots) else "idle"
    except (OSError, ValueError, AttributeError, TypeError):
        state = "down"
    return (state, *LABELS[state])
