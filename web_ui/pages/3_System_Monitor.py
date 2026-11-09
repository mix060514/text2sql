"""GPU summary for the machine that runs the model.

Only aggregate numbers are shown; the raw nvidia-smi table would expose the host's
process list on a public page. A fragment reruns every few seconds instead of a
blocking loop, so each open tab does not hold a server thread forever.
"""
import subprocess

import streamlit as st

st.set_page_config(page_title="系統監控", page_icon="🖥️", layout="wide")
st.title("🖥️ 系統監控")
st.markdown("問答系統的模型在這張 GPU 上執行。提問時可以在這裡看到使用率與顯示記憶體上升。")

FIELDS = "name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw"
REFRESH_SECONDS = 2


def read_gpus():
    result = subprocess.run(
        ["nvidia-smi", f"--query-gpu={FIELDS}", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=5,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip())
    return [[v.strip() for v in line.split(",")] for line in result.stdout.strip().splitlines()]


def number(text):
    try:
        return float(text)
    except ValueError:
        return None


@st.fragment(run_every=REFRESH_SECONDS)
def gpu_panel():
    try:
        gpus = read_gpus()
    except Exception:
        st.warning("目前無法讀取 GPU 狀態。")
        return
    for name, util, used, total, temp, power in gpus:
        st.subheader(name)
        used_mb, total_mb = number(used), number(total)
        cols = st.columns(4)
        cols[0].metric("GPU 使用率", f"{util}%")
        cols[1].metric(f"顯示記憶體（共 {total_mb / 1024:.0f} GB）" if total_mb else "顯示記憶體",
                       f"{used_mb / 1024:.1f} GB" if used_mb is not None else "–")
        cols[2].metric("溫度", f"{temp} °C")
        cols[3].metric("功耗", f"{number(power):.0f} W" if number(power) is not None else "–")
        if used_mb is not None and total_mb:
            st.progress(min(used_mb / total_mb, 1.0), text="顯示記憶體使用量")
    st.caption(f"每 {REFRESH_SECONDS} 秒自動更新。")


gpu_panel()
