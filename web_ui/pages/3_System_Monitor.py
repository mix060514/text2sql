import streamlit as st
import subprocess
import time

st.set_page_config(page_title="System Monitor", page_icon="🖥️", layout="wide")

st.title("🖥️ GPU System Monitor")

st.markdown("Live view of `nvidia-smi`. Toggle the switch below to start monitoring.")

run_monitor = st.toggle("Start Monitoring", value=True)
monitor_placeholder = st.empty()

if run_monitor:
    while True:
        try:
            # Run nvidia-smi
            result = subprocess.run(
                ["nvidia-smi"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            output = result.stdout

            # If command failed (e.g. no nvidia driver), show error
            if result.returncode != 0:
                output = f"Error running nvidia-smi:\n{result.stderr}"
                if not output.strip():
                    output = "Error: nvidia-smi return non-zero exit code. Is NVIDIA driver installed?"

        except FileNotFoundError:
            output = "Error: `nvidia-smi` command not found. Are you on a machine with NVIDIA GPU?"
        except Exception as e:
            output = f"Error: {e}"

        # Update display
        monitor_placeholder.code(output, language="text")

        # Sleep 1s
        time.sleep(1)

        # Check if toggle changed state requires a rerun to break loop cleanly?
        # Streamlit scripts are rerun on interaction.
        # Inside a while loop, we might block interactions unless we yield or use st.empty correctly.
        # Actually in Streamlit, a long running loop blocks other interactions unless we are careful.
        # But for a simple "watch" page, this is usually acceptable as long as the user can kill the tab or stop via UI if implemented with st.experimental_rerun (deprecated) or just reliance on Streamlit's interruption model.
        # Better approach for responsiveness: use st.empty and check value, but the toggle value won't update *inside* the loop unless script restarts.
        # So we need to break periodically or just rely on the fact that changing the toggle triggers a re-run which kills the current script execution.
        # Yes, toggling the widget updates the session state and restarts the script, interrupting the loop.
else:
    monitor_placeholder.info("Monitoring stopped.")
