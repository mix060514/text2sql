import streamlit as st
import pandas as pd
import json
import pathlib
import os
from glob import glob

# Setup path
project_root = pathlib.Path(__file__).parent.parent.parent.resolve()
results_dir = project_root / "eval" / "results"

st.set_page_config(page_title="Evaluation Results", page_icon="📊", layout="wide")

st.title("📊 Evaluation Results Viewer")

# Find available results
# Structure: eval/results/YYYY-MM-DD_HH-MM-SS/eval_results.json
result_files = sorted(
    glob(str(results_dir / "**" / "eval_results.json"), recursive=True), reverse=True
)


def get_run_label(path):
    # Extract timestamp folder name
    return pathlib.Path(path).parent.name


if not result_files:
    st.warning("No evaluation results found in `eval/results/`.")
    st.stop()

selected_file = st.sidebar.selectbox(
    "Select Evaluation Run", result_files, format_func=get_run_label
)

if selected_file:
    with open(selected_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not data:
        st.error("Selected file is empty.")
        st.stop()

    df = pd.DataFrame(data)

    # Calculate Metrics
    total = len(df)
    # Judge score is usually 0-10. passing score might be implicit, checking > 7?
    # Or rely on 'llm_judge_score' directly.
    avg_score = df["llm_judge_score"].mean()

    # Try to determine "Success" if not explicitly in JSON, but eval.py logs it.
    # eval.py saves 'llm_judge_score'. Let's assume score >= 7 is 'Pass' for visualization if not specified.
    # Actually eval_row has fields. Let's just show Average Score.

    st.metric(label="Total Questions", value=total)
    st.metric(label="Average Judge Score", value=f"{avg_score:.2f} / 10")

    st.divider()

    st.subheader("Detailed Results")

    for index, row in df.iterrows():
        with st.expander(
            f"Q{index+1}: {row['question'][:80]}... (Score: {row['llm_judge_score']})"
        ):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Expected Result:**")
                st.code(
                    json.dumps(row["expected"], ensure_ascii=False, indent=2),
                    language="json",
                )

                st.markdown("**Agent Answer:**")
                st.write(row["final_answer"])

                st.markdown("**SQL Query:**")
                st.code(row["sql_query"], language="sql")

            with col2:
                st.markdown("**Judge Reasoning:**")
                st.info(row["llm_judge_reasoning"])

                st.markdown("**Execution Result:**")
                st.code(str(row["execute_result"]), language="json")
