import json
import pathlib

import pandas as pd
import streamlit as st

project_root = pathlib.Path(__file__).parent.parent.parent.resolve()
results_dir = project_root / "eval" / "results"
eval_set_path = project_root / "data" / "eval_set_v3.jsonl"

st.set_page_config(page_title="Evaluation Results", page_icon="📊", layout="wide")
st.title("📊 Evaluation Results Viewer")

# eval/eval_e2e.py writes <run>/e2e_results.json (+ e2e_summary.json); the older eval/eval.py
# wrote <run>/eval_results.json. List current e2e runs first, newest first, then legacy ones.
E2E, LEGACY = "e2e", "legacy"
runs = sorted(
    [(p, E2E) for p in results_dir.glob("*/e2e_results.json")],
    key=lambda r: r[0].parent.name, reverse=True,
) + sorted(
    [(p, LEGACY) for p in results_dir.glob("*/eval_results.json")],
    key=lambda r: r[0].parent.name, reverse=True,
)

if not runs:
    st.warning("No evaluation results found in `eval/results/`.")
    st.stop()

labels = {(p.parent.name if k == E2E else f"[legacy] {p.parent.name}"): (p, k) for p, k in runs}
selected_file, kind = labels[st.sidebar.selectbox("Select Evaluation Run", list(labels))]

data = json.loads(selected_file.read_text(encoding="utf-8"))
if not data:
    st.error("Selected file is empty.")
    st.stop()


def show_legacy(data):
    st.caption("Legacy eval.py run (older database / eval set); not comparable with e2e runs.")
    df = pd.DataFrame(data)
    st.metric(label="Total Questions", value=len(df))
    st.metric(label="Average Judge Score", value=f"{df['llm_judge_score'].mean():.2f} / 10")
    st.divider()
    st.subheader("Detailed Results")
    for index, row in df.iterrows():
        with st.expander(f"Q{index+1}: {row['question'][:80]}... (Score: {row['llm_judge_score']})"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Expected Result:**")
                st.code(json.dumps(row["expected"], ensure_ascii=False, indent=2), language="json")
                st.markdown("**Agent Answer:**")
                st.write(row["final_answer"])
                st.markdown("**SQL Query:**")
                st.code(row["sql_query"], language="sql")
            with col2:
                st.markdown("**Judge Reasoning:**")
                st.info(row["llm_judge_reasoning"])
                st.markdown("**Execution Result:**")
                st.code(str(row["execute_result"]), language="json")


def load_ground_truth():
    if not eval_set_path.exists():
        return {}
    rows = (json.loads(line) for line in eval_set_path.read_text(encoding="utf-8").splitlines() if line.strip())
    return {r["question"]: r.get("ground_truth") for r in rows}


def show_e2e(data, summary_path):
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    meta = summary.get("run_metadata", {})
    if meta:
        st.caption(f"model `{meta.get('model')}` · workflow `{meta.get('workflow')}` · "
                   f"dataset `{meta.get('dataset')}` · {summary.get('timestamp', '')}")

    df = pd.DataFrame(data)
    rate = lambda col: f"{df[col].mean():.1%}" if col in df else "–"
    cols = st.columns(4)
    cols[0].metric("Total Questions", len(df))
    cols[1].metric("E2E Success", rate("e2e_success"))
    cols[2].metric("SQL Correct (judge)", rate("sql_correct"))
    plotted = df[df["has_expected_plot"]] if "has_expected_plot" in df else df.iloc[0:0]
    cols[3].metric("Plot Decision Correct",
                   f"{plotted['plot_decision_correct'].mean():.1%}" if len(plotted) else "–")
    st.caption("SQL correctness comes from an LLM judge, which can mis-score; "
               "check failures against the expected result before trusting the rate.")

    if {"difficulty", "e2e_success"} <= set(df.columns):
        st.dataframe(
            df.groupby("difficulty")["e2e_success"].agg(["count", "mean"])
              .rename(columns={"count": "questions", "mean": "e2e success"}),
        )

    st.divider()
    st.subheader("Detailed Results")
    only_failed = st.checkbox("Show failures only")
    truth = load_ground_truth() if meta.get("dataset", "eval_set_v3.jsonl") == "eval_set_v3.jsonl" else {}

    for index, row in df.iterrows():
        ok = bool(row.get("e2e_success"))
        if only_failed and ok:
            continue
        mark = "✅" if ok else "❌"
        with st.expander(f"{mark} Q{index+1}: {row['question'][:80]} (judge {row.get('sql_score')}/10)"):
            col1, col2 = st.columns(2)
            with col1:
                if row["question"] in truth:
                    st.markdown("**Expected Result:**")
                    st.code(json.dumps(truth[row["question"]], ensure_ascii=False, indent=2), language="json")
                st.markdown("**Agent Answer:**")
                st.write(row.get("answer"))
                st.markdown("**SQL Query:**")
                st.code(row.get("sql_query") or "", language="sql")
            with col2:
                st.markdown("**Judge Reasoning:**")
                st.info(row.get("sql_reasoning") or "")
                if row.get("has_expected_plot"):
                    st.markdown(
                        f"**Plot:** expected `{row.get('expected_chart_type') if row.get('expected_should_plot') else 'no plot'}`, "
                        f"actual `{row.get('actual_chart_type') if row.get('actual_should_plot') else 'no plot'}`"
                    )
                st.markdown("**SQL Result:**")
                st.code(json.dumps(row.get("sql_result"), ensure_ascii=False, default=str)[:3000], language="json")


if kind == E2E:
    show_e2e(data, selected_file.parent / "e2e_summary.json")
else:
    show_legacy(data)
