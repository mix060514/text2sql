"""評估結果: the newest run of the exam (data/exam), graded by code against the database."""

import json
import pathlib

import pandas as pd
import streamlit as st

project_root = pathlib.Path(__file__).parent.parent.parent.resolve()
results_dir = project_root / "eval" / "results"
exam_path = project_root / "data" / "exam" / "dataset.jsonl"

st.set_page_config(page_title="評估結果", page_icon="📊", layout="wide")
st.title("📊 評估結果", anchor="overview")

GROUPS = {
    "基本查詢": "一條查詢就能回答：加總、訂單數、平均單價、取第一名或前幾名、年增率。",
    "畫圖": "除了數字，還要畫對圖：長條圖、折線圖、圓餅圖，檢查圖表類型與橫軸順序。",
    "多步驟查詢": "要分幾步才算得出來：每組各取第一名、累計、占比、跨年比較、客戶是否回購。",
    "不同叫法": "問題裡的地區或國家名稱和資料庫存的不一樣（歐非中東、大陸、US…）。",
    "該拒答": "不該去查資料的輸入：要求改資料、資料庫沒有的資料、閒聊、提示注入。",
}
CHART = {"bar": "長條圖", "line": "折線圖", "pie": "圓餅圖"}

with st.container(border=True):
    st.markdown(
        "**這一頁是問答系統的考試成績。** 考卷的每一題都考不同的東西，每題只出現一次；"
        "約一半的題目用口語問法。系統從讀題、寫 SQL、查資料庫到寫出回答與圖表完整跑一次，"
        "再由程式逐題和標準答案比對（不用 LLM 評審）。\n\n"
        + "\n".join(f"- **{name}**：{text}" for name, text in GROUPS.items())
        + "\n\n標準答案是每題附的 SQL 直接在資料庫上執行的結果，不由被測的系統產生，也不由人填寫。"
    )


def load_items():
    if not exam_path.exists():
        return {}
    rows = (json.loads(line) for line in exam_path.read_text(encoding="utf-8").splitlines() if line.strip())
    return {r["id"]: r for r in rows}


def chart_name(plot):
    plot = plot or {}
    return CHART.get(plot.get("chart_type"), plot.get("chart_type")) if plot.get("should_plot") else "不畫圖"


def show_table(columns, rows):
    rows = rows if isinstance(rows, list) else []
    columns = columns if isinstance(columns, list) and rows and len(columns) == len(rows[0]) else None
    if rows:
        st.dataframe(pd.DataFrame(rows, columns=columns), hide_index=True)
    else:
        st.caption("（空結果）")


def anchor(i, text):
    # Streamlit rewrites heading ids to text slugs, so the anchor is a separate element.
    st.html(f'<div id="group-{i}" style="scroll-margin-top: 4rem"></div>')
    st.markdown(f"#### {text}")


items = load_items()
if not items:
    st.info("還沒有考卷（data/exam）。")
    st.stop()

runs = sorted(results_dir.glob("exam_*/synthetic_results.json"), reverse=True)
if not runs:
    st.info("考卷還沒有評估結果；以下是考卷的題目。")
    st.dataframe(pd.DataFrame([{"題組": i["group"], "考什麼": i["point"], "題目": i["question"]}
                               for i in items.values()]), hide_index=True, width="stretch")
    st.stop()

results_path = runs[0]  # newest run
records = [r for r in json.loads(results_path.read_text(encoding="utf-8")) if r.get("id") in items]
summary_path = results_path.parent / "synthetic_summary.json"
summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
meta = summary.get("run_metadata", {})
if meta:
    st.caption(f"模型 `{str(meta.get('model', '')).removeprefix('openai/')}` · {summary.get('timestamp', '')}")

df = pd.DataFrame(records)
df["_ok"] = df["passed"].astype(bool)
df["題組"] = [items[i]["group"] for i in df["id"]]
allow = df[df["label"] == "allow"]
block = df[df["label"] == "block"]

cols = st.columns(4)
cols[0].metric("題數", f"{len(df)}／{len(items)}" if len(df) < len(items) else len(df))
cols[1].metric("通過", f"{int(df['_ok'].sum())}/{len(df)}")
if len(allow):
    cols[2].metric("該回答的題：答案正確", f"{int(allow['_ok'].sum())}/{len(allow)}")
if len(block):
    cols[3].metric("該拒答的題：成功拒答", f"{int(block['_ok'].sum())}/{len(block)}")
st.caption("答案正確：查詢結果與標準答案逐列相符；要畫圖的題目，畫不畫圖、圖表類型、橫軸順序也要相符。"
           "成功拒答：系統沒有執行任何查詢，也沒有發布查詢結果。")
if len(df) < len(items):
    st.warning(f"這次評估只完成 {len(df)} 題，考卷共 {len(items)} 題。")

order = [g for g in GROUPS if (df["題組"] == g).any()]
table = df.groupby("題組")["_ok"].agg(["count", "sum"]).loc[order]
table["通過率"] = (table["sum"] / table["count"]).map("{:.0%}".format)
table["考什麼"] = [GROUPS[g] for g in table.index]
st.dataframe(table.rename(columns={"count": "題數", "sum": "通過"}), width="stretch")

st.sidebar.markdown("**目錄**\n\n- [總覽](#overview)\n" + "\n".join(
    f"- [{g}（{int(table.loc[g, 'sum'])}/{int(table.loc[g, 'count'])}）](#group-{i})" for i, g in enumerate(order)))

st.divider()
st.subheader("每一題的結果", anchor="questions")
for i, group in enumerate(order):
    part = df[df["題組"] == group].sort_values("_ok", kind="stable")  # failed questions first
    anchor(i, f"{group}（通過 {int(part['_ok'].sum())}/{len(part)}）")
    for _, row in part.iterrows():
        item = items[row["id"]]
        with st.expander(f"{'✅' if row['_ok'] else '❌'} {item['point']}｜{row['question'][:90]}"):
            st.caption(f"問法：{item.get('wording', '')}　判定：{row.get('reason')}")
            if item["label"] == "block":
                st.markdown("**系統回答：**")
                st.write(row.get("answer") or "")
                continue
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**標準答案：**")
                show_table(item["ground_truth"].get("columns"), item["ground_truth"].get("rows"))
            with col2:
                st.markdown(f"**系統查詢結果**（{row.get('row_count', 0)} 列）：")
                show_table(row.get("column_names"), row.get("rows"))
            if (item.get("expected_plot") or {}).get("should_plot"):
                actual = {"should_plot": row.get("should_plot", bool(row.get("chart_type"))),
                          "chart_type": row.get("chart_type")}
                st.markdown(f"**圖表：** 預期 `{chart_name(item['expected_plot'])}`，實際 `{chart_name(actual)}`")
            st.markdown("**標準答案的 SQL：**")
            st.code(item.get("sql") or "", language="sql")
            st.markdown("**系統寫的 SQL：**")
            st.code(row.get("sql_query") or "", language="sql")
            st.markdown("**系統回答：**")
            st.write(row.get("answer") or "")
