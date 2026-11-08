import json
import pathlib

import pandas as pd
import streamlit as st

project_root = pathlib.Path(__file__).parent.parent.parent.resolve()
results_dir = project_root / "eval" / "results"
eval_set_path = project_root / "data" / "eval_set_v3.jsonl"

st.set_page_config(page_title="評估結果", page_icon="📊", layout="wide")
st.title("📊 評估結果")

# eval/eval_e2e.py writes <run>/e2e_results.json (+ e2e_summary.json); the older eval/eval.py
# wrote <run>/eval_results.json. List current e2e runs first, newest first, then legacy ones.
E2E, LEGACY = "e2e", "legacy"
runs = sorted(
    [(p, E2E) for p in results_dir.glob("*/e2e_results.json")],
    key=lambda r: r[0].parent.name, reverse=True,
)
if st.sidebar.checkbox("顯示舊版評估", help="舊資料庫與舊題庫的結果，數字不能與目前的評估直接比較。"):
    runs += sorted(
        [(p, LEGACY) for p in results_dir.glob("*/eval_results.json")],
        key=lambda r: r[0].parent.name, reverse=True,
    )

if not runs:
    st.warning("目前沒有評估結果。")
    st.stop()

labels = {(p.parent.name if k == E2E else f"[舊版] {p.parent.name}"): (p, k) for p, k in runs}
selected_file, kind = labels[st.sidebar.selectbox("選擇評估批次", list(labels))]

data = json.loads(selected_file.read_text(encoding="utf-8"))
if not data:
    st.error("這個評估批次沒有資料。")
    st.stop()


def show_legacy(data):
    st.caption("舊版評估（舊資料庫與舊題庫），不能與目前的評估直接比較。")
    df = pd.DataFrame(data)
    st.metric(label="題數", value=len(df))
    st.metric(label="LLM 評審平均分數", value=f"{df['llm_judge_score'].mean():.2f} / 10")
    st.divider()
    st.subheader("逐題結果")
    for index, row in df.iterrows():
        with st.expander(f"Q{index+1}: {row['question'][:80]}...（評審 {row['llm_judge_score']}）"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**標準答案：**")
                st.code(json.dumps(row["expected"], ensure_ascii=False, indent=2), language="json")
                st.markdown("**系統回答：**")
                st.write(row["final_answer"])
                st.markdown("**SQL：**")
                st.code(row["sql_query"], language="sql")
            with col2:
                st.markdown("**評審理由：**")
                st.info(row["llm_judge_reasoning"])
                st.markdown("**查詢結果：**")
                st.code(str(row["execute_result"]), language="json")


def load_eval_set():
    if not eval_set_path.exists():
        return {}
    rows = (json.loads(line) for line in eval_set_path.read_text(encoding="utf-8").splitlines() if line.strip())
    return {r["question"]: r for r in rows}


# First tag of each eval_set_v3 item is its question type (see scripts/generate_ground_truth.py).
CATEGORY = {
    "revenue": "銷售額", "top_products": "熱銷產品", "trend": "趨勢", "top_customer": "重要客戶",
    "advanced": "進階分析", "yoy": "年增率", "share": "占比", "quarterly": "季度",
    "comparison": "比較", "simple": "單一數值", "regional_preference": "地區偏好",
    "top": "最高排名", "asp": "平均單價", "orders": "訂單數", "quantity": "銷售量",
}


def category_of(item):
    tags = (item or {}).get("tags") or []
    return CATEGORY.get(tags[0], tags[0]) if tags else "其他"


DIFFICULTY = {"simple": "簡單", "medium": "中等", "hard": "困難"}
CHART = {"bar": "長條圖", "line": "折線圖", "pie": "圓餅圖", "table": "表格"}
chart_name = lambda should, kind: CHART.get(kind, kind) if should else "不畫圖"


def show_e2e(data, summary_path):
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    meta = summary.get("run_metadata", {})
    if meta:
        model = str(meta.get("model", ""))
        model = model.split("/", 1)[1] if model.startswith("openai/") else model
        st.caption(f"模型 `{model}` · 流程 `{meta.get('workflow')}` · "
                   f"題庫 `{meta.get('dataset')}` · {summary.get('timestamp', '')}")

    df = pd.DataFrame(data)
    rate = lambda col: f"{df[col].mean():.1%}" if col in df else "–"
    cols = st.columns(4)
    cols[0].metric("題數", len(df))
    cols[1].metric("整題答對", rate("e2e_success"))
    cols[2].metric("SQL 正確（LLM 評審）", rate("sql_correct"))
    plotted = df[df["has_expected_plot"]] if "has_expected_plot" in df else df.iloc[0:0]
    cols[3].metric("畫圖判斷正確", f"{plotted['plot_decision_correct'].mean():.1%}" if len(plotted) else "–")

    # Two shapes: the eval's own per-question deadline (sql_query == "TIMEOUT") and a model
    # request timeout surfaced as workflow_error {'code': 'Timeout', ...} in sql_reasoning.
    timed_out = pd.Series(False, index=df.index)
    if "sql_query" in df:
        timed_out |= df["sql_query"].eq("TIMEOUT")
    if "sql_reasoning" in df:
        timed_out |= df["sql_reasoning"].astype(str).str.contains("'code': 'Timeout'", regex=False)
    timeouts = int(timed_out.sum())
    with st.expander("指標怎麼算", expanded=True):
        st.markdown(f"""
- **題庫**：132 題模板考卷，題目由 17 個題型填入地區、產品等產生，標準答案用 pandas 從原始資料計算，
  與問答系統寫的 SQL 是兩條獨立路徑。其中 54 題只是同一個地區換不同叫法（例如 EMEA／歐非中東）。
- **SQL 正確**：由另一個 LLM 評審比對查詢結果與標準答案，給 0–10 分。LLM 評審可能誤判，
  失敗題請展開對照「標準答案」與「查詢結果」再下結論。
- **畫圖判斷正確**：只計算有標註預期圖表的題目，比對該不該畫圖以及圖表類型。
- **整題答對**：SQL 正確，且（有預期圖表時）畫圖判斷、圖表類型、X 軸順序與資料序列數都正確。
- 每題有執行時間上限，逾時算答錯；這個批次有 {timeouts} 題逾時（含模型請求逾時）。
""")

    if {"difficulty", "e2e_success"} <= set(df.columns):
        by_level = (df.groupby("difficulty")["e2e_success"].agg(["count", "mean"])
                      .reindex([k for k in DIFFICULTY if k in set(df["difficulty"])]))
        by_level.index = [DIFFICULTY[k] for k in by_level.index]
        by_level.index.name = "難度"
        by_level["mean"] = by_level["mean"].map("{:.1%}".format)
        st.dataframe(by_level.rename(columns={"count": "題數", "mean": "整題答對"}))

    st.divider()
    st.subheader("依題型的題目與結果")
    items = load_eval_set() if meta.get("dataset", "eval_set_v3.jsonl") == "eval_set_v3.jsonl" else {}
    df["題型"] = [category_of(items.get(q)) for q in df["question"]]
    df["_ok"] = df["e2e_success"].astype(bool) if "e2e_success" in df else False

    by_type = df.groupby("題型", sort=False)["_ok"].agg(["count", "sum"])
    by_type["整題答對"] = (by_type["sum"] / by_type["count"]).map("{:.0%}".format)
    st.dataframe(by_type.rename(columns={"count": "題數", "sum": "答對"}), width="stretch")

    view = st.radio("顯示", ["隱藏答錯的題目", "全部題目", "只看答錯的題目"], horizontal=True,
                    help="只影響下方逐題清單；上方指標一律以全部題目計算。")
    for category, group in df.groupby("題型", sort=False):
        shown = group[group["_ok"]] if view == "隱藏答錯的題目" else (
            group[~group["_ok"]] if view == "只看答錯的題目" else group)
        if shown.empty:
            continue
        st.markdown(f"#### {category}（答對 {int(group['_ok'].sum())}/{len(group)}）")
        for index, row in shown.iterrows():
            show_row(index, row, items)


def show_row(index, row, items):
    ok = bool(row["_ok"])
    truth = items.get(row["question"], {})
    with st.expander(f"{'✅' if ok else '❌'} Q{index+1}: {row['question'][:80]}（評審 {row.get('sql_score')}/10）"):
        col1, col2 = st.columns(2)
        with col1:
            if "ground_truth" in truth:
                st.markdown("**標準答案：**")
                st.code(json.dumps(truth["ground_truth"], ensure_ascii=False, indent=2), language="json")
            st.markdown("**系統回答：**")
            st.write(row.get("answer"))
            st.markdown("**SQL：**")
            st.code(row.get("sql_query") or "", language="sql")
        with col2:
            st.markdown("**評審理由：**")
            st.info(row.get("sql_reasoning") or "")
            if row.get("has_expected_plot"):
                st.markdown(
                    f"**圖表：** 預期 `{chart_name(row.get('expected_should_plot'), row.get('expected_chart_type'))}`，"
                    f"實際 `{chart_name(row.get('actual_should_plot'), row.get('actual_chart_type'))}`"
                )
            st.markdown("**查詢結果：**")
            st.code(json.dumps(row.get("sql_result"), ensure_ascii=False, default=str)[:3000], language="json")

if kind == E2E:
    show_e2e(data, selected_file.parent / "e2e_summary.json")
else:
    show_legacy(data)
