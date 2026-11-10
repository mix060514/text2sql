import json
import pathlib

import pandas as pd
import streamlit as st

project_root = pathlib.Path(__file__).parent.parent.parent.resolve()
results_dir = project_root / "eval" / "results"
eval_set_path = project_root / "data" / "eval_set_v3.jsonl"

st.set_page_config(page_title="評估結果", page_icon="📊", layout="wide")
st.title("📊 評估結果", anchor="overview")

with st.container(border=True):
    st.markdown(
        "**這一頁是問答系統的考試成績。** 每一題都有事先算好的標準答案，系統從讀題、寫 SQL、"
        "查資料庫到寫出回答與圖表，整條流程完整跑一次，再逐題和標準答案比對。\n\n"
        "- **模板問題**：常見的業務提問（銷售額、排名、趨勢、占比、年增率等），"
        "檢查查出來的數字，以及該不該畫圖、畫哪種圖。\n"
        "- **攔截問題**：不該去查資料的輸入（要求修改資料、閒聊、提示注入、資料庫裡沒有的東西），"
        "檢查系統有沒有拒絕，而不是硬去執行查詢。\n"
        "- **複雜問題**：需要多步驟計算的查詢，每題還有換句話說的改寫版，"
        "檢查換了問法答案是否仍然一樣。\n\n"
        "模板與複雜問題的標準答案不由被測的系統產生，而是用 pandas 直接從原始資料另外計算，兩條路徑互相獨立。"
        "左側可切換題組；顯示的是最新一次完整考試的結果。"
    )

GROUPS = {"模板問題": None, "攔截問題": "block", "複雜問題": "complex"}
group = st.sidebar.radio("題組", list(GROUPS), help=(
    "模板問題：132 題模板考卷（eval_set_v3）。攔截問題：不該查資料的輸入（寫入、閒聊、"
    "提示注入、資料庫外問題），系統不得執行任何 SELECT。複雜問題：34 種 SQL-first 複雜查詢"
    "與其改寫，標準答案由 SQL 與 pandas 雙重計算。"))
BLOCK_KIND = {"write_request": "寫入／修改請求", "chitchat": "閒聊", "prompt_injection": "提示注入",
              "not_in_data": "資料庫外的問題"}
SYNTHETIC_RUNS = project_root / "data" / "synthetic" / "runs"


def show_toc(groups):
    """Sidebar table of contents: overview, the per-type section, then each question type."""
    lines = ["- [總覽與指標](#overview)", "- [依題型的題目與結果](#by-type)"]
    lines += [f"    - [{name}（{ok}/{n}）](#type-{i})" for i, (name, ok, n) in enumerate(groups)]
    st.sidebar.divider()
    st.sidebar.markdown("**目錄**\n\n" + "\n".join(lines))


def type_heading(i, text):
    # Streamlit rewrites heading ids to text slugs, so the anchor is a separate element.
    st.html(f'<div id="type-{i}" style="scroll-margin-top: 4rem"></div>')
    st.markdown(f"#### {text}")


def weakest_first(df):
    """Question types ordered by pass rate (lowest first), ties keep their original order."""
    groups = list(df.groupby("題型", sort=False))
    return sorted(groups, key=lambda g: g[1]["_ok"].mean())


def failed_first(part):
    """Failed questions first, each half in its original order."""
    return part.sort_values("_ok", kind="stable")


def subset_of(record):
    if record.get("subset"):
        return record["subset"]
    if record.get("label") == "block":
        return "block"
    return "complex" if str(record.get("seed_id") or "").startswith("X-") else "template"


def seed_questions():
    """Complex items are grouped by their seed question (3 rewrites each)."""
    out = {}
    for run in sorted(SYNTHETIC_RUNS.glob("*/dataset.jsonl")):
        for line in run.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                if item.get("source") == "seed":
                    out[item["seed_id"]] = item["question"]
    return out


def show_synthetic(subset, title):
    runs = []
    for path in sorted(results_dir.glob("synthetic*/synthetic_results.json"), reverse=True):
        records = [r for r in json.loads(path.read_text(encoding="utf-8")) if subset_of(r) == subset]
        if records:
            runs.append((path.parent.name, path.parent / "synthetic_summary.json", records))
    if not runs:
        st.info(f"還沒有{title}的評估結果。")
        st.stop()
    _, summary_path, records = runs[0]  # newest run
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    meta = summary.get("run_metadata", {})
    if meta:
        model = str(meta.get("model", "")).removeprefix("openai/")
        st.caption(f"模型 `{model}` · 題庫 `{meta.get('dataset')}` · {summary.get('timestamp', '')}")

    df = pd.DataFrame(records)
    df["_ok"] = df["passed"].astype(bool)
    if subset == "block":
        df["題型"] = df["category"].map(lambda c: BLOCK_KIND.get(c, c))
        metric, note = "攔截成功", "攔截成功：系統沒有執行任何 SELECT，也沒有發布查詢結果。"
    else:
        seeds = seed_questions()
        df["題型"] = df["seed_id"].map(lambda s: seeds.get(s, s))
        metric, note = "答案正確", "答案正確：查詢結果與 SQL＋pandas 雙重計算的標準答案逐列相符（程式判定，不用 LLM 評審）。"
    cols = st.columns(3)
    cols[0].metric("題數", len(df))
    cols[1].metric(metric, f"{df['_ok'].mean():.1%}")
    cols[2].metric("原題／改寫", f"{(df['source'] == 'seed').sum()}／{(df['source'] == 'rewrite').sum()}")
    st.caption(note)

    by_type = df.groupby("題型", sort=False)["_ok"].agg(["count", "sum"])
    by_type["通過率"] = (by_type["sum"] / by_type["count"]).map("{:.0%}".format)
    st.dataframe(by_type.rename(columns={"count": "題數", "sum": "通過"}), width="stretch")

    st.divider()
    st.subheader("依題型的題目與結果", anchor="by-type")
    groups = weakest_first(df)
    show_toc([(c, int(part["_ok"].sum()), len(part)) for c, part in groups])
    for i, (category, part) in enumerate(groups):
        type_heading(i, f"{category}（通過 {int(part['_ok'].sum())}/{len(part)}）")
        for _, row in failed_first(part).iterrows():
            tag = "原題" if row.get("source") == "seed" else "改寫"
            with st.expander(f"{'✅' if row['_ok'] else '❌'} [{tag}] {row['question'][:90]}"):
                st.markdown("**系統回答：**")
                st.write(row.get("answer") or "")
                if subset != "block":
                    st.markdown("**SQL：**")
                    st.code(row.get("sql_query") or "", language="sql")
                st.caption(f"判定：{row.get('reason')}")
    st.stop()


if GROUPS[group]:
    show_synthetic(GROUPS[group], group)

# eval/eval_e2e.py writes <run>/e2e_results.json (+ e2e_summary.json); show the newest run.
runs = sorted(results_dir.glob("*/e2e_results.json"), key=lambda p: p.parent.name, reverse=True)
if not runs:
    st.warning("目前沒有評估結果。")
    st.stop()

selected_file = runs[0]
data = json.loads(selected_file.read_text(encoding="utf-8"))
if not data:
    st.error("這個評估批次沒有資料。")
    st.stop()


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
    st.subheader("依題型的題目與結果", anchor="by-type")
    items = load_eval_set() if meta.get("dataset", "eval_set_v3.jsonl") == "eval_set_v3.jsonl" else {}
    df["題型"] = [category_of(items.get(q)) for q in df["question"]]
    df["_ok"] = df["e2e_success"].astype(bool) if "e2e_success" in df else False

    by_type = df.groupby("題型", sort=False)["_ok"].agg(["count", "sum"])
    by_type["整題答對"] = (by_type["sum"] / by_type["count"]).map("{:.0%}".format)
    st.dataframe(by_type.rename(columns={"count": "題數", "sum": "答對"}), width="stretch")

    groups = weakest_first(df)
    show_toc([(c, int(g["_ok"].sum()), len(g)) for c, g in groups])
    for i, (category, group) in enumerate(groups):
        type_heading(i, f"{category}（答對 {int(group['_ok'].sum())}/{len(group)}）")
        for index, row in failed_first(group).iterrows():
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

show_e2e(data, selected_file.parent / "e2e_summary.json")
