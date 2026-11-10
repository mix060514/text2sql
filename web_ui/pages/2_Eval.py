import json
import pathlib

import pandas as pd
import streamlit as st

project_root = pathlib.Path(__file__).parent.parent.parent.resolve()
results_dir = project_root / "eval" / "results"
data_dir = project_root / "data"

st.set_page_config(page_title="評估結果", page_icon="📊", layout="wide")
st.title("📊 評估結果", anchor="overview")

with st.container(border=True):
    st.markdown(
        "**這一頁是問答系統的考試成績。** 每一題都有事先算好的標準答案，系統從讀題、寫 SQL、"
        "查資料庫到寫出回答與圖表，整條流程完整跑一次，再由程式逐題和標準答案比對（不用 LLM 評審）。\n\n"
        "- **模板問題**：16 題常見業務提問，每題測一種能力（篩選加總、訂單數、平均單價、Top-N、"
        "年增率、同季比較、組內取最大、長條／折線／圓餅圖），檢查查出來的數字與圖表。\n"
        "- **說法對齊**：地區與國家的各種叫法（北美、歐非中東、US、大陸…），每種叫法一題，"
        "句型固定，答錯就代表那個叫法沒有對上資料庫裡的值。\n"
        "- **攔截問題**：不該去查資料的輸入（要求修改資料、閒聊、提示注入、資料庫裡沒有的東西），"
        "檢查系統有沒有拒絕，而不是硬去執行查詢。\n"
        "- **複雜問題**：需要多步驟計算的查詢，每題還有換句話說的改寫版，"
        "檢查換了問法答案是否仍然一樣。\n\n"
        "標準答案由題庫裡的 SQL 直接在資料庫上計算，不由被測的系統產生。"
        "左側可切換題組；顯示的是最新一次考試的結果。"
    )

# title -> (results folder prefix, subset)
GROUPS = {"模板問題": ("bank", "template"), "說法對齊": ("bank", "alias"),
          "攔截問題": ("synthetic", "block"), "複雜問題": ("synthetic", "complex")}
group = st.sidebar.radio("題組", list(GROUPS), help=(
    "模板問題與說法對齊：手工挑選的小題庫（data/banks/）。攔截問題：不該查資料的輸入，"
    "系統不得執行任何 SELECT。複雜問題：34 種 SQL-first 複雜查詢與其改寫。"))
BLOCK_KIND = {"write_request": "寫入／修改請求", "chitchat": "閒聊", "prompt_injection": "提示注入",
              "not_in_data": "資料庫外的問題"}
CHART = {"bar": "長條圖", "line": "折線圖", "pie": "圓餅圖"}
SYNTHETIC_RUNS = data_dir / "synthetic" / "runs"


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


def load_items(path):
    rows = (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    return {r["id"]: r for r in rows}


def synthetic_items(dataset):
    """Items of the graded dataset by id. Metadata names it relative to data/
    (banks/<name>/dataset.jsonl); older synthetic runs wrote synthetic/<run>/dataset.jsonl."""
    if not dataset:
        return {}
    path = data_dir / dataset
    if not path.exists():
        path = SYNTHETIC_RUNS / pathlib.Path(dataset).parent.name / "dataset.jsonl"
    return load_items(path) if path.exists() else {}


def chart_name(plot):
    plot = plot or {}
    return CHART.get(plot.get("chart_type"), plot.get("chart_type")) if plot.get("should_plot") else "不畫圖"


def group_of(subset, row, item, seeds):
    if subset == "block":
        return BLOCK_KIND.get(row["category"], row["category"])
    if subset == "template":
        return chart_name(item.get("expected_plot"))
    if subset == "alias":
        return "地區叫法" if item.get("column") == "Region" else "國家叫法"
    return seeds.get(row["seed_id"], row["seed_id"])


def item_label(subset, row, item):
    if subset == "template":
        return f"{row['question'][:80]}（{item.get('ability', '')}）"
    if subset == "alias":
        return f"{item.get('alias')} → {item.get('canonical')}"
    tag = "原題" if row.get("source") == "seed" else "改寫"
    return f"[{tag}] {row['question'][:90]}"


def show_table(columns, rows):
    # Regraded eval_e2e rows carry no column_names; DataFrame cells read missing keys as NaN.
    rows = rows if isinstance(rows, list) else []
    columns = columns if isinstance(columns, list) and rows and len(columns) == len(rows[0]) else None
    if rows:
        st.dataframe(pd.DataFrame(rows, columns=columns), hide_index=True)
    else:
        st.caption("（空結果）")


def show_bank_questions(subset, title):
    """Before the first run, list the bank so the reader still sees what it asks."""
    path = data_dir / "banks" / subset / "dataset.jsonl"
    st.info(f"{title}還沒有評估結果；以下是題庫裡的題目與標準答案。")
    if path.exists():
        items = load_items(path).values()
        st.dataframe(pd.DataFrame([{"題目": i["question"], "測什麼": i.get("ability") or i.get("canonical"),
                                    "預期圖表": chart_name(i.get("expected_plot")),
                                    "標準答案": json.dumps(i["ground_truth"]["rows"][:3], ensure_ascii=False)}
                                   for i in items]), hide_index=True, width="stretch")
    st.stop()


def show_synthetic(prefix, subset, title):
    runs = []
    for path in sorted(results_dir.glob(f"{prefix}*/synthetic_results.json"), reverse=True):
        records = [r for r in json.loads(path.read_text(encoding="utf-8")) if subset_of(r) == subset]
        if records:
            runs.append((path.parent.name, path.parent / "synthetic_summary.json", records))
    if not runs:
        show_bank_questions(subset, title) if prefix == "bank" else st.info(f"還沒有{title}的評估結果。")
        st.stop()
    _, summary_path, records = runs[0]  # newest run
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    meta = summary.get("run_metadata", {})
    if meta:
        model = str(meta.get("model", "")).removeprefix("openai/")
        st.caption(f"模型 `{model}` · 題庫 `{meta.get('dataset')}` · {summary.get('timestamp', '')}")

    items = synthetic_items(meta.get("dataset"))
    df = pd.DataFrame(records)
    df["_ok"] = df["passed"].astype(bool)
    seeds = seed_questions() if subset == "complex" else {}
    df["題型"] = [group_of(subset, row, items.get(row["id"]) or {}, seeds) for _, row in df.iterrows()]
    if subset == "block":
        metric, note = "攔截成功", "攔截成功：系統沒有執行任何 SELECT，也沒有發布查詢結果。"
    else:
        metric, note = "答案正確", ("答案正確：查詢結果與標準答案逐列相符，且有預期圖表的題目，"
                                    "畫不畫圖、圖表類型、X 軸順序都相符（程式判定，不用 LLM 評審）。")
    cols = st.columns(3)
    cols[0].metric("題數", len(df))
    cols[1].metric(metric, f"{df['_ok'].mean():.1%}")
    if (df["source"] == "rewrite").any():
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
            item = items.get(row.get("id")) or {}
            with st.expander(f"{'✅' if row['_ok'] else '❌'} {item_label(subset, row, item)}"):
                if subset == "alias":
                    st.markdown(f"**題目：** {row['question']}")
                st.caption(f"判定：{row.get('reason')}")
                truth = item.get("ground_truth")
                if subset == "block":
                    st.markdown("**系統回答：**")
                    st.write(row.get("answer") or "")
                    continue
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**標準答案：**")
                    if truth:
                        show_table(truth.get("columns"), truth.get("rows"))
                    else:
                        st.caption("（找不到這題的標準答案）")
                with col2:
                    st.markdown(f"**系統查詢結果**（{row.get('row_count', 0)} 列）：")
                    show_table(row.get("column_names"), row.get("rows"))
                if item.get("expected_plot"):
                    actual = {"should_plot": row.get("should_plot", bool(row.get("chart_type"))),
                              "chart_type": row.get("chart_type")}
                    st.markdown(f"**圖表：** 預期 `{chart_name(item['expected_plot'])}`，實際 `{chart_name(actual)}`")
                st.markdown("**SQL：**")
                st.code(row.get("sql_query") or "", language="sql")
                st.markdown("**系統回答：**")
                st.write(row.get("answer") or "")
    st.stop()


show_synthetic(*GROUPS[group], group)
