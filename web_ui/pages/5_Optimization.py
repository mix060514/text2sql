"""優化成果: what was hard, why, what changed, and the measured before/after numbers.

All numbers come from docs/optimization/report.json, built by
scripts/build_optimization_report.py from evaluation result files.
"""

import json
import pathlib

import pandas as pd
import streamlit as st

project_root = pathlib.Path(__file__).parent.parent.parent.resolve()
report_path = project_root / "docs" / "optimization" / "report.json"

st.set_page_config(page_title="優化成果", page_icon="🛠️", layout="wide")
st.title("🛠️ 優化成果")

if not report_path.exists():
    st.info("還沒有優化報告。")
    st.stop()
report = json.loads(report_path.read_text(encoding="utf-8"))

pct = lambda v: "–" if v is None else f"{v:.0%}"
sec = lambda v: "–" if v is None else f"{v:.0f} 秒"


def delta(before, after, key, fmt, better="up"):
    if not before or not after or before.get(key) is None or after.get(key) is None:
        return None
    d = after[key] - before[key]
    text = (f"{d:+.0%}" if fmt == "pct" else f"{d:+.0f} 秒")
    return text


def metric_row(before, after, labels=("成功率", "逾時率", "平均耗時")):
    cols = st.columns(len(labels))
    spec = {"成功率": ("success", "pct", "normal"), "逾時率": ("timeout", "pct", "inverse"),
            "平均耗時": ("avg_seconds", "sec", "inverse"), "一次解決率": ("first_try", "pct", "normal")}
    for col, label in zip(cols, labels):
        key, fmt, color = spec[label]
        value = (after or {}).get(key)
        shown = pct(value) if fmt == "pct" else sec(value)
        before_v = (before or {}).get(key)
        help_text = f"優化前：{pct(before_v) if fmt == 'pct' else sec(before_v)}"
        col.metric(label, shown, delta(before, after, key, fmt), delta_color=color, help=help_text)


st.markdown(
    "本頁整理問答系統在本機 27B 模型上**難以一次解決的題型**、找到的根本原因、做了什麼修改，"
    "以及修改前後的實測數字。所有數字由評估結果檔案自動計算"
    "（`scripts/build_optimization_report.py`），不是手動填寫。"
)

# ---- Overview ---------------------------------------------------------------------------
overall = report["overall"]["template"]
st.subheader("模板題整體（132 題）")
stages = [("2026-11-07 第一次完整評估", overall.get("2026-11-07")),
          ("基礎設施與評分修正後", overall.get("baseline")),
          ("定版（few-shot＋單一時限＋思考 4096）", overall.get("few_shot"))]
table = pd.DataFrame([
    {"階段": name, "成功率": pct(s["success"]), "逾時率": pct(s["timeout"]),
     "一次解決率": pct(s["first_try"]), "平均耗時": sec(s["avg_seconds"])}
    for name, s in stages if s
])
st.dataframe(table, hide_index=True, width="stretch")
st.caption("成功率＝SQL 結果正確且圖表檢查通過；一次解決率＝第一次寫出的 SQL 就成功（不需重試）；"
           "平均耗時含 LLM 評審。2026-11-07 的評估當時尚未記錄耗時與重試次數。")

cx = report["overall"].get("complex") or {}
if cx.get("baseline"):
    st.subheader("複雜題整體（34 種題型 × 原題與 2 個改寫）")
    rows = [("基準（嚴格比對）", cx.get("baseline")), ("基準（人工複核後）", cx.get("baseline_reviewed")),
            ("定版（嚴格比對）", cx.get("few_shot")), ("定版（人工複核後）", cx.get("few_shot_reviewed"))]
    st.dataframe(pd.DataFrame([
        {"階段": name, "題數": s["n"], "成功率": pct(s["success"]), "逾時率": pct(s["timeout"]),
         "一次解決率": pct(s["first_try"]), "平均耗時": sec(s["avg_seconds"])}
        for name, s in rows if s
    ]), hide_index=True, width="stretch")
    st.caption("嚴格比對＝結果逐列等於 SQLite 標準答案；人工複核見下方「複雜題標準答案過嚴」案例。")

# ---- Cases ------------------------------------------------------------------------------
st.divider()
st.subheader("優化案例")
for case in report["cases"]:
    with st.container(border=True):
        st.markdown(f"#### {case['title']}")
        st.caption(f"範圍：{case.get('scope', '')}")
        c1, c2, c3 = st.columns(3)
        c1.markdown(f"**問題**\n\n{case['problem']}")
        c2.markdown(f"**根本原因**\n\n{case['cause']}")
        c3.markdown(f"**修改**\n\n{case['fix']}")
        if case.get("metrics"):
            m = case["metrics"]
            if m.get("after"):
                st.markdown(f"**修改前 → 修改後**（{m['before']['n']} 題）")
                labels = case.get("labels") or (
                    ("成功率", "逾時率", "平均耗時") if m["after"].get("avg_seconds") else ("成功率", "逾時率"))
                metric_row(m["before"], m["after"], labels)
            else:
                st.caption("修改後的數據將在下一次完整評估後更新。")
        if case.get("timing"):
            t = case["timing"]
            plain = t["plain"]
            plain_text = f"{plain['seconds']:.0f} 秒仍未完成" if not plain["finished"] else f"{plain['seconds']:.2f} 秒"
            cols = st.columns(2)
            cols[0].metric("一般 CTE", plain_text)
            cols[1].metric("AS MATERIALIZED", f"{t['materialized']['seconds']:.2f} 秒")

if report.get("complex_review"):
    with st.expander("複雜題人工複核明細"):
        st.dataframe(pd.DataFrame([
            {"題目": r["question"], "複核理由": r["reason"], "轉換後與標準答案一致": "是" if r["accepted"] else "否"}
            for r in report["complex_review"]
        ]), hide_index=True, width="stretch")

# ---- Few-shot ---------------------------------------------------------------------------
st.divider()
st.subheader("Few-shot：難題型加入同類範例")
st.markdown(
    "以「基礎設施與評分修正後」的評估為基準，找出**有失敗、需要重試或平均超過 120 秒**的題型，"
    "為它們各寫一個**同題型、不同內容**的範例（例如考題問 LATAM，範例用 Japan；考題問前五大，範例用前三大），"
    "範例題目不在任何評估集中（有自動測試檢查），只示範做法，不提供答案。"
)


def few_shot_table(rows: dict, name: str):
    if not rows:
        st.caption(f"{name}：基準評估完成後更新。")
        return
    data = []
    for t, m in rows.items():
        b, a = m["before"] or {}, m["after"] or {}
        data.append({"題型": t, "題數": b.get("n"),
                     "成功率（前→後）": f"{pct(b.get('success'))} → {pct(a.get('success'))}",
                     "一次解決率（前→後）": f"{pct(b.get('first_try'))} → {pct(a.get('first_try'))}",
                     "逾時率（前→後）": f"{pct(b.get('timeout'))} → {pct(a.get('timeout'))}",
                     "平均耗時（前→後）": f"{sec(b.get('avg_seconds'))} → {sec(a.get('avg_seconds'))}"})
    st.markdown(f"**{name}**")
    st.dataframe(pd.DataFrame(data), hide_index=True, width="stretch")


few = report.get("few_shot", {})
few_shot_table(few.get("template"), "模板題")
few_shot_table(few.get("complex"), "複雜題（34 種 SQL-first 題型，每種含原題與 2 個改寫）")

changes = report.get("changes") or {}
if any(changes.values()):
    st.divider()
    st.subheader("逐題變化：基準 → 定版")
    st.markdown("同一批題目在基準與定版各跑一次。除了看整體成功率，也逐題檢查**原本答對、定版答錯**的題目與原因，"
                "確認優化沒有把原本對的題目改壞；模型生成本身有隨機性，少數翻轉可能只是波動。")
    for key, name in (("template", "模板題"), ("complex", "複雜題（人工複核後）")):
        c = changes.get(key)
        if not c:
            continue
        st.markdown(f"**{name}**：修好 {len(c['fixed'])} 題、由對變錯 {len(c['regressed'])} 題")
        if c["regressed"]:
            st.dataframe(pd.DataFrame([{"題目": r["question"], "題型": r["type"], "原因": r["reason"]}
                                       for r in c["regressed"]]), hide_index=True, width="stretch")
        if c["fixed"]:
            with st.expander(f"{name}修好的題目"):
                st.dataframe(pd.DataFrame([{"題目": r["question"], "題型": r["type"]} for r in c["fixed"]]),
                             hide_index=True, width="stretch")

blk = (report.get("block") or {}).get("overall")
if blk:
    st.divider()
    st.subheader("攔截題（定版）")
    st.markdown("不該執行查詢的問題（越權、改資料、離題、提示注入等）：成功＝沒有執行任何 SELECT 就拒絕。")
    st.dataframe(pd.DataFrame([{"題數": blk["n"], "正確攔截率": pct(blk["success"]), "逾時率": pct(blk["timeout"]),
                                "平均耗時": sec(blk["avg_seconds"])}]), hide_index=True, width="stretch")

conc = report.get("concurrency") or []
if conc:
    st.divider()
    st.subheader("並發測試：同時處理幾個請求")
    st.markdown(
        "模型伺服器原本一次只處理一個請求，其他人要排隊。這裡用同一張 GPU、同樣的總上下文，"
        "測試同時處理 1 到 N 個寫 SQL 請求（真實 prompt、思考上限與輸出上限）時的吞吐量與單一請求耗時。"
    )
    failed = [r for r in conc if r.get("failed")]
    conc = [r for r in conc if not r.get("failed")]
    base = next((r for r in conc if r["parallel"] == 1 and r.get("drafting", "mtp") == "mtp"), None)
    st.dataframe(pd.DataFrame([
        {"草稿方式": r.get("drafting", "mtp").upper(), "並發路數": r["parallel"], "請求數": r["requests"],
         "總吞吐量 (tokens/秒)": r["aggregate_tps"],
         "相對 MTP 1 路": f"{r['aggregate_tps'] / base['aggregate_tps']:.2f}×" if base else "–",
         "單一請求平均耗時": sec(r["avg_request_seconds"]), "單一請求最長": sec(r["max_request_seconds"]),
         "每路生成速度 (tokens/秒)": r["avg_decode_tps_per_request"],
         "prompt 快取命中": f"{r['avg_cached_tokens']}/{r['avg_prompt_tokens']}",
         "輸出被截斷": r["truncated"], "空答案": r["empty_answers"],
         "顯存 (MiB)": r["gpu_mem_mib"]}
        for r in conc
    ]), hide_index=True, width="stretch")
    mtp_rows = [r for r in conc if r.get("drafting", "mtp") == "mtp"]
    if base and mtp_rows:
        best = max(mtp_rows, key=lambda r: r["aggregate_tps"])
        st.success(
            f"建議：多人使用時設 {best['parallel']} 路。總吞吐量 {best['aggregate_tps']} tokens/秒，是 1 路的 "
            f"{best['aggregate_tps'] / base['aggregate_tps']:.1f} 倍；代價是單一請求平均 {best['avg_request_seconds']:.0f} 秒"
            f"（1 路 {base['avg_request_seconds']:.0f} 秒），顯存 {best['gpu_mem_mib']:,} MiB。"
            "只有一個人使用（例如面試 demo）時 1 路最快。")
    for r in failed:
        st.warning(f"{r.get('drafting', 'mtp').upper()} {r['parallel']} 路：{r['failed']}")
    st.caption("總吞吐量＝所有請求生成的 tokens ÷ 整批耗時；prompt 快取命中＝固定開頭（資料庫說明與範例）"
               "直接沿用 KV cache、不需重算的 token 數。")

slp = report.get("sleep")
if slp:
    st.divider()
    st.subheader("閒置休眠：不用時釋放顯存")
    st.markdown(f"伺服器加上 `--sleep-idle-seconds`，閒置超過設定秒數就卸載模型、釋放顯存，下一個請求進來時自動載回"
                f"（類似 Ollama 的 keep_alive）。實測設定 {slp['sleep_idle_seconds']} 秒：")
    c = st.columns(4)
    c[0].metric("使用中顯存", f"{slp['vram_awake_mib']:,} MiB")
    c[1].metric(f"閒置 {slp['sleep_idle_seconds']} 秒後顯存", f"{slp['vram_idle_45s_mib']:,} MiB")
    c[2].metric("一般請求", f"{slp['awake_request_seconds']:.2f} 秒")
    c[3].metric("休眠後第一個請求", f"{slp['cold_start_request_seconds']:.2f} 秒")
    st.caption(slp["note"])

st.caption(f"報告產生時間：{report.get('generated_at', '')}")
