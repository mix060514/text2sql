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


def flow(dot: str, caption: str):
    """A small left-to-right flow chart describing how a measurement was made."""
    st.graphviz_chart("digraph { rankdir=LR; bgcolor=transparent; "
                      "node [shape=box, style=\"rounded,filled\", fillcolor=\"#eef2ff\", color=\"#6b7bd6\", "
                      "fontname=\"Noto Sans TC\", fontsize=11]; edge [color=\"#8c959f\", fontsize=10, "
                      "fontname=\"Noto Sans TC\"]; " + dot + " }", width="stretch")
    st.caption(caption)


toc: list[tuple[int, str, str]] = []  # (level, title, anchor) for the sidebar table of contents


def section(title: str, anchor: str):
    """A subheader that is also listed in the sidebar table of contents."""
    st.subheader(title, anchor=anchor)
    toc.append((0, title, anchor))


def metric_row(before, after, labels=("成功率", "逾時率", "平均耗時")):
    cols = st.columns(len(labels))
    spec = {"成功率": ("success", "pct", "normal"), "逾時率": ("timeout", "pct", "inverse"),
            "平均耗時": ("avg_seconds", "sec", "inverse"), "一次解決率": ("first_try", "pct", "normal")}
    for col, label in zip(cols, labels):
        key, fmt, color = spec[label]
        show = pct if fmt == "pct" else sec
        value = (after or {}).get(key)
        before_v = (before or {}).get(key)
        # Show "before → after" directly; a lone after-value hides what changed.
        shown = show(value) if before_v is None else f"{show(before_v)} → {show(value)}"
        col.metric(label, shown, delta(before, after, key, fmt), delta_color=color)


st.markdown(
    "本頁整理問答系統在本機 27B 模型上**難以一次解決的題型**、找到的根本原因、做了什麼修改，"
    "以及修改前後的實測數字。所有數字由評估結果檔案自動計算"
    "（`scripts/build_optimization_report.py`），不是手動填寫。"
)

# ---- Overview ---------------------------------------------------------------------------
overall = report["overall"]["template"]
section("模板題整體（132 題）", "overall-template")
stages = [("2026-11-07 第一次完整評估", overall.get("2026-11-07")),
          ("基礎設施與評分修正後", overall.get("baseline")),
          ("定版（few-shot＋單一時限＋思考 4096）", overall.get("few_shot"))]
limit = report.get("question_seconds")
table = pd.DataFrame([
    {"階段": name, "成功率": pct(s["success"]), "逾時率": pct(s["timeout"]),
     "一次解決率": pct(s["first_try"]), "平均耗時": sec(s["avg_seconds"])}
    for name, s in stages if s
])
st.dataframe(table, hide_index=True, width="stretch")
st.caption("成功率＝SQL 結果正確且圖表檢查通過；一次解決率＝第一次寫出的 SQL 就成功（不需重試）；"
           "平均耗時含 LLM 評審。2026-11-07 的評估當時尚未記錄耗時與重試次數。")
st.caption("定版評估一次送 4 題（伺服器 4 路並發），每題時限 "
           + (f"{limit:.0f} 秒" if limit else "見設定")
           + "；基準一次 1 題。所以定版的「平均耗時」是 4 題同時跑時每題從送出到完成的時間，"
           "整批跑完的總時間反而比較短（同批題目約快 1.6 倍）。")

cx = report["overall"].get("complex") or {}
if cx.get("baseline"):
    section("複雜題整體（34 種題型 × 原題與 2 個改寫）", "overall-complex")
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
section("優化案例", "cases")
for i, case in enumerate(report["cases"], 1):
    with st.container(border=True):
        # Streamlit rewrites heading ids, so the anchor is a separate element in the same block.
        st.markdown(f'<span id="case-{i}" style="display:block; scroll-margin-top:4.5rem"></span>\n\n#### {case["title"]}', unsafe_allow_html=True)
        toc.append((1, case["title"], f"case-{i}"))
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
section("Few-shot：難題型加入同類範例", "few-shot")
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
    section("逐題變化：基準 → 定版", "changes")
    st.markdown("同一批題目在基準與定版各跑一次。除了看整體成功率，也逐題檢查**原本答對、定版答錯**的題目與原因，"
                "確認優化沒有把原本對的題目改壞；模型生成本身有隨機性，少數翻轉可能只是波動。")
    for key, name in (("template", "模板題"), ("complex", "複雜題（人工複核後）")):
        c = changes.get(key)
        if not c:
            continue
        st.markdown(f"**{name}**：修好 {len(c['fixed'])} 題、由對變錯 {len(c['regressed'])} 題")
        if c["regressed"]:
            st.dataframe(pd.DataFrame([{"題目": r["question"], "題型": r["type"], "原因": r["reason"]}
                                       for r in c["regressed"]]), hide_index=True, width="stretch",
                         column_config={"原因": st.column_config.TextColumn(width="large")})
        if c["fixed"]:
            with st.expander(f"{name}修好的題目"):
                st.dataframe(pd.DataFrame([{"題目": r["question"], "題型": r["type"]} for r in c["fixed"]]),
                             hide_index=True, width="stretch")

intent = report.get("intent") or {}
ib = intent.get("block") or {}
names = {"write_request": "要求改資料", "not_in_data": "資料庫沒有的資料（預測、毛利）",
         "chitchat": "閒聊、寫詩", "prompt_injection": "提示注入、越權"}
if ib.get("before") or ib.get("after"):
    st.divider()
    section("攔截題：加入意圖判斷前後", "intent-block")
    st.markdown("不該執行查詢的問題（改資料、資料庫沒有的資料、閒聊、提示注入）：成功＝沒有執行任何 SELECT 就拒絕。"
                "2026-11-09 在寫 SQL 之前加入「意圖判斷」步驟：模型只輸出類別，拒答由程式用固定文字執行，不寫也不執行 SQL。")
    metric_row(ib.get("before"), ib.get("after"), ("成功率", "逾時率", "平均耗時"))
    with st.expander("怎麼測的（流程圖）", expanded=False):
        flow('q [label="使用者問題"]; i [label="agent_intent\\n本機模型、不思考"]; '
             'r [label="程式固定文字拒答\\n不寫、不執行 SQL", fillcolor="#fde8e8", color="#cf222e"]; '
             'w [label="agent_sql → 唯讀執行 → 審查\\n→ 報告 → 發布"]; '
             'q -> i; i -> r [label="改資料／資料沒有／閒聊／注入"]; i -> w [label="sales_query"];',
             "加入的步驟：模型只輸出類別與理由，拒答由程式執行；理由只進執行紀錄，不進回答。")
        flow('d [label="合成考卷 354 題\\n＋模板考卷 132 題"]; s [label="臨時伺服器 18090\\nrouter 同參數、一次 4 題"]; '
             'b [label="攔截 126 題\\n沒有任何 SELECT 成功＝擋下"]; c [label="複雜 102 題\\n資料列與 SQLite 標準答案比對\\n＋人工複核規則"]; '
             't [label="模板 132 題\\nLLM 評審＋程式檢查圖表"]; v [label="與加入前的定版逐題比較\\n誤擋＝回答是固定拒答文字"]; '
             'x [label="逾時題單獨重跑\\n分辨並發還是題目本身"]; '
             'd -> s; s -> b; s -> c; s -> t; b -> v; c -> v; t -> v; v -> x;',
             "評估流程：同一份考卷、同一組伺服器參數，只差有沒有意圖判斷；由對變錯的題目逐題寫原因。")
    rows = []
    for label, cats, overall in (("加入前", ib.get("before_by_category") or {}, ib.get("before")),
                                 ("加入後", ib.get("after_by_category") or {}, ib.get("after"))):
        if not overall:
            continue
        rows.append({"版本": label, "類別": "全部", "題數": overall["n"], "正確攔截率": pct(overall["success"]),
                     "逾時率": pct(overall["timeout"]), "平均耗時": sec(overall["avg_seconds"])})
        rows += [{"版本": label, "類別": names.get(k, k), "題數": v["n"], "正確攔截率": pct(v["success"]),
                  "逾時率": pct(v["timeout"]), "平均耗時": sec(v["avg_seconds"])} for k, v in cats.items()]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    if ib.get("before"):
        st.caption("加入前：流程沒有「先判斷該不該查」，每題直接寫 SQL。寫入全被唯讀檢查（READ_ONLY）擋下、資料沒被改，"
                   "但模型重試到用完或逾時，還會把詩寫進 SELECT、把過去成長率當預測。")
    if ib.get("missed"):
        with st.expander(f"加入後仍未攔下的 {len(ib['missed'])} 題"):
            st.dataframe(pd.DataFrame(ib["missed"]), hide_index=True, width="stretch")
    examples = (report.get("block") or {}).get("examples") or []
    if examples:
        with st.expander("加入前的實例"):
            st.dataframe(pd.DataFrame(examples), hide_index=True, width="stretch")

allow = intent.get("allow") or {}
if allow:
    section("放行題有沒有被誤擋", "intent-allow")
    st.markdown("同一份模板考卷與複雜題，跟加入意圖判斷前的定版比較。誤擋＝回答是固定拒答文字（逐題列出）。")
    rows = []
    for key, name in (("template", "模板題"), ("complex", "複雜題（複核後）")):
        a = allow.get(key)
        if not a:
            continue
        for label, st_ in (("加入前", a.get("before")), ("加入後", a.get("after"))):
            if st_:
                rows.append({"考卷": name, "版本": label, "題數": st_["n"], "成功率": pct(st_["success"]),
                             "逾時率": pct(st_["timeout"]), "平均耗時": sec(st_["avg_seconds"]),
                             "誤擋": str(len(a["false_blocks"])) if label == "加入後" else "–"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    for key, name in (("template", "模板題"), ("complex", "複雜題")):
        a = allow.get(key) or {}
        if a.get("false_blocks"):
            st.warning(f"{name}誤擋 {len(a['false_blocks'])} 題")
            st.dataframe(pd.DataFrame([{"題目": r["question"], "被判為": names.get(r["refused_as"], r["refused_as"]),
                                        "加入前是否答對": "是" if r["was_correct_before"] else "否",
                                        "原因": r.get("reason", "")}
                                       for r in a["false_blocks"]]), hide_index=True, width="stretch")
        elif a:
            st.success(f"{name}：沒有任何一題被誤擋。")
        if a.get("regressed"):
            with st.expander(f"{name}由對變錯 {len(a['regressed'])} 題（含誤擋）與原因"):
                st.dataframe(pd.DataFrame([{"題目": r["question"], "原因": r["reason"]} for r in a["regressed"]]),
                             hide_index=True, width="stretch",
                             column_config={"原因": st.column_config.TextColumn(width="large")})
        if a.get("fixed"):
            with st.expander(f"{name}由錯變對 {len(a['fixed'])} 題"):
                st.dataframe(pd.DataFrame({"題目": a["fixed"]}), hide_index=True, width="stretch")
    if intent.get("timeout_reruns"):
        with st.expander("加入意圖後逾時的題目：單獨重跑"):
            st.dataframe(pd.DataFrame(intent["timeout_reruns"]), hide_index=True, width="stretch")

tools = (intent.get("tools") or {})
if tools.get("tools"):
    section("意圖判斷用什麼工具：實測比較", "intent-tools")
    st.markdown(
        "條件：不能佔用 GPU、不能影響本機模型。先試 Jev 類型的開源判斷模型（Jev 本身只有託管 API；開源對應是 Laya）"
        "與兩個開源提示注入分類器，全部在 CPU 上跑；再與「用已載入的本機模型判斷」比較。"
        "題目：合成考卷攔截 126 題＋放行 228 題、模板考卷 132 題（離線分類，不執行 SQL）。")
    trs = []
    for t in tools["tools"]:
        for i, e in enumerate(t.get("folds") or [t.get("eval")]):
            if not e:
                continue
            b, f = e["block_recall"], e["false_block"]
            trs.append({
                "工具": t["name"] + (f"（第 {i + 1} 折）" if t.get("folds") else ""), "類型": t["kind"],
                "授權": t["license"], "大小": t["params"],
                "攔截率": f"{b[0]}/{b[1]}（{b[0] / b[1]:.0%}）", "誤擋": f"{f[0]}/{f[1]}（{f[0] / f[1]:.1%}）",
                "類別完全正確": f"{e['category_exact'][0]}/{e['category_exact'][1]}",
                "主記憶體": f"{e['model_rss_mb']:,} MB" if "model_rss_mb" in e else "不新增",
                "顯存": "0（CPU）" if "model_rss_mb" in e else "不新增（沿用已載入模型）",
                "每題延遲 p50": f"{e['lat_ms_p50']:.0f} ms"})
    with st.expander("怎麼測的（流程圖）", expanded=False):
        flow('seed [label="種子題\\n攔截 42＋放行 76"]; tune [label="只用種子調整\\n選項說明／提示"]; '
             'cpu [label="CPU 工具\\n獨立 venv、torch CPU 版\\nCUDA_VISIBLE_DEVICES 為空"]; '
             'ft [label="Laya 微調\\n種子分組 2 折"]; llm [label="本機模型\\n不思考、一次短呼叫"]; '
             'ev [label="離線分類 486 題\\n攔截 126＋放行 360\\n不執行 SQL"]; '
             'm [label="攔截率、誤擋率、類別正確\\n主記憶體（RSS）、延遲\\nnvidia-smi 確認無 GPU 程序"]; '
             'seed -> tune; tune -> cpu; tune -> llm; seed -> ft; cpu -> ev; ft -> ev [label="只評未參與訓練的題"]; '
             'llm -> ev; ev -> m;',
             "工具比較流程：調整只看種子，成績只看考卷；微調模型只用沒見過的種子改寫題評分，避免拿訓練題考自己。")
    st.dataframe(pd.DataFrame(trs), hide_index=True, width="stretch")
    st.success("採用：" + tools.get("decision", ""))
    g = tools.get("gpu_evidence") or {}
    st.caption(f"不佔 GPU 的證據：CPU 工具在 torch CPU 版（{g.get('torch', '')}）、CUDA_VISIBLE_DEVICES 為空的子程序執行；"
               f"全部跑完後 nvidia-smi 顯存 {g.get('after_mib', '')}、沒有任何 GPU 程序。"
               "採用的本機模型方案不載入新模型：伺服器顯存與加入前相同，只多一次不思考的短呼叫。"
               + (" " + tools["prompt_guard_2"] if tools.get("prompt_guard_2") else ""))
    if tools.get("local_model_errors"):
        with st.expander("本機模型離線分類判錯的題目"):
            st.dataframe(pd.DataFrame(tools["local_model_errors"]), hide_index=True, width="stretch")

reruns = report.get("timeout_reruns") or []
if reruns:
    st.divider()
    section("逾時題單獨重跑", "timeout-reruns")
    st.markdown("定版評估一次送 4 題；為了分清逾時是並發造成還是題目本身難，把定版逾時的放行題在同一台伺服器上一次只送 1 題重跑。")
    st.dataframe(pd.DataFrame(reruns), hide_index=True, width="stretch")

conc = report.get("concurrency") or []
if conc:
    st.divider()
    section("並發測試：同時處理幾個請求", "concurrency")
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
            f"結果：MTP {best['parallel']} 路總吞吐量 {best['aggregate_tps']} tokens/秒，是 1 路的 "
            f"{best['aggregate_tps'] / base['aggregate_tps']:.1f} 倍；代價是單一請求平均 {best['avg_request_seconds']:.0f} 秒"
            f"（1 路 {base['avg_request_seconds']:.0f} 秒），顯存 {best['gpu_mem_mib']:,} MiB。")
        dfl = [r for r in conc if r.get("drafting") == "dflash2"]
        if dfl:
            d_best = max(dfl, key=lambda r: r["aggregate_tps"])
            d_one = next((r for r in dfl if r["parallel"] == 1), None)
            st.success(
                f"DFlash2（另一種草稿模型，2026-11-09 測）：1 路 {d_one['aggregate_tps']} tokens/秒、單一請求 "
                f"{d_one['avg_request_seconds']:.0f} 秒（MTP 1 路 {base['aggregate_tps']}、{base['avg_request_seconds']:.0f} 秒）；"
                f"{d_best['parallel']} 路 {d_best['aggregate_tps']} tokens/秒、單一請求 {d_best['avg_request_seconds']:.0f} 秒，"
                f"吞吐量和延遲都勝過 MTP {best['parallel']} 路（{best['aggregate_tps']}、{best['avg_request_seconds']:.0f} 秒）。"
                "代價：草稿模型多佔顯存，最多 2 路（4 路試了 5 種省顯存設定，不是開不起來就是第一個請求就崩潰，見下方）。"
                "草稿模型是為原版 Qwen3.8-27B 訓練、這裡配 Swift 1.5，接受率仍有約 50–75%；推測解碼由主模型逐 token 驗證，理論上輸出不變（未另跑答對率評估）；每種只有 12 個請求，倍數要保守看待。"
                "之後改用每種 50 題重測，DFlash2 沒有這個優勢，router 最後選 MTP 4 路（見下方「並發掃描」）。")
        st.info("2026-11-08 已套用到模型服務：4 路、各路共用 64k 上下文（kv-unified）、閒置 10 分鐘自動休眠釋放顯存"
                "（2026-11-09 再加主記憶體上限、拿掉視覺模組）。"
                "套用後以伺服器程序參數與 /slots 確認：4 路、每路可用 65,536 tokens，顯存 15.7 GB。")
        st.warning(
            "套用後真實運行才發現的副作用：4 路同時跑評估約 15 分鐘後，伺服器在**主記憶體**為每一路保存的 "
            "context checkpoint（每份約 150 MiB、每路預設最多 32 份）加上 8 GB prompt 快取，用到 9.9 GB，"
            "32 GB 主機開始換頁；swap 在傳統硬碟上，負載升到 103、IO 九成時間卡住。"
            "短時間的並發測試看不到這個問題（checkpoint 還沒累積）。"
            "處理：卸載模型恢復主機，並限制每路 checkpoint 數（ctx-checkpoints 4）與 prompt 快取大小（cache-ram 2048 MiB）。"
            "加上限制後跑完整個定版評估（約 6 小時、360 題、一次 4 題），伺服器主記憶體最高 5.2 GB（之前 9.9 GB），swap 沒有增加。")
    for r in failed:
        st.warning(f"{r.get('drafting', 'mtp').upper()} {r['parallel']} 路：{r['failed']}")
    st.caption("總吞吐量＝所有請求生成的 tokens ÷ 整批耗時；prompt 快取命中＝固定開頭（資料庫說明與範例）"
               "直接沿用 KV cache、不需重算的 token 數。")

sweep = report.get("concurrency_sweep") or {}
if sweep.get("rows"):
    st.divider()
    section("並發掃描：每種 50 題，找出 router 的設定（2026-11-09）", "concurrency-sweep")
    st.markdown(
        "上面的並發測試每種只有 12 個請求、題目 8 題輪流。這裡改用合成考卷固定抽出的 50 題，"
        "MTP 與 DFlash2 各從 1 路加到放不下為止；伺服器都不載入視覺模組、加主記憶體上限（與 router 相同）。"
        "只量「寫 SQL」這一步的速度，不評答對率。")
    with st.expander("怎麼測的（流程圖）", expanded=False):
        flow('q [label="固定 50 題\\n合成考卷放行題\\n固定亂數種子"]; srv [label="啟動臨時伺服器\\nMTP 或 DFlash2、N 路\\n不載入視覺模組、主記憶體上限"]; '
             'ok [label="health 正常？", shape=diamond, fillcolor="#fff4e0", color="#c27c0e"]; '
             'run [label="N 個客戶端同時送\\n真實 agent_sql 請求\\n一個回來就送下一個"]; '
             'rec [label="記錄總吞吐量、每題耗時\\n顯存、伺服器主記憶體\\n截斷、空答案"]; '
             'stop [label="放不下或崩潰\\n記錄原因、換下一種", fillcolor="#fde8e8", color="#cf222e"]; '
             'pick [label="選設定：吞吐量最高的 10% 內\\n挑單題最快"]; '
             'q -> srv [label="N＝1"]; srv -> ok; ok -> run [label="是"]; run -> rec; rec -> pick; '
             'rec -> srv [label="N＋1，重開", constraint=false, style=dashed]; '
             'ok -> stop [label="否"]; run -> stop [label="CUDA OOM"]; stop -> pick;',
             "掃描流程：每個設定都重開伺服器，讓 N 路一直塞滿；放不下時再試 token_embd 放 CPU、縮小 context。")
    rows = sweep["rows"]
    best = max(r["aggregate_tps"] for r in rows)
    st.dataframe(pd.DataFrame([{
        "草稿方式": r["drafting"].upper(), "路數": r["parallel"], "條件": r["variant"],
        "總吞吐量 (tokens/秒)": r["aggregate_tps"], "相對最高": f"{r['aggregate_tps'] / best:.0%}",
        "單題中位數": sec(r["median_seconds"]), "單題平均": sec(r["avg_seconds"]),
        "最慢 10%": sec(r["p90_seconds"]), "最慢": sec(r["max_seconds"]),
        "平均生成 tokens": r["avg_completion_tokens"], "顯存 (MiB)": r["gpu_mem_mib"],
        "伺服器主記憶體 (MiB)": r["server_rss_mib"], "截斷": r["truncated"], "空答案": r["empty_answers"]}
        for r in sorted(rows, key=lambda r: (r["drafting"], r["parallel"]))]), hide_index=True, width="stretch")
    with st.expander("放不下或崩潰的設定"):
        st.dataframe(pd.DataFrame(sweep.get("failures") or []), hide_index=True, width="stretch")
    st.success(
        "選定 MTP 4 路（已套用到 router，2026-11-09 15:39 重啟後以程序參數與 /slots 確認 4 路、每路 65,536）。"
        "規則：先找總吞吐量最高者，再從相差 10% 以內的設定中選單題最快者。MTP 5～7 路吞吐量只多約 10–13%，"
        "單題卻多等 4–6 成；DFlash2 2／3 路單題較快，但總吞吐量少 18%／4%（3 路單題平均也不比 MTP 4 路快）。")
    st.warning(
        "DFlash2 這次沒有之前量到的優勢（1 路 29.3 對 MTP 29.7 tokens/秒）。之前的 1.5 倍只有 12 個請求。"
        "整輪測試時 GPU 都在過熱降頻：89°C、核心時脈約 1 GHz（上限 2.1 GHz），驅動回報 SW Thermal Slowdown；"
        "所有設定在同樣狀態下比較，但絕對數字會比散熱良好時低。這次的 50 題比 8 題難，平均生成約 780 tokens（之前約 420），"
        "所以單題時間也比上面的表長。")
    st.info("拿掉視覺模組省約 1.2 GB 顯存；MTP 6 路因此放得下（之前差 885 MiB）。"
            "加主記憶體上限後，伺服器主記憶體最高 6.2 GB（6 路），主機沒有再進 swap。")

st.caption(f"報告產生時間：{report.get('generated_at', '')}")

with st.sidebar:
    st.markdown("**本頁目錄**\n\n" + "\n".join(
        f"{'    ' * level}- [{title}](#{anchor})" for level, title, anchor in toc))
