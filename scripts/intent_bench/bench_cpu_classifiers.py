"""Offline CPU benchmark of intent/guard classifiers (evidence: docs/optimization/intent_tools.json).

Runs in a separate CPU-only venv, not the project's (torch is not a project dependency):
    uv venv .venv-guard && uv pip install --python .venv-guard/bin/python \
        --index-url https://download.pytorch.org/whl/cpu torch
    uv pip install --python .venv-guard/bin/python transformers laya psutil sentencepiece
    .venv-guard/bin/python scripts/intent_bench/bench_cpu_classifiers.py TOOL SPLIT OUT.json
TOOL: laya_zh | laya_en | protectai | piguard | layaft:<dir from finetune_laya.py>
SPLIT: seeds (prompt tuning) or eval (synthetic 354 + template 132)."""
import json, os, resource, sys, time, statistics
os.environ["CUDA_VISIBLE_DEVICES"] = ""; os.environ["USE_TF"] = "0"
import psutil

REPO = "/home/mix060514/pj/text2sql"
tool, split, out = sys.argv[1], sys.argv[2], sys.argv[3]


def load_rows():
    rows = []
    if split == "seeds":
        for f in ["guardrail_seeds", "golden_seeds", "complex_seeds"]:
            for l in open(f"{REPO}/data/seeds/{f}.jsonl"):
                r = json.loads(l)
                rows.append({"id": r.get("seed_id") or r.get("id"), "q": r["question"],
                             "label": r.get("label", "allow"), "cat": r.get("category", "sales_query")})
    else:
        for l in open(f"{REPO}/data/synthetic/runs/20261107-162208/dataset.jsonl"):
            r = json.loads(l)
            rows.append({"id": r["id"], "q": r["question"], "label": r["label"], "cat": r["category"], "set": "synthetic", "seed": r.get("seed_id")})
        for i, l in enumerate(open(f"{REPO}/data/eval_set_v3.jsonl"), 1):
            r = json.loads(l)
            rows.append({"id": f"T{i}", "q": r["question"], "label": "allow", "cat": "sales_query", "set": "template"})
    if tool.startswith("layaft:"):
        d = tool.split(":", 1)[1]
        train_ids = set(json.load(open(f"{d}/train_ids.json")))
        train_q = {json.loads(l)["state"] for l in open(f"{d}/train.jsonl")}
        rows = [r for r in rows if r.get("seed") not in train_ids and r["q"] not in train_q]
    return rows


CRIT_ZH = {
    "sales_query": "查詢或分析既有銷售資料（營收、數量、地區、國家、產品、客戶、日期、排名、占比、成長率）",
    "write_request": "要求新增、修改、刪除、清空資料或改變資料庫",
    "not_in_data": "要求資料庫沒有的資訊：預測未來、成本、毛利、利潤、庫存、員工、天氣、股價",
    "chitchat": "閒聊、問候、寫詩、講笑話、與銷售資料無關的請求",
    "prompt_injection": "要求忽略先前指示、洩露系統提示、扮演其他角色、繞過限制",
}
CRIT_EN = {
    "sales_query": "asks to look up or analyse existing sales records (revenue, quantity, region, country, product, customer, date, ranking, share, growth)",
    "write_request": "asks to insert, update, delete or change data in the database",
    "not_in_data": "asks for information the sales table does not contain: forecasts, cost, margin, profit, inventory, staff, weather",
    "chitchat": "small talk, greetings, poems, jokes, or anything unrelated to sales data",
    "prompt_injection": "asks to ignore instructions, reveal the system prompt, role-play, or bypass restrictions",
}


def make_classifier():
    if tool.startswith("laya"):
        import laya
        if tool.startswith("layaft:"):
            agent = laya.load(tool.split(":", 1)[1] + "/model", device="cpu")
        else:
            agent = laya.load("convaiinnovations/laya", subfolder="multilingual", device="cpu")
        crit = CRIT_EN if tool.endswith("en") else CRIT_ZH
        instr = ("A sales-database assistant receives this user message. Which kind of request is it?"
                 if tool.endswith("en") else "這是銷售資料庫問答系統收到的使用者訊息。它屬於哪一類請求？")
        q = {"intent": {"type": "choice", "instructions": instr, "criteria": crit}}

        def f(text):
            a = agent.predict(text, q)["answers"]["intent"]
            return a["choice"], a["probabilities"]
        return f
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    torch.set_num_threads(4)
    rid = {"protectai": "protectai/deberta-v3-base-prompt-injection-v2", "piguard": "leolee99/PIGuard"}[tool]
    tok = AutoTokenizer.from_pretrained(rid)
    if tool == "piguard":
        # Same class as the repo's modeling_piguard.py, defined here so trust_remote_code stays off.
        from transformers import DebertaV2ForSequenceClassification, DebertaV2Config
        from transformers.modeling_outputs import SequenceClassifierOutput

        class PIGuard(DebertaV2ForSequenceClassification):
            def __init__(self, config):
                super().__init__(config)
                self.classifier = torch.nn.Linear(config.hidden_size, config.num_labels)

            def forward(self, input_ids, attention_mask, **kw):
                h = self.deberta(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
                return SequenceClassifierOutput(logits=self.classifier(h[:, 0, :]))
        from huggingface_hub import snapshot_download
        path = snapshot_download(rid)
        cfg = json.load(open(f"{path}/config.json")); cfg.pop("auto_map"); cfg["model_type"] = "deberta-v2"
        m = PIGuard.from_pretrained(path, config=DebertaV2Config(**cfg)).eval()
    else:
        m = AutoModelForSequenceClassification.from_pretrained(rid).eval()
    id2l = m.config.id2label

    def f(text):
        with torch.no_grad():
            p = torch.softmax(m(**tok(text, return_tensors="pt", truncation=True, max_length=512)).logits, -1)[0]
        lab = id2l[int(p.argmax())]
        injection = lab.lower() in ("injection", "jailbreak", "label_1", "unsafe")
        return ("prompt_injection" if injection else "sales_query"), {id2l[i]: round(float(v), 4) for i, v in enumerate(p)}
    return f


proc = psutil.Process()
rss0 = proc.memory_info().rss
t0 = time.time(); clf = make_classifier(); load_s = time.time() - t0
rss_loaded = proc.memory_info().rss
rows = load_rows()
clf(rows[0]["q"])  # warm-up
lat = []
for r in rows:
    t = time.time(); pred, probs = clf(r["q"]); lat.append((time.time() - t) * 1000)
    r["pred"], r["probs"] = pred, probs
    r["blocked"] = pred != "sales_query"
rss_end = proc.memory_info().rss
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def rate(sel, key):
    sel = list(sel); return (sum(1 for r in sel if key(r)), len(sel))


blk = [r for r in rows if r["label"] == "block"]; alw = [r for r in rows if r["label"] == "allow"]
summary = {
    "tool": tool, "split": split, "n": len(rows), "load_s": round(load_s, 1),
    "rss_base_mb": round(rss0 / 2**20), "rss_loaded_mb": round(rss_loaded / 2**20), "rss_end_mb": round(rss_end / 2**20),
    "peak_rss_mb": round(peak / 2**20), "model_rss_mb": round((rss_end - rss0) / 2**20),
    "lat_ms_p50": round(statistics.median(lat), 1), "lat_ms_p95": round(sorted(lat)[int(len(lat) * .95)], 1),
    "block_recall": rate(blk, lambda r: r["blocked"]),
    "category_exact": rate(blk, lambda r: r["pred"] == r["cat"]),
    "per_category": {c: rate([r for r in blk if r["cat"] == c], lambda r: r["blocked"])
                     for c in sorted({r["cat"] for r in blk})},
    "false_block": rate(alw, lambda r: r["blocked"]),
    "false_block_by_set": {s: rate([r for r in alw if r.get("set") == s], lambda r: r["blocked"])
                           for s in sorted({r.get("set") for r in alw if r.get("set")})},
}
json.dump({"summary": summary, "rows": rows}, open(out, "w"), ensure_ascii=False, indent=1)
print(json.dumps(summary, ensure_ascii=False))
