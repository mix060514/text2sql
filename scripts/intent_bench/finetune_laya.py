"""Grouped-CV fine-tune of Laya-multilingual on CPU. Usage: ft.py FOLD(0|1|all) OUTDIR"""
import json, os, sys, time, random
os.environ["CUDA_VISIBLE_DEVICES"] = ""; os.environ["USE_TF"] = "0"
import torch
torch.set_num_threads(12)
from huggingface_hub import snapshot_download
from laya.train import finetune, TrainConfig

sys.path.insert(0, os.path.dirname(__file__))
REPO = "/home/mix060514/pj/text2sql"
fold, outdir = sys.argv[1], sys.argv[2]
os.makedirs(outdir, exist_ok=True)
CRIT_ZH = {
    "sales_query": "查詢或分析既有銷售資料（營收、數量、地區、國家、產品、客戶、日期、排名、占比、成長率）",
    "write_request": "要求新增、修改、刪除、清空資料或改變資料庫",
    "not_in_data": "要求資料庫沒有的資訊：預測未來、成本、毛利、利潤、庫存、員工、天氣、股價",
    "chitchat": "閒聊、問候、寫詩、講笑話、與銷售資料無關的請求",
    "prompt_injection": "要求忽略先前指示、洩露系統提示、扮演其他角色、繞過限制",
}
Q = {"intent": {"type": "choice", "instructions": "這是銷售資料庫問答系統收到的使用者訊息。它屬於哪一類請求？", "criteria": CRIT_ZH}}

seeds = []
for f in ["guardrail_seeds", "golden_seeds", "complex_seeds"]:
    for l in open(f"{REPO}/data/seeds/{f}.jsonl"):
        r = json.loads(l); seeds.append((r["seed_id"], r["question"], r.get("category", "sales_query")))
ids = sorted({s[0] for s in seeds}); random.Random(7).shuffle(ids)
half = set(ids[: len(ids) // 2])
train_ids = set(ids) if fold == "all" else (half if fold == "0" else set(ids) - half)
rows = [{"state": q, "questions": Q, "gold": {"intent": {"probabilities": {k: float(k == c) for k in CRIT_ZH}}}}
        for sid, q, c in seeds if sid in train_ids]
data = f"{outdir}/train.jsonl"
with open(data, "w") as f:
    for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
json.dump(sorted(train_ids), open(f"{outdir}/train_ids.json", "w"))
base = snapshot_download("convaiinnovations/laya") + "/multilingual"
t = time.time()
summary = finetune(data, base, f"{outdir}/model",
                   TrainConfig(epochs=int(os.environ.get("EPOCHS", 6)), micro_batch=8, grad_accum=1, loss="soft-ce",
                               shuffle_options=("choice",), calib_frac=0.0), device="cpu")
print("train rows", len(rows), "seconds", round(time.time() - t), json.dumps({k: v for k, v in summary.items() if not isinstance(v, (list, dict))}, default=str)[:800])
