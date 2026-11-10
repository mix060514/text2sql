#!/usr/bin/env python3
"""Concurrency benchmark for the local llama.cpp server (evidence for the 優化成果 page).

Keeps N real agent_sql requests in flight (closed loop: N clients, each sends its next
request when the previous returns) (few-shot instruction, thinking budget,
max_output_tokens as configured) to a llama-server started with --parallel N, and
records per-request latency, generated tokens, aggregate tokens/s, prompt-cache hits
and truncation. Results are appended to docs/optimization/concurrency.json.

    uv run python scripts/bench_concurrency.py --base http://127.0.0.1:18090 --parallel 4
    # 2026-11-09 sweep: 50 distinct questions per setting, separate output file
    uv run python scripts/bench_concurrency.py --parallel 4 --requests 50 --question-set synthetic \
        --drafting dflash2 --out docs/optimization/concurrency_sweep.json
"""

import argparse
import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from text2sql import config  # noqa: E402
from text2sql.agents.planned_agent import qwen_workflow as qw  # noqa: E402

CTX_TOTAL = 65536  # --ctx-size of the server under test (kv-unified: shared by all slots)
OUT = ROOT / "docs" / "optimization" / "concurrency.json"
# Same type as real traffic, none of them evaluation questions.
QUESTIONS = [
    "Germany 在 2024 年每一季的銷售額是多少？",
    "2023 年 Software 類別銷售量最高的三個國家是哪些？",
    "LATAM 區域 2024 年的訂單數是多少？",
    "Canada 在 2023 至 2024 年每個月的銷售量趨勢。",
    "Office Supplies 在各區域 2024 年的銷售額占比為何？",
    "2024 年 APAC 銷售額前三大的客戶是誰？",
    "India 2024 年相較 2023 年的銷售額成長率是多少？",
    "每個產品類別 2023 年的平均單價是多少？",
]


def synthetic_questions(n: int, seed: int = 20261109) -> list[str]:
    """n distinct allow questions from the latest synthetic run, fixed sample (speed only, not graded)."""
    import random
    runs = sorted((ROOT / "data" / "synthetic" / "runs").glob("*/dataset.jsonl"))
    rows = [json.loads(l) for l in runs[-1].read_text(encoding="utf-8").splitlines() if l.strip()]
    qs = sorted({r["question"] for r in rows if r.get("label") == "allow"})
    return random.Random(seed).sample(qs, n)


def server_rss_mib(base):
    """RSS of the llama-server listening on base's port (host RAM evidence)."""
    port = base.rsplit(":", 1)[-1].strip("/")
    try:
        pids = subprocess.run(["pgrep", "-f", f"llama-server.*--port {port} "], capture_output=True,
                              text=True, timeout=5).stdout.split()
        return max(int(open(f"/proc/{p}/status").read().split("VmRSS:")[1].split()[0]) // 1024 for p in pids)
    except Exception:
        return None


def gpu_mem_mib():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5).stdout
        return int(out.split()[0])
    except Exception:
        return None


async def one(client, base, model, question):
    system = qw.sql_instruction(True).replace("{user_question}", question)
    body = {
        "model": model, "stream": False,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": question}],
        "temperature": 0.1, "max_tokens": config.SQL_THINKING_BUDGET + 2048,
        "thinking_budget_tokens": config.SQL_THINKING_BUDGET,
    }
    start = time.monotonic()
    r = await client.post(f"{base}/v1/chat/completions", json=body)
    r.raise_for_status()
    d = r.json()
    t = d.get("timings") or {}
    msg = d["choices"][0]["message"]
    return {
        "seconds": round(time.monotonic() - start, 1),
        "completion_tokens": d["usage"]["completion_tokens"],
        "prompt_tokens": d["usage"]["prompt_tokens"],
        "cached_tokens": t.get("cache_n"),
        "decode_tps": round(t.get("predicted_per_second") or 0, 1),
        "finish_reason": d["choices"][0].get("finish_reason"),
        "answer_empty": not (msg.get("content") or "").strip(),
    }


async def bench(base, model, parallel, rounds, questions=None, requests=None):
    questions = questions or QUESTIONS
    results = []
    async with httpx.AsyncClient(timeout=900) as client:
        await one(client, base, model, questions[0])  # warm-up: loads the shared prompt prefix
        mem = gpu_mem_mib()
        # Closed loop, like vLLM's serving benchmark: N clients each send their next request
        # as soon as the previous one returns, so the server always has N requests in flight
        # and llama.cpp's continuous batching admits each one into a free slot immediately.
        queue = asyncio.Queue()
        for i in range(requests or rounds * parallel):
            queue.put_nowait(questions[i % len(questions)])

        async def client_loop():
            while not queue.empty():
                q = queue.get_nowait()
                results.append(await one(client, base, model, q))

        start = time.monotonic()
        await asyncio.gather(*(client_loop() for _ in range(parallel)))
        wall = time.monotonic() - start
        slots = (await client.get(f"{base}/slots")).json()
    tokens = sum(x["completion_tokens"] for x in results)
    return {
        "parallel": parallel, "requests": len(results), "wall_seconds": round(wall, 1),
        "aggregate_tps": round(tokens / wall, 1),
        "avg_request_seconds": round(sum(x["seconds"] for x in results) / len(results), 1),
        "max_request_seconds": max(x["seconds"] for x in results),
        "avg_decode_tps_per_request": round(sum(x["decode_tps"] for x in results) / len(results), 1),
        "avg_prompt_tokens": round(sum(x["prompt_tokens"] for x in results) / len(results)),
        "avg_cached_tokens": round(sum((x["cached_tokens"] or 0) for x in results) / len(results)),
        "max_completion_tokens": max(x["completion_tokens"] for x in results),
        "truncated": sum(x["finish_reason"] == "length" for x in results),
        "empty_answers": sum(x["answer_empty"] for x in results),
        "n_ctx_per_slot": slots[0].get("n_ctx") if slots else None, "ctx_size_total": CTX_TOTAL,
        "gpu_mem_mib": mem, "server_rss_mib": server_rss_mib(base),
        "sql_thinking_budget": config.SQL_THINKING_BUDGET,
        "max_output_tokens": config.SQL_THINKING_BUDGET + 2048,
        "requests_detail": results,
        "measured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:18090")
    ap.add_argument("--model", default=config.LLAMA_CPP_MODEL.split("/")[-1])
    ap.add_argument("--parallel", type=int, required=True)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--drafting", default="mtp", help="speculative drafter of the server under test (label only)")
    ap.add_argument("--requests", type=int, help="total requests (default rounds x parallel)")
    ap.add_argument("--question-set", choices=["bench", "synthetic"], default="bench")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--note", default="", help="server settings to record with the row")
    args = ap.parse_args()
    questions = synthetic_questions(args.requests or 50) if args.question_set == "synthetic" else QUESTIONS
    row = asyncio.run(bench(args.base, args.model, args.parallel, args.rounds, questions, args.requests))
    row["drafting"] = args.drafting
    row["question_set"] = args.question_set
    row["note"] = args.note
    out = args.out
    data = json.loads(out.read_text(encoding="utf-8")) if out.exists() else []
    key = lambda d: (d.get("drafting", "mtp"), d["parallel"])
    data = [d for d in data if key(d) != key(row)] + [row]
    out.write_text(json.dumps(sorted(data, key=key), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in row.items() if k != "requests_detail"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
