"""Thin CLI adapter; Codex owns planning, tool use, and error recovery."""

import asyncio
import json
import os
from pathlib import Path
import signal
import sys
import tempfile

from text2sql.config import CODEX_BINARY, CODEX_MODEL, CODEX_REASONING_EFFORT
from .contracts import AnswerSubmission
from .prompts import DOMAIN, REPORT_RULES

SOURCE = Path(__file__).resolve().parents[3]


def command(workdir: Path) -> list[str]:
    args = [
        CODEX_BINARY, "exec", "--ignore-user-config", "--ephemeral", "--skip-git-repo-check",
        "--json", "--sandbox", "read-only", "--model", CODEX_MODEL,
        "--cd", str(workdir), "--output-schema", str(workdir / "schema.json"),
        "--output-last-message", str(workdir / "answer.json"),
        "-c", 'approval_policy="never"',
        "-c", "model_reasoning_effort=" + json.dumps(CODEX_REASONING_EFFORT),
        "-c", 'web_search="disabled"',
        "--enable", "skip_host_skill_discovery",
        "-c", "mcp_servers.sales.command=" + json.dumps(sys.executable),
        "-c", "mcp_servers.sales.args=" + json.dumps(["-m", "text2sql.agents.planned_agent.mcp_server"]),
        "-c", "mcp_servers.sales.env={PYTHONPATH=" + json.dumps(str(SOURCE))
              + ",TEXT2SQL_RECEIPT=" + json.dumps(str(workdir / "receipt.json")) + "}",
        "-c", "mcp_servers.sales.startup_timeout_sec=30",
        "-c", "mcp_servers.sales.tool_timeout_sec=15",
    ]
    # Only the business MCP tool is exposed; don't inherit personal connectors
    # or allow a shell to bypass its data limits.
    for feature in ("shell_tool", "unified_exec", "multi_agent", "apps", "plugins",
                    "browser_use", "computer_use"):
        args.extend(["--disable", feature])
    return args + ["-"]


async def stop_process(process):
    if process.returncode is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        await asyncio.wait_for(process.wait(), timeout=3)
    except asyncio.TimeoutError:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        await process.wait()


async def run_codex(question: str, previous_context: str = "{}"):
    prompt = DOMAIN + REPORT_RULES + """
你是銷售分析 Agent。使用 sales MCP 的 query_sales_data 查詢真實資料。
可以根據工具的錯誤修正查詢；工具直接驗證與執行 SQL。
回答必須依本次最後一次成功執行的結果。不要自行編造或使用常數查詢代替 sales_data。
若最後一次工具回傳失敗，先修正後重查，再提交回答。
完成後依提供的 JSON schema 提交 answer 與圖表規格。圖表由應用直接產生。
前一回合意圖只用於理解追問，仍要查詢本回合資料。
""" + "\n前一回合：" + previous_context + "\n使用者問題：" + question
    with tempfile.TemporaryDirectory(prefix="text2sql_codex_") as directory:
        workdir = Path(directory)
        schema = AnswerSubmission.model_json_schema()
        schema["required"] = list(schema["properties"])
        for field in schema["properties"].values():
            field.pop("default", None)
        (workdir / "schema.json").write_text(json.dumps(schema))
        process = await asyncio.create_subprocess_exec(
            *command(workdir), stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        try:
            stdout, stderr = await process.communicate(prompt.encode("utf-8"))
        finally:
            # The shared invocation deadline cancels us; also stop the MCP
            # child so an interrupted request cannot keep running.
            await stop_process(process)
        events = []
        for line in stdout.decode(errors="replace").splitlines():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        if process.returncode:
            failures = [event for event in events if event.get("type") in ("error", "turn.failed")]
            detail = json.dumps(failures[-1], ensure_ascii=False) if failures else stderr.decode(errors="replace")[-500:]
            raise RuntimeError("Codex exited unsuccessfully: " + detail[:500])
        receipt_path = workdir / "receipt.json"
        if not receipt_path.exists():
            raise ValueError("Codex did not execute a sales query; no answer can be published.")
        receipt = json.loads(receipt_path.read_text())
        if not receipt["state"].get("query_success"):
            raise ValueError("The final sales query failed: " + str(receipt["state"].get("query_error")))
        spec = AnswerSubmission.model_validate_json((workdir / "answer.json").read_text())
        usage = next((event.get("usage") for event in reversed(events)
                      if event.get("type") == "turn.completed"), None)
        return spec, receipt["state"], {
            "model": CODEX_MODEL, "query_calls": receipt["query_calls"],
            "event_count": len(events), "usage": usage,
        }
