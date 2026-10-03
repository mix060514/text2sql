#!/usr/bin/env python3
"""Run the current branch's workflow and consume the complete invocation."""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from google.adk.runners import InMemoryRunner
from text2sql.agents.planned_agent.agent import root_agent
from text2sql.runtime import run_question


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="?", default="EMEA 的總銷售額是多少？")
    args = parser.parse_args()
    runner = InMemoryRunner(agent=root_agent, app_name="text2sql_cli")
    state = await run_question(runner, args.question, user_id="cli_user")
    print(state.get("final_answer", "No answer generated."))
    print(f"Status: {state.get('workflow_status')}")
    print(f"SQL: {state.get('sql_query', '')}")
    if state.get("workflow_error"):
        print(state["workflow_error"])
    return 0 if state.get("workflow_status") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
