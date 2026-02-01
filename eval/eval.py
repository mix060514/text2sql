# %%
import json
import pathlib
from datetime import datetime
import time
import random
import numpy as np
import pandas as pd
import sys
import os
import asyncio
import logging
import uuid
from typing import Dict, Any, List
from dataclasses import dataclass, asdict
import warnings

warnings.filterwarnings("ignore")


from google.adk.agents import Agent
from google.adk.runners import InMemoryRunner
from google.genai import types

# Setup path to import local src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../eval")))

# Import Agents
from text2sql.agents.planned_agent.agent import root_agent
from judge.agent import judge_agent


# Shared runner instances (using ADK built-in InMemoryRunner)
eval_runner = InMemoryRunner(agent=root_agent, app_name="text2sql_eval")
judge_runner = InMemoryRunner(agent=judge_agent, app_name="judge_eval")


async def run_agent_and_get_state(
    runner: InMemoryRunner, prompt: str, user_id: str = "eval_user"
) -> dict:
    """執行 agent 並返回 session state"""
    session_id = f"{runner.app_name}_{int(time.time()*1000)}_{uuid.uuid4().hex[:6]}"

    await runner.session_service.create_session(
        app_name=runner.app_name, user_id=user_id, session_id=session_id
    )

    content = types.Content(role="user", parts=[types.Part(text=prompt)])

    # 消費完整 event stream
    async for _ in runner.run_async(
        user_id=user_id, session_id=session_id, new_message=content
    ):
        pass

    session = await runner.session_service.get_session(
        app_name=runner.app_name, user_id=user_id, session_id=session_id
    )
    return session.state if session else {}


PROJECT_DIR = pathlib.Path(__file__).parent.parent.resolve()
DATA_DIR = PROJECT_DIR / "data"
ground_truth_set = DATA_DIR / "eval_set_v2.jsonl"

# Load Eval Set
with open(ground_truth_set, "r", encoding="utf-8") as f:
    all_eval_set = [json.loads(line) for line in f]

# 隨機選取 5 個測試案例（測試完成後手動改回全部）
random.seed(42)  # 固定種子以便重現
eval_set = all_eval_set
eval_set = random.sample(all_eval_set, min(5, len(all_eval_set)))
print(f"Selected {len(eval_set)} test cases from {len(all_eval_set)} total")


# ============================================================
# Data Classes
# ============================================================
@dataclass
class AgentResult:
    """Agent 執行結果"""

    final_text: str
    executed_sql: str | None
    sql_result: Any


@dataclass
class JudgeResult:
    """Judge Agent 評估結果"""

    confidence_score: int
    is_correct: bool
    reasoning: str
    structured_output_success: bool = False


@dataclass
class EvalRow:
    """單筆評估結果"""

    id: int
    question: str
    expected: Any
    eval_type: str
    final_answer: str
    sql_query: str
    execute_result: Any
    llm_judge_score: int
    llm_judge_reasoning: str


# %% LLM-as-Judge Configuration (Using Judge Agent)


def _build_judge_prompt(
    question: str,
    expected_answer: Any,
    actual_answer: str,
    sql_query: str = None,
    sql_result: Any = None,
) -> str:
    """構造 Judge Agent 的評估 prompt"""
    return f"""請評估以下 Text-to-SQL Agent 的回答品質：

**使用者問題**: {question}

**預期正確答案**: {json.dumps(expected_answer, ensure_ascii=False)}

**SQL 查詢**: {sql_query if sql_query else "無法取得"}

**SQL 結果**: {json.dumps(sql_result, ensure_ascii=False) if sql_result else "無法取得"}

**Agent 回答**: {actual_answer}
"""


def _parse_judge_result(session_state: dict, fallback_text: str) -> JudgeResult:
    """解析 Judge Agent 的 structured output"""
    structured_output = session_state.get("judge_result")

    if not structured_output:
        return JudgeResult(
            confidence_score=0,
            is_correct=False,
            reasoning=f"無法解析 Judge Agent 輸出: {fallback_text[:100]}",
            structured_output_success=False,
        )

    required_keys = ["confidence_score", "is_correct", "reasoning"]
    missing_keys = [k for k in required_keys if k not in structured_output]

    if missing_keys:
        return JudgeResult(
            confidence_score=0,
            is_correct=False,
            reasoning=f"Structured output 缺少欄位: {missing_keys}",
            structured_output_success=False,
        )

    score = max(0, min(10, int(structured_output["confidence_score"])))
    return JudgeResult(
        confidence_score=score,
        is_correct=structured_output["is_correct"],
        reasoning=structured_output["reasoning"],
        structured_output_success=True,
    )


async def judge_confidence(
    question: str,
    expected_answer: Any,
    actual_answer: str,
    sql_query: str = None,
    sql_result: Any = None,
) -> JudgeResult:
    """使用 Judge Agent 評估回答正確性的信心分數 (0-10)。"""
    try:
        prompt = _build_judge_prompt(
            question, expected_answer, actual_answer, sql_query, sql_result
        )
        session_state = await run_agent_and_get_state(judge_runner, prompt)
        return _parse_judge_result(session_state, "")

    except Exception as e:
        logging.error(f"Judge error: {e}")
        return JudgeResult(
            confidence_score=0,
            is_correct=False,
            reasoning=f"Judge error: {str(e)}",
            structured_output_success=False,
        )


# %% Evaluation


@dataclass
class EvalMetrics:
    """評估指標"""

    confidence_scores: List[int] = None
    correct_count: int = 0
    total_count: int = 0

    def __post_init__(self):
        self.confidence_scores = []

    def add_result(self, confidence_score: int, is_correct: bool):
        self.confidence_scores.append(confidence_score)
        self.total_count += 1
        if is_correct:
            self.correct_count += 1

    def get_summary(self) -> Dict[str, Any]:
        return {
            "avg_confidence": (
                np.mean(self.confidence_scores) if self.confidence_scores else 0.0
            ),
            "accuracy": (
                self.correct_count / self.total_count if self.total_count > 0 else 0.0
            ),
            "correct_count": self.correct_count,
            "total_count": self.total_count,
        }


AGENT_TIMEOUT_SECONDS = 300


async def evaluate_question(
    agent, question: str, expected: Any
) -> tuple[AgentResult, JudgeResult]:
    """評估單一問題"""
    state = await asyncio.wait_for(
        run_agent_and_get_state(eval_runner, question), timeout=AGENT_TIMEOUT_SECONDS
    )
    result = AgentResult(
        final_text=state.get("final_answer", ""),
        executed_sql=state.get("sql_query"),
        sql_result=state.get("query_result"),
    )
    judge_result = await judge_confidence(
        question=question,
        expected_answer=expected,
        actual_answer=result.final_text,
        sql_query=result.executed_sql,
        sql_result=result.sql_result,
    )
    return result, judge_result


def create_eval_row(
    i: int,
    question: str,
    expected: Any,
    eval_type: str,
    result: AgentResult,
    judge_result: JudgeResult,
) -> EvalRow:
    """建立評估結果 row"""
    return EvalRow(
        id=i,
        question=question,
        expected=expected,
        eval_type=eval_type,
        final_answer=result.final_text or "",
        sql_query=result.executed_sql or "",
        execute_result=result.sql_result if result.sql_result else "",
        llm_judge_score=judge_result.confidence_score,
        llm_judge_reasoning=judge_result.reasoning,
    )


def create_error_row(
    i: int, question: str, expected: Any, eval_type: str, error: Exception
) -> EvalRow:
    """建立錯誤 row"""
    return EvalRow(
        id=i,
        question=question,
        expected=expected,
        eval_type=eval_type,
        final_answer=f"Error: {error}",
        sql_query="",
        execute_result="",
        llm_judge_score=0,
        llm_judge_reasoning=str(error),
    )


async def run_evaluation(agent: Agent, eval_set: List[Dict]):
    """執行評估"""
    output_dir = (
        PROJECT_DIR / "eval" / "results" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "eval_results.json"

    results: List[EvalRow] = []
    metrics = EvalMetrics()

    print(f"\n{'='*60}")
    print(f"Text-to-SQL Eval | Agent: {agent.name} | Questions: {len(eval_set)}")
    print(f"{'='*60}\n")

    for i, entry in enumerate(eval_set):
        question, expected = entry["question"], entry["ground_truth"]
        eval_type = entry.get("eval_type", "unknown")

        print(f"[{i+1}/{len(eval_set)}] {question[:50]}... ", end="", flush=True)

        try:
            result, judge_result = await evaluate_question(agent, question, expected)
            row = create_eval_row(
                i, question, expected, eval_type, result, judge_result
            )
            metrics.add_result(judge_result.confidence_score, judge_result.is_correct)

            status = "✓" if judge_result.is_correct else "✗"
            print(f"{status} {judge_result.confidence_score}/10")

        except Exception as e:
            row = create_error_row(i, question, expected, eval_type, e)
            metrics.add_result(0, False)
            print(f"❌ {e}")

        results.append(row)

        # 中途儲存
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump([asdict(r) for r in results], f, indent=2, ensure_ascii=False)

    # 最終報告
    summary = metrics.get_summary()
    print(f"\n{'='*60}")
    print(
        f"準確率: {summary['accuracy']:.1%} | 平均信心: {summary['avg_confidence']:.1f}/10"
    )
    print(f"{'='*60}\n")

    # 儲存檔案
    pd.DataFrame([asdict(r) for r in results]).to_csv(
        output_dir / "eval_results.csv", index=False
    )
    with open(output_dir / "eval_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), **summary},
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"結果已儲存至 {output_dir}")


def main(agent, eval_set):
    asyncio.run(run_evaluation(agent, eval_set))


if __name__ == "__main__":
    main(root_agent, eval_set)
