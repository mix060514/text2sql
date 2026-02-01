# %%
import json
import pathlib
from datetime import datetime
import time
import random
import numpy as np
import mlflow
import pandas as pd
import sys
import os
import asyncio
import logging
import uuid
from typing import Dict, Any, List
import warnings

warnings.filterwarnings("ignore")

# MLflow Setup
mlflow.set_tracking_uri("http://localhost:5000")
experiment_name = "Text-to-SQL-Eval"
mlflow.set_experiment(experiment_name)


from google.adk.sessions import InMemorySessionService
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.genai import types

# Setup path to import local src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../eval")))

# Import Agents
from text2sql.agents.planned_agent.agent import root_agent
from judge.agent import judge_agent


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
session_service = InMemorySessionService()
APP_NAME = "text2sql_eval"
USER_ID = "eval_runner"


# %% LLM-as-Judge Configuration (Using Judge Agent)

# Judge Agent Session Service
judge_session_service = InMemorySessionService()
JUDGE_APP_NAME = "judge_eval"
JUDGE_USER_ID = "judge_runner"


async def judge_confidence(
    question: str,
    expected_answer: Any,
    actual_answer: str,
    sql_query: str = None,
    sql_result: Any = None,
) -> Dict[str, Any]:
    """
    使用 Judge Agent 評估回答正確性的信心分數 (0-10)。
    Returns:
        {
            "confidence_score": int (0-10),
            "reasoning": str,
            "is_correct": bool,
            "issues": List[str]
        }
    """
    # 構造評估請求
    eval_prompt = f"""請評估以下 Text-to-SQL Agent 的回答品質：

**使用者問題**: {question}

**預期正確答案**: {json.dumps(expected_answer, ensure_ascii=False)}

**SQL 查詢**: {sql_query if sql_query else "無法取得"}

**SQL 結果**: {json.dumps(sql_result, ensure_ascii=False) if sql_result else "無法取得"}

**Agent 回答**: {actual_answer}
"""

    try:
        judge_session_id = f"judge_{int(time.time()*1000)}_{uuid.uuid4().hex[:6]}"
        await judge_session_service.create_session(
            app_name=JUDGE_APP_NAME, user_id=JUDGE_USER_ID, session_id=judge_session_id
        )

        runner = Runner(
            agent=judge_agent,
            app_name=JUDGE_APP_NAME,
            session_service=judge_session_service,
        )

        content = types.Content(role="user", parts=[types.Part(text=eval_prompt)])

        result_text = ""

        async for event in runner.run_async(
            user_id=JUDGE_USER_ID, session_id=judge_session_id, new_message=content
        ):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if hasattr(part, "text") and part.text:
                        result_text += part.text

        # 從 session state 取得 structured output
        session = await judge_session_service.get_session(
            app_name=JUDGE_APP_NAME, user_id=JUDGE_USER_ID, session_id=judge_session_id
        )
        structured_output = (
            session.state.get("judge_result") if session and session.state else None
        )

        # 驗證 structured output 是否完整
        structured_output_success = False
        if structured_output:
            required_keys = ["confidence_score", "is_correct", "reasoning"]
            missing_keys = [k for k in required_keys if k not in structured_output]

            if missing_keys:
                # 缺少欄位 = 失敗
                score = 0
                is_correct = False
                reasoning = f"Structured output 缺少欄位: {missing_keys}"
            else:
                structured_output_success = True
                score = structured_output["confidence_score"]
                is_correct = structured_output["is_correct"]
                reasoning = structured_output["reasoning"]
        else:
            score = 0
            is_correct = False
            reasoning = f"無法解析 Judge Agent 輸出: {result_text[:100]}"

        score = max(0, min(10, int(score)))

        return {
            "confidence_score": score,
            "is_correct": is_correct,
            "reasoning": reasoning,
            "structured_output_success": structured_output_success,
        }

    except Exception as e:
        logging.error(f"Judge error: {e}")
        return {
            "confidence_score": 0,
            "is_correct": False,
            "reasoning": f"Judge error: {str(e)}",
        }


# %% Async Agent Wrapper

# AGENT_TIMEOUT_SECONDS = 120  # 單個問題的超時時間
AGENT_TIMEOUT_SECONDS = 300  # 單個問題的超時時間


async def _run_agent_impl(agent, question: str, session_id: str) -> Dict[str, Any]:
    """
    Expert V8 Implementation:
    從 Event Stream 中攔截 Sub-agent 的 SQL、Result 與 Answer，
    無視 Root Agent 的包裝干擾。
    """
    await session_service.create_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=session_id
    )

    runner = Runner(
        agent=agent,
        app_name=APP_NAME,
        session_service=session_service,
    )

    content = types.Content(role="user", parts=[types.Part(text=question)])

    # 1. 初始化狀態容器
    result_context = {
        "final_text": "",  # 這是給 User 看的最終回答 (Answer Agent)
        "executed_sql": None,  # 這是給 Eval 比對的 SQL (SQL Gen Agent)
        "sql_result": None,  # 這是給 Eval 比對的 Data (Execute SQL Agent)
        "raw_events": [],
    }

    # 用來追蹤當前的對話狀態
    candidate_answer = ""

    # 2. 啟動事件流監聽
    async for event in runner.run_async(
        user_id=USER_ID, session_id=session_id, new_message=content
    ):
        # 紀錄原始事件以便 Debug
        result_context["raw_events"].append(str(event))

        if event.content and event.content.parts:
            for part in event.content.parts:

                # --- A. 攔截 SQL (當 Tool 被呼叫時) ---
                if part.function_call and "execute_sql" in part.function_call.name:
                    args = part.function_call.args
                    # 支援多種參數命名慣例，防止 Prompt 飄移
                    sql = args.get("sql_query") or args.get("sql") or args.get("query")
                    if sql:
                        result_context["executed_sql"] = sql

                # --- B. 攔截 Result (當 Tool 回傳結果時) ---
                if (
                    part.function_response
                    and "execute_sql" in part.function_response.name
                ):
                    resp = part.function_response.response
                    # 強壯的解析邏輯：處理 ADK 包裝的 Dict 或原始 List
                    if isinstance(resp, dict):
                        # 優先拿 'result' 或 'content'，如果都沒有就拿整個 dict
                        result_context["sql_result"] = (
                            resp.get("result") or resp.get("content") or resp
                        )
                    else:
                        result_context["sql_result"] = resp

                # --- C. 攔截 Answer (Answer Agent 的發言) ---
                # 邏輯：
                # 1. 必須是純文字 (has text)
                # 2. 不能是 SQL 代碼塊 (過濾掉 SQL Gen Agent 的思考)
                # 3. 必須是在我們拿到 SQL Result 之後產生的 (過濾掉開場白)
                if hasattr(part, "text") and part.text:
                    text = part.text.strip()

                    # 過濾條件：不要 SQL，不要空白
                    if (
                        text
                        and "```sql" not in text
                        and "SELECT" not in text[:20].upper()
                    ):
                        # 更新候選答案。
                        # 因為 Answer Agent 是在 SQL 執行完後才發言，
                        # 所以它會覆蓋掉前面的對話，成為最後的 candidate_answer
                        candidate_answer = text

    # 3. 最終組裝
    # 如果我們有抓到候選答案（Answer Agent 的發言），就優先使用它
    # 這樣可以避開 Root Agent 最後可能加上的 "I have delegated..." 廢話
    if candidate_answer:
        result_context["final_text"] = candidate_answer
    else:
        # 萬一真的沒抓到，只好回傳空字串或錯誤提示
        result_context["final_text"] = "No answer generated."

    # 4. 針對 Eval 的最後防線：如果 SQL 沒抓到，標記為 Error
    if not result_context["executed_sql"]:
        result_context["executed_sql"] = "Error: No SQL executed"

    return result_context


async def run_agent_once(agent, question: str) -> Dict[str, Any]:
    """
    執行 Agent 並返回結構化結果，帶有超時機制。
    """
    session_id = f"eval8_{agent.name}_{int(time.time()*1000)}_{uuid.uuid4().hex[:6]}"

    try:
        # 使用 asyncio.wait_for 加入超時
        result = await asyncio.wait_for(
            _run_agent_impl(agent, question, session_id), timeout=AGENT_TIMEOUT_SECONDS
        )
        return result
    except asyncio.TimeoutError:
        return {
            "final_text": f"Timeout after {AGENT_TIMEOUT_SECONDS}s",
            "structured_output": None,
            "executed_sql": None,
            "sql_result": None,
            "raw_events": [],
        }
    except Exception as e:
        return {
            "final_text": f"Agent Error: {str(e)}",
            "structured_output": None,
            "executed_sql": None,
            "sql_result": None,
            "raw_events": [],
        }


def agent_inference(agent, question: str) -> Dict[str, Any]:
    return asyncio.run(run_agent_once(agent, question))


# %% Simplified Evaluation Metrics (0-10 Confidence Score)


class EvalMetrics:
    """評估指標容器 - 簡化版，專注於信心分數"""

    def __init__(self):
        self.confidence_scores: List[int] = []
        self.correct_count: int = 0
        self.total_count: int = 0

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


# %% Evaluation Loop


async def run_one_question(
    i, agent, question, expected, eval_type, results, metrics_by_agent
):
    print(f"  → {agent.name}: ", end="", flush=True)

    try:
        # Run agent
        result = await run_agent_once(agent, question)

        actual_answer = result["final_text"]
        sql_query = result["executed_sql"]
        sql_result = result["sql_result"]

        # LLM-as-Judge evaluation
        judge_result = await judge_confidence(
            question=question,
            expected_answer=expected,
            actual_answer=actual_answer,
            sql_query=sql_query,
            sql_result=sql_result,
        )

        confidence = judge_result["confidence_score"]
        is_correct = judge_result["is_correct"]
        reasoning = judge_result.get("reasoning", "")

        # Record metrics
        metrics_by_agent.add_result(confidence, is_correct)

        # Store result with clean format
        row = {
            "id": i,
            "question": question,
            "expected": expected,
            "eval_type": eval_type,
            "final_answer": actual_answer or "",
            "sql_query": sql_query or "",
            "execute_result": sql_result if sql_result else "",
            "llm_judge_score": confidence,
            "llm_judge_reasoning": reasoning,
        }
        results.append(row)

        # Print result
        status = "✓" if is_correct else "✗"
        print(f"{status} Confidence: {confidence}/10 - {reasoning[:50]}...")

    except Exception as e:
        print(f"❌ Error: {e}")
        row = {
            "id": i,
            "question": question,
            "expected": expected,
            "eval_type": eval_type,
            "final_answer": f"Error: {e}",
            "sql_query": "",
            "execute_result": "",
            "llm_judge_score": 0,
            "llm_judge_reasoning": str(e),
        }
        results.append(row)
        metrics_by_agent.add_result(0, False)


async def run_evaluation(agent: Agent, eval_set: List[Dict]):
    """
    執行 V8 評估 - 使用 LLM-as-Judge 給出 0-10 信心分數
    """
    run_name = f"eval_{int(time.time())}"

    # Output directory
    output_dir = (
        PROJECT_DIR / "eval" / "results" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "eval_results.json"

    with mlflow.start_run(run_name=run_name):
        results = []
        metrics_by_agent = EvalMetrics()

        print(f"\n{'='*60}")
        print(f"Text-to-SQL Eval V8 (LLM-as-Judge, 0-10 Confidence)")
        print(f"Agent: {agent.name}")
        print(f"Questions: {len(eval_set)}")
        print(f"{'='*60}\n")

        # experimen params
        mlflow.log_param("agent", agent.name)
        mlflow.log_param("num_questions", len(eval_set))

        for i, entry in enumerate(eval_set):
            question = entry["question"]
            expected = entry["ground_truth"]
            eval_type = entry.get("eval_type", "unknown")

            print(f"[{i+1}/{len(eval_set)}] {question[:50]}...")

            await run_one_question(
                i, agent, question, expected, eval_type, results, metrics_by_agent
            )

            # Save intermediate results
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

        # Final summary
        print(f"\n{'='*60}")
        print("Evaluation Complete!")
        print(f"{'='*60}\n")

        summary = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_questions": len(eval_set),
            "agents": {},
        }

        agent_summary = metrics_by_agent.get_summary()
        summary["agents"] = agent_summary

        print(f"{agent.name}:")
        print(
            f"  準確率: {agent_summary['accuracy']:.1%} ({agent_summary['correct_count']}/{agent_summary['total_count']})"
        )
        print(f"  平均信心分數: {agent_summary['avg_confidence']:.1f}/10")

        mlflow.log_metric(f"accuracy", agent_summary["accuracy"])
        mlflow.log_metric(f"avg_confidence", agent_summary["avg_confidence"])

        # Save final results
        df = pd.DataFrame(results)

        csv_path = output_dir / "eval_results.csv"
        df.to_csv(csv_path, index=False)
        mlflow.log_artifact(str(csv_path))

        summary_path = output_dir / "eval_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        mlflow.log_artifact(str(summary_path))

        print(f"\n結果已儲存至 {output_dir}")


def main(agent, eval_set):
    """入口函數"""
    asyncio.run(run_evaluation(agent, eval_set))


if __name__ == "__main__":
    main(root_agent, eval_set)
