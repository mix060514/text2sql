"""
Eval V8: LLM-as-Judge Evaluation with Confidence Scoring (0-10)
- Works with SequentialAgent + LoopAgent architecture
- LLM-as-Judge for holistic evaluation
- Confidence scoring on 0-10 scale
- Extracts SQL and results from event stream
"""

# %%
import json
import pathlib
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
import re
import ast
from typing import Optional, Dict, Any, List

# Ignore warnings
import warnings

warnings.filterwarnings("ignore")

# Configure Logging
logging.basicConfig(level=logging.ERROR)

# Setup path to import local src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

# Import Agents
from text2sql.agents.planned_agent.agent import root_agent
from text2sql.agents.planned_agent.tools import check_sql_syntax
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types

# For LLM-as-Judge
from google.adk.models.lite_llm import LiteLlm

# MLflow Setup
mlflow.set_tracking_uri("http://localhost:5000")
experiment_name = "Text-to-SQL-Eval-v8"
mlflow.set_experiment(experiment_name)

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
APP_NAME = "text2sql_eval_v7"
USER_ID = "eval_runner_v7"


# %% LLM-as-Judge Configuration (0-10 Confidence Score)


async def judge_confidence(
    question: str,
    expected_answer: Any,
    actual_answer: str,
    sql_query: str = None,
    sql_result: Any = None,
) -> Dict[str, Any]:
    """
    使用 LLM 評估回答的信心分數 (0-10)。

    評估維度：
    1. 答案正確性：回答是否與預期答案一致
    2. 數據準確性：數據是否正確無誤
    3. 回答完整性：是否完整回答問題

    Returns:
        {
            "confidence_score": int (0-10),
            "reasoning": str,
            "is_correct": bool,
            "issues": List[str]
        }
    """
    prompt = f"""你是一個專業的 Text-to-SQL 系統評估專家。請評估以下回答的品質。

**使用者問題**: {question}

**預期正確答案**: {json.dumps(expected_answer, ensure_ascii=False)}

**SQL 查詢**: {sql_query if sql_query else "無法取得"}

**SQL 結果**: {json.dumps(sql_result, ensure_ascii=False) if sql_result else "無法取得"}

**Agent 回答**: {actual_answer}

請根據以下標準給出 0-10 的信心分數：
- 10分：完全正確，數據精確匹配
- 8-9分：基本正確，有微小格式或表達差異
- 6-7分：大致正確，但有部分數據偏差
- 4-5分：部分正確，有明顯錯誤
- 2-3分：大部分錯誤
- 0-1分：完全錯誤或無法回答

請以 JSON 格式回覆（不要有其他文字）：
{{
    "confidence_score": 0-10 的整數,
    "is_correct": true/false (分數>=7視為正確),
    "reasoning": "簡短說明評分理由（一句話）",
    "issues": ["問題1", "問題2"] 或空陣列
}}
"""

    try:
        import litellm

        response = await asyncio.to_thread(
            litellm.completion,
            model="openai/qwen3-4b-instruct-2507",
            messages=[{"role": "user", "content": prompt}],
            api_key="aaa",
            api_base="http://localhost:8081",
        )

        result_text = response.choices[0].message.content

        # Extract JSON from response (handle markdown code blocks)
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0]
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0]

        result = json.loads(result_text.strip())

        # Ensure confidence_score is int and in range
        score = int(result.get("confidence_score", 5))
        score = max(0, min(10, score))
        result["confidence_score"] = score
        result["is_correct"] = score >= 7

        return result

    except Exception as e:
        logging.error(f"Judge error: {e}")
        return {
            "confidence_score": 5,
            "is_correct": False,
            "reasoning": f"Judge error: {str(e)}",
            "issues": ["LLM Judge 評估失敗"],
        }


# %% Helper Functions


def extract_numbers(text):
    matches = re.findall(r"-?\d{1,3}(?:,\d{3})*(?:\.\d+)?|-?\d+(?:\.\d+)?", str(text))
    cleaned = []
    for m in matches:
        try:
            val = float(m.replace(",", ""))
            cleaned.append(val)
        except:
            pass
    return cleaned


def normalize_string(s):
    """Remove punctuation and lowercase."""
    return re.sub(r"[^\w\s]", "", str(s).lower())


def normalize_month_key(key):
    """Normalize month keys: '2023-1' and '2023-01' should be treated as equal."""
    if isinstance(key, str) and "-" in key:
        parts = key.split("-")
        if len(parts) == 2:
            try:
                year, month = parts
                return f"{int(year)}-{int(month)}"
            except:
                pass
    return str(key)


def validate_sql_syntax(sql_query: str) -> Dict[str, Any]:
    """
    驗證 SQL 語法並返回詳細結果。
    """
    if not sql_query or sql_query == "Error" or sql_query == "None":
        return {"is_valid": False, "error": "No SQL query provided"}

    result = check_sql_syntax(sql_query)
    if result == "Valid":
        return {"is_valid": True, "error": None}
    else:
        return {"is_valid": False, "error": result}


def compare_data_accuracy(
    actual_data: List[Any], expected: Any, eval_type: str, eps: float = 1e-2
) -> Dict[str, Any]:
    """
    比對數據準確性，返回詳細結果。
    """
    result = {
        "is_correct": False,
        "match_type": "none",
        "details": "",
    }

    try:
        if eval_type == "number":
            expected_val = float(expected)
            for item in actual_data:
                if isinstance(item, (list, tuple)):
                    for sub_item in item:
                        if isinstance(sub_item, (int, float)):
                            if abs(sub_item - expected_val) < eps:
                                result["is_correct"] = True
                                result["match_type"] = "exact_number"
                                result["details"] = (
                                    f"Found {sub_item}, expected {expected_val}"
                                )
                                return result
                elif isinstance(item, (int, float)):
                    if abs(item - expected_val) < eps:
                        result["is_correct"] = True
                        result["match_type"] = "exact_number"
                        result["details"] = f"Found {item}, expected {expected_val}"
                        return result

            result["details"] = (
                f"No matching number found. Expected {expected_val}, got {actual_data}"
            )
            return result

        elif eval_type == "string":
            norm_expected = normalize_string(expected)
            norm_actual = normalize_string(str(actual_data))

            if norm_expected in norm_actual:
                result["is_correct"] = True
                result["match_type"] = "string_contains"
                result["details"] = f"Found '{expected}' in result"
            else:
                result["details"] = f"'{expected}' not found in '{actual_data}'"
            return result

        elif eval_type == "list":
            if not isinstance(expected, (list, tuple)):
                result["details"] = "Expected is not a list"
                return result

            norm_actual = normalize_string(str(actual_data))
            all_found = True
            missing = []

            for item in expected:
                norm_item = normalize_string(item)
                if norm_item not in norm_actual:
                    all_found = False
                    missing.append(item)

            if all_found:
                result["is_correct"] = True
                result["match_type"] = "list_complete"
                result["details"] = f"All {len(expected)} items found"
            else:
                result["details"] = f"Missing items: {missing}"
            return result

        elif eval_type == "dataframe":
            expected_dict = (
                ast.literal_eval(str(expected))
                if isinstance(expected, str)
                else expected
            )
            if not isinstance(expected_dict, dict):
                result["details"] = "Expected is not a dict"
                return result

            # Normalize expected dict keys
            normalized_expected = {
                normalize_month_key(k): v for k, v in expected_dict.items()
            }

            # Extract key-value pairs from actual data
            result_dict = {}
            if isinstance(actual_data, list):
                for row in actual_data:
                    if isinstance(row, (list, tuple)) and len(row) >= 2:
                        key = normalize_month_key(row[0])
                        value = row[1]
                        result_dict[key] = value

            # Compare
            all_match = True
            mismatches = []

            for exp_key, exp_val in normalized_expected.items():
                if exp_key not in result_dict:
                    all_match = False
                    mismatches.append(f"Missing key: {exp_key}")
                    continue

                result_val = result_dict[exp_key]
                if isinstance(exp_val, (int, float)) and isinstance(
                    result_val, (int, float)
                ):
                    if abs(float(exp_val) - float(result_val)) >= eps:
                        all_match = False
                        mismatches.append(
                            f"{exp_key}: expected {exp_val}, got {result_val}"
                        )
                else:
                    if normalize_string(exp_val) != normalize_string(result_val):
                        all_match = False
                        mismatches.append(
                            f"{exp_key}: expected {exp_val}, got {result_val}"
                        )

            if all_match:
                result["is_correct"] = True
                result["match_type"] = "dataframe_complete"
                result["details"] = f"All {len(normalized_expected)} entries match"
            else:
                result["details"] = "; ".join(mismatches)
            return result

    except Exception as e:
        result["details"] = f"Comparison error: {str(e)}"
        return result


# %% Async Agent Wrapper

AGENT_TIMEOUT_SECONDS = 120  # 單個問題的超時時間


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


async def run_evaluation_v8(agents: List, eval_set: List[Dict]):
    """
    執行 V8 評估 - 使用 LLM-as-Judge 給出 0-10 信心分數
    """
    run_name = f"eval_v8_{int(time.time())}"

    # Output directory
    output_dir = PROJECT_DIR / "eval" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "eval8_results.json"

    with mlflow.start_run(run_name=run_name):
        results = []
        metrics_by_agent = {agent.name: EvalMetrics() for agent in agents}

        print(f"\n{'='*60}")
        print(f"Text-to-SQL Eval V8 (LLM-as-Judge, 0-10 Confidence)")
        print(f"Agents: {[a.name for a in agents]}")
        print(f"Questions: {len(eval_set)}")
        print(f"{'='*60}\n")

        mlflow.log_param("agents", [a.name for a in agents])
        mlflow.log_param("num_questions", len(eval_set))

        for i, entry in enumerate(eval_set):
            question = entry["question"]
            expected = entry["ground_truth"]
            eval_type = entry.get("eval_type", "unknown")

            print(f"[{i+1}/{len(eval_set)}] {question[:50]}...")

            for agent in agents:
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
                    metrics_by_agent[agent.name].add_result(confidence, is_correct)

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
                    metrics_by_agent[agent.name].add_result(0, False)

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

        for agent_name, metrics in metrics_by_agent.items():
            agent_summary = metrics.get_summary()
            summary["agents"][agent_name] = agent_summary

            print(f"{agent_name}:")
            print(
                f"  準確率: {agent_summary['accuracy']:.1%} ({agent_summary['correct_count']}/{agent_summary['total_count']})"
            )
            print(f"  平均信心分數: {agent_summary['avg_confidence']:.1f}/10")

            mlflow.log_metric(f"{agent_name}_accuracy", agent_summary["accuracy"])
            mlflow.log_metric(
                f"{agent_name}_avg_confidence", agent_summary["avg_confidence"]
            )

        # Save final results
        df = pd.DataFrame(results)

        csv_path = output_dir / "eval8_results.csv"
        df.to_csv(csv_path, index=False)
        mlflow.log_artifact(str(csv_path))

        summary_path = output_dir / "eval8_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        mlflow.log_artifact(str(summary_path))

        print(f"\n結果已儲存至 {output_dir}")


def run_evaluation(agents, eval_set):
    """入口函數"""
    asyncio.run(run_evaluation_v8(agents, eval_set))


if __name__ == "__main__":
    agents_to_eval = [root_agent]
    run_evaluation(agents_to_eval, eval_set)
