"""
Eval V7: Multi-dimensional Evaluation with Structured Output Support
- Direct reading of Pydantic structured output
- SQL syntax validation
- LLM-as-Judge for faithfulness evaluation
- Three scoring dimensions: data_accuracy, faithfulness, sql_validity
"""

# %%
import json
import pathlib
import time
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
from text2sql.agents.planned_agent.agent_v7 import planned_agent_v7, Text2SQLOutput
from text2sql.agents.planned_agent.tools import check_sql_syntax
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types

# For LLM-as-Judge
from google.adk.models.lite_llm import LiteLlm

# MLflow Setup
mlflow.set_tracking_uri("http://localhost:5000")
experiment_name = "Text-to-SQL-Comparison-v7"
mlflow.set_experiment(experiment_name)

PROJECT_DIR = pathlib.Path(__file__).parent.parent.resolve()
DATA_DIR = PROJECT_DIR / "data"
ground_truth_set = DATA_DIR / "eval_set_v2.jsonl"

# Load Eval Set
with open(ground_truth_set, "r", encoding="utf-8") as f:
    eval_set = [json.loads(line) for line in f]

# Global Session Service
session_service = InMemorySessionService()
APP_NAME = "text2sql_eval_v7"
USER_ID = "eval_runner_v7"


# %% LLM-as-Judge Configuration

JUDGE_MODEL = LiteLlm(
    model="openai/qwen3-4b-instruct-2507",
    api_key="aaa",
    api_base="http://localhost:8081",
)


async def judge_faithfulness(
    question: str, raw_data: List[Any], answer: str
) -> Dict[str, Any]:
    """
    使用 LLM 判斷回答是否忠實反映數據。
    Returns:
        {
            "score": float (0.0-1.0),
            "reasoning": str,
            "issues": List[str]
        }
    """
    prompt = f"""你是一個評估專家。請判斷以下「回答」是否忠實反映了「原始數據」，並正確回答了「問題」。

**問題**: {question}

**原始數據**: {json.dumps(raw_data, ensure_ascii=False)}

**回答**: {answer}

請從以下三個維度評分（每個 0-1 分）：
1. **數據忠實度**: 回答中的數據是否與原始數據一致？沒有捏造或誤報？
2. **問題相關度**: 回答是否直接回應了問題？沒有答非所問？
3. **表達準確度**: 回答的表達是否清晰且沒有歧義？

請以 JSON 格式回覆：
{{
    "data_faithfulness": 0.0-1.0,
    "question_relevance": 0.0-1.0,
    "expression_accuracy": 0.0-1.0,
    "overall_score": 0.0-1.0,
    "reasoning": "簡短說明評分理由",
    "issues": ["問題1", "問題2"] 或 []
}}

只輸出 JSON，不要有其他文字。
"""

    try:
        # Use LiteLLM to call the judge model
        import litellm

        response = await asyncio.to_thread(
            litellm.completion,
            model="openai/qwen3-4b-instruct-2507",
            messages=[{"role": "user", "content": prompt}],
            api_key="aaa",
            api_base="http://localhost:8081",
        )

        result_text = response.choices[0].message.content

        # Try to extract JSON from the response
        # Handle potential markdown code blocks
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0]
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0]

        result = json.loads(result_text.strip())
        return result

    except Exception as e:
        logging.error(f"Judge error: {e}")
        return {
            "data_faithfulness": 0.5,
            "question_relevance": 0.5,
            "expression_accuracy": 0.5,
            "overall_score": 0.5,
            "reasoning": f"Judge error: {str(e)}",
            "issues": ["Judge evaluation failed"],
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


async def run_agent_once(agent, question: str) -> Dict[str, Any]:
    """
    執行 Agent 並返回結構化結果。
    """
    session_id = f"eval7_{agent.name}_{int(time.time()*1000)}_{uuid.uuid4().hex[:6]}"

    await session_service.create_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=session_id
    )

    runner = Runner(
        agent=agent,
        app_name=APP_NAME,
        session_service=session_service,
    )

    content = types.Content(role="user", parts=[types.Part(text=question)])

    result = {
        "final_text": "No response",
        "structured_output": None,
        "executed_sql": None,
        "sql_result": None,
        "raw_events": [],
    }

    try:
        async for event in runner.run_async(
            user_id=USER_ID, session_id=session_id, new_message=content
        ):
            # Record raw events for debugging
            event_info = {
                "author": getattr(event, "author", "unknown"),
                "is_final": (
                    event.is_final_response()
                    if hasattr(event, "is_final_response")
                    else False
                ),
            }

            # Capture tool results
            if event.content and event.content.parts:
                for part in event.content.parts:
                    # Capture function call (The Query)
                    if part.function_call and part.function_call.name == "execute_sql":
                        args = part.function_call.args
                        if args:
                            result["executed_sql"] = args.get("sql_query") or args.get(
                                "sql"
                            )

                    # Capture function response (The Result)
                    if (
                        part.function_response
                        and part.function_response.name == "execute_sql"
                    ):
                        resp = part.function_response.response
                        if isinstance(resp, dict) and "result" in resp:
                            result["sql_result"] = resp["result"]
                        else:
                            result["sql_result"] = str(resp)

            result["raw_events"].append(event_info)

            if event.is_final_response():
                if event.content and event.content.parts:
                    final_part = event.content.parts[0]

                    # Try to get structured output first
                    if hasattr(final_part, "text"):
                        result["final_text"] = final_part.text

                        # Try to parse as structured output
                        try:
                            parsed = json.loads(final_part.text)
                            if isinstance(parsed, dict):
                                result["structured_output"] = parsed
                                # Extract fields if they match Text2SQLOutput
                                if "executed_sql" in parsed:
                                    result["executed_sql"] = parsed["executed_sql"]
                                if "raw_results" in parsed:
                                    result["sql_result"] = parsed["raw_results"]
                        except json.JSONDecodeError:
                            pass

                elif event.actions and event.actions.escalate:
                    result["final_text"] = f"Escalated: {event.error_message}"
                break

    except Exception as e:
        result["final_text"] = f"Agent Error: {str(e)}"

    return result


def agent_inference(agent, question: str) -> Dict[str, Any]:
    return asyncio.run(run_agent_once(agent, question))


# %% Multi-dimensional Scoring


class EvalMetrics:
    """評估指標容器"""

    def __init__(self):
        self.data_accuracy_scores: List[float] = []
        self.sql_validity_scores: List[float] = []
        self.faithfulness_scores: List[float] = []
        self.question_relevance_scores: List[float] = []
        self.expression_accuracy_scores: List[float] = []

    def add_result(
        self,
        data_accuracy: float,
        sql_validity: float,
        faithfulness: float = 0.0,
        question_relevance: float = 0.0,
        expression_accuracy: float = 0.0,
    ):
        self.data_accuracy_scores.append(data_accuracy)
        self.sql_validity_scores.append(sql_validity)
        self.faithfulness_scores.append(faithfulness)
        self.question_relevance_scores.append(question_relevance)
        self.expression_accuracy_scores.append(expression_accuracy)

    def get_averages(self) -> Dict[str, float]:
        return {
            "data_accuracy": (
                np.mean(self.data_accuracy_scores) if self.data_accuracy_scores else 0.0
            ),
            "sql_validity": (
                np.mean(self.sql_validity_scores) if self.sql_validity_scores else 0.0
            ),
            "faithfulness": (
                np.mean(self.faithfulness_scores) if self.faithfulness_scores else 0.0
            ),
            "question_relevance": (
                np.mean(self.question_relevance_scores)
                if self.question_relevance_scores
                else 0.0
            ),
            "expression_accuracy": (
                np.mean(self.expression_accuracy_scores)
                if self.expression_accuracy_scores
                else 0.0
            ),
        }


# %% Evaluation Loop


async def run_comprehensive_evaluation(
    agents: List, eval_set: List[Dict], enable_judge: bool = True
):
    """
    執行完整的多維度評估。
    """
    run_name = f"comprehensive_v7_{int(time.time())}"

    # Output directory
    output_dir = PROJECT_DIR / "eval" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "eval7_comparison.json"

    with mlflow.start_run(run_name=run_name):
        results_matrix = []
        metrics_by_agent = {agent.name: EvalMetrics() for agent in agents}

        print(
            f"Starting V7 comprehensive evaluation for {len(agents)} agents on {len(eval_set)} questions..."
        )
        print(f"LLM-as-Judge: {'Enabled' if enable_judge else 'Disabled'}")

        agent_names = [a.name for a in agents]
        mlflow.log_param("agents", agent_names)
        mlflow.log_param("enable_judge", enable_judge)

        # Error categorization
        error_categories = {
            "region_alias_error": [],
            "sql_syntax_error": [],
            "empty_result_error": [],
            "dataframe_format_mismatch": [],
            "faithfulness_error": [],
            "other_error": [],
        }

        for i, entry in enumerate(eval_set):
            question = entry["question"]
            gt = entry["ground_truth"]
            e_type = entry["eval_type"]

            print(f"\n[{i+1}/{len(eval_set)}] Q: {question[:40]}...")

            row = {
                "id": i,
                "question": question,
                "expected": str(gt),
                "type": e_type,
            }

            for agent in agents:
                print(f"  Running {agent.name}...", end=" ")

                try:
                    # Run agent (use await since we're already in async context)
                    inference_result = await run_agent_once(agent, question)

                    actual_text = inference_result["final_text"]
                    sql_query = inference_result["executed_sql"]
                    sql_result = inference_result["sql_result"]
                    structured_output = inference_result["structured_output"]

                    # Parse sql_result as list if possible
                    actual_data = []
                    if sql_result:
                        try:
                            actual_data = ast.literal_eval(str(sql_result))
                        except:
                            actual_data = [sql_result]

                    # 1. SQL Validity Check
                    sql_validity = validate_sql_syntax(sql_query)
                    sql_score = 1.0 if sql_validity["is_valid"] else 0.0

                    # 2. Data Accuracy Check
                    data_accuracy = compare_data_accuracy(actual_data, gt, e_type)
                    data_score = 1.0 if data_accuracy["is_correct"] else 0.0

                    # 3. LLM-as-Judge (Faithfulness)
                    judge_result = {
                        "data_faithfulness": 0.5,
                        "question_relevance": 0.5,
                        "expression_accuracy": 0.5,
                        "overall_score": 0.5,
                        "reasoning": "Judge disabled",
                        "issues": [],
                    }

                    if enable_judge and data_score > 0:
                        # Only judge if data is correct
                        judge_result = await judge_faithfulness(
                            question, actual_data, actual_text
                        )

                    # Record metrics
                    metrics_by_agent[agent.name].add_result(
                        data_accuracy=data_score,
                        sql_validity=sql_score,
                        faithfulness=judge_result.get("data_faithfulness", 0.5),
                        question_relevance=judge_result.get("question_relevance", 0.5),
                        expression_accuracy=judge_result.get(
                            "expression_accuracy", 0.5
                        ),
                    )

                    # Overall correctness (weighted)
                    is_correct = data_score > 0.5

                    # Store results
                    row[f"{agent.name}_actual"] = (
                        actual_text[:500] if actual_text else ""
                    )
                    row[f"{agent.name}_sql_query"] = sql_query
                    row[f"{agent.name}_sql_result"] = (
                        str(sql_result)[:500] if sql_result else ""
                    )
                    row[f"{agent.name}_data_correct"] = data_accuracy["is_correct"]
                    row[f"{agent.name}_data_details"] = data_accuracy["details"]
                    row[f"{agent.name}_sql_valid"] = sql_validity["is_valid"]
                    row[f"{agent.name}_sql_error"] = sql_validity["error"]
                    row[f"{agent.name}_judge_score"] = judge_result.get(
                        "overall_score", 0.5
                    )
                    row[f"{agent.name}_judge_reasoning"] = judge_result.get(
                        "reasoning", ""
                    )
                    row[f"{agent.name}_judge_issues"] = str(
                        judge_result.get("issues", [])
                    )

                    print(
                        f"Data: {'✓' if data_accuracy['is_correct'] else '✗'}, SQL: {'✓' if sql_validity['is_valid'] else '✗'}, Judge: {judge_result.get('overall_score', 0.5):.2f}"
                    )

                    # Categorize errors
                    if not is_correct:
                        if not sql_validity["is_valid"]:
                            error_categories["sql_syntax_error"].append(i)
                        elif sql_result == "[]" or sql_result == "[(None,)]":
                            error_categories["empty_result_error"].append(i)
                        elif "EMEA" in question or "歐非中東" in question:
                            error_categories["region_alias_error"].append(i)
                        elif e_type == "dataframe":
                            error_categories["dataframe_format_mismatch"].append(i)
                        elif judge_result.get("overall_score", 1.0) < 0.5:
                            error_categories["faithfulness_error"].append(i)
                        else:
                            error_categories["other_error"].append(i)

                except Exception as e:
                    print(f"Error: {e}")
                    row[f"{agent.name}_actual"] = f"Error: {e}"
                    row[f"{agent.name}_sql_query"] = "Error"
                    row[f"{agent.name}_sql_result"] = "Error"
                    row[f"{agent.name}_data_correct"] = False
                    row[f"{agent.name}_data_details"] = str(e)
                    row[f"{agent.name}_sql_valid"] = False
                    row[f"{agent.name}_sql_error"] = str(e)
                    row[f"{agent.name}_judge_score"] = 0.0
                    row[f"{agent.name}_judge_reasoning"] = f"Error: {e}"
                    row[f"{agent.name}_judge_issues"] = "[]"

                    metrics_by_agent[agent.name].add_result(0.0, 0.0, 0.0, 0.0, 0.0)
                    error_categories["other_error"].append(i)

            results_matrix.append(row)

            # Dynamic output
            try:
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(results_matrix, f, indent=4, ensure_ascii=False)
            except Exception as e:
                print(f"Warning: Failed to write intermediate JSON: {e}")

        print("\n" + "=" * 60)
        print("Evaluation Complete!")
        print("=" * 60)

        # Log metrics per agent
        for agent_name, metrics in metrics_by_agent.items():
            averages = metrics.get_averages()
            print(f"\n{agent_name}:")
            for metric_name, value in averages.items():
                print(f"  {metric_name}: {value:.2%}")
                mlflow.log_metric(f"{agent_name}_{metric_name}", value)

        # Log error categories
        print("\n=== Error Categories ===")
        for category, ids in error_categories.items():
            if ids:
                unique_ids = list(set(ids))
                print(f"{category}: {len(unique_ids)} errors")
                mlflow.log_metric(f"error_{category}_count", len(unique_ids))

        # Save results
        df = pd.DataFrame(results_matrix)

        csv_path = output_dir / "eval7_comparison.csv"
        df.to_csv(csv_path, index=False)
        mlflow.log_artifact(str(csv_path))

        html_path = output_dir / "eval7_comparison.html"
        df.to_html(html_path)
        mlflow.log_artifact(str(html_path))

        df.to_json(json_path, orient="records", indent=4, force_ascii=False)
        mlflow.log_artifact(str(json_path))

        # Summary report
        summary_path = output_dir / "eval7_summary.json"
        summary = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_questions": len(eval_set),
            "agents": {},
            "error_categories": {k: len(set(v)) for k, v in error_categories.items()},
        }
        for agent_name, metrics in metrics_by_agent.items():
            summary["agents"][agent_name] = metrics.get_averages()

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4, ensure_ascii=False)
        mlflow.log_artifact(str(summary_path))

        print(f"\nSaved results to {output_dir}")


def run_evaluation(agents, eval_set, enable_judge=True):
    asyncio.run(run_comprehensive_evaluation(agents, eval_set, enable_judge))


if __name__ == "__main__":
    agents_to_eval = [planned_agent_v7]

    # Set enable_judge=False for faster evaluation without LLM-as-Judge
    run_evaluation(agents_to_eval, eval_set, enable_judge=True)
