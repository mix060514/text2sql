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

# Ignore warnings
import warnings

warnings.filterwarnings("ignore")

# Configure Logging
logging.basicConfig(level=logging.ERROR)

# Setup path to import local src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

# Import Agents
from text2sql.agents.planned_agent.agent_v5 import planned_agent_v5
from text2sql.agents.planned_agent.agent_v6 import planned_agent_v6
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types


# MLflow Setup
mlflow.set_tracking_uri("http://localhost:5000")
experiment_name = "Text-to-SQL-Comparison-v6"
mlflow.set_experiment(experiment_name)

PROJECT_DIR = pathlib.Path(__file__).parent.parent.resolve()
DATA_DIR = PROJECT_DIR / "data"
# Use v2 ground truth set (with fixed month format)
ground_truth_set = DATA_DIR / "eval_set_v2.jsonl"

# Load Eval Set
with open(ground_truth_set, "r", encoding="utf-8") as f:
    eval_set = [json.loads(line) for line in f]

# Global Session Service
session_service = InMemorySessionService()
APP_NAME = "text2sql_eval_v6"
USER_ID = "eval_runner_v6"

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
                return f"{int(year)}-{int(month)}"  # Remove leading zeros
            except:
                pass
    return str(key)


def compare_answers_smart(actual_text, expected, eval_type, eps=1e-2, sql_result=None):
    """
    Improved comparison logic with:
    - Better dataframe month key normalization
    - Structured JSON parsing support
    - Floating point tolerance
    """
    try:
        # Priority 1: Use sql_result if available as it is more structured
        parsed_result = None
        if sql_result:
            try:
                parsed_result = ast.literal_eval(sql_result)
            except:
                pass

        if eval_type == "number":
            # Extract numbers from both structure and text
            candidates = []

            if parsed_result:

                def extract_from_struct(data):
                    if isinstance(data, (list, tuple)):
                        for item in data:
                            extract_from_struct(item)
                    elif isinstance(data, (int, float)):
                        candidates.append(data)
                    elif isinstance(data, str):
                        try:
                            candidates.append(float(data))
                        except:
                            pass

                extract_from_struct(parsed_result)

            # Fallback to text extraction
            candidates.extend(extract_numbers(actual_text))

            expected_val = float(expected)
            for n in candidates:
                if abs(n - expected_val) < eps:
                    return True
            return False

        elif eval_type == "string":
            # For string, we check if expected is in actual
            norm_expected = normalize_string(expected)
            norm_actual = normalize_string(actual_text)

            if norm_expected in norm_actual:
                return True

            # Also check sql_result content
            if parsed_result:
                norm_sql = normalize_string(str(parsed_result))
                if norm_expected in norm_sql:
                    return True
            return False

        elif eval_type == "list":
            if not isinstance(expected, (list, tuple)):
                return False

            # Check if all expected items are present
            norm_actual = normalize_string(actual_text)
            norm_sql = normalize_string(str(parsed_result)) if parsed_result else ""

            all_found = True
            for item in expected:
                norm_item = normalize_string(item)
                if (norm_item not in norm_actual) and (norm_item not in norm_sql):
                    all_found = False
                    break
            return all_found

        elif eval_type == "dataframe":
            # Expected is typically a dict: {'2023-1': value, '2023-2': value, ...}
            try:
                expected_dict = (
                    ast.literal_eval(str(expected))
                    if isinstance(expected, str)
                    else expected
                )
                if not isinstance(expected_dict, dict):
                    return False
            except:
                return False

            # Normalize expected dict keys (remove leading zeros from months)
            normalized_expected = {
                normalize_month_key(k): v for k, v in expected_dict.items()
            }

            # Extract key-value pairs from SQL result
            result_dict = {}
            if parsed_result and isinstance(parsed_result, list):
                for row in parsed_result:
                    if isinstance(row, (list, tuple)) and len(row) >= 2:
                        # Assume format: [(key, value), ...]
                        key = normalize_month_key(row[0])
                        value = row[1]
                        result_dict[key] = value

            # Compare all expected key-value pairs
            all_match = True
            for exp_key, exp_val in normalized_expected.items():
                if exp_key not in result_dict:
                    all_match = False
                    break

                result_val = result_dict[exp_key]

                # Compare values with tolerance for floats
                if isinstance(exp_val, (int, float)) and isinstance(
                    result_val, (int, float)
                ):
                    if abs(float(exp_val) - float(result_val)) >= eps:
                        all_match = False
                        break
                else:
                    # String comparison
                    if normalize_string(exp_val) != normalize_string(result_val):
                        all_match = False
                        break

            return all_match

    except Exception as e:
        # print(f"Comparison Error: {e}")
        return False
    return False


# %% Async Agent Wrapper


async def run_agent_once(agent, question):
    session_id = f"eval6_{agent.name}_{int(time.time()*1000)}_{uuid.uuid4().hex[:6]}"

    await session_service.create_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=session_id
    )

    runner = Runner(
        agent=agent,
        app_name=APP_NAME,
        session_service=session_service,
    )

    content = types.Content(role="user", parts=[types.Part(text=question)])
    final_text = "No response"
    executed_sql_result = None
    executed_sql_query = None

    try:
        async for event in runner.run_async(
            user_id=USER_ID, session_id=session_id, new_message=content
        ):
            # Capture tool results
            if event.content and event.content.parts:
                for part in event.content.parts:
                    # Capture function call (The Query)
                    if part.function_call and part.function_call.name == "execute_sql":
                        args = part.function_call.args
                        if args:
                            # The tool arg is 'sql_query' but LLM might produce 'sql'
                            executed_sql_query = args.get("sql_query") or args.get(
                                "sql"
                            )

                    # Capture function response (The Result)
                    if (
                        part.function_response
                        and part.function_response.name == "execute_sql"
                    ):
                        resp = part.function_response.response
                        if isinstance(resp, dict) and "result" in resp:
                            executed_sql_result = resp["result"]
                        else:
                            executed_sql_result = str(resp)

            if event.is_final_response():
                if event.content and event.content.parts:
                    final_text = event.content.parts[0].text
                elif event.actions and event.actions.escalate:
                    final_text = f"Escalated: {event.error_message}"
                break
    except Exception as e:
        final_text = f"Agent Error: {str(e)}"

    return final_text, executed_sql_result, executed_sql_query


def agent_inference(agent, question):
    return asyncio.run(run_agent_once(agent, question))


# %% Evaluation Loop


def run_comparative_evaluation(agents, eval_set):
    run_name = f"comparison_v6_{int(time.time())}"

    # Output directory
    output_dir = PROJECT_DIR / "eval" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "eval6_comparison.json"

    with mlflow.start_run(run_name=run_name):
        results_matrix = []

        print(
            f"Starting comparison for {len(agents)} agents on {len(eval_set)} questions..."
        )

        agent_names = [a.name for a in agents]
        mlflow.log_param("agents", agent_names)

        total_scores = {name: 0 for name in agent_names}

        # Error categorization
        error_categories = {
            "region_alias_error": [],
            "sql_syntax_error": [],
            "empty_result_error": [],
            "dataframe_format_mismatch": [],
            "ground_truth_error": [],
            "other_error": [],
        }

        for i, entry in enumerate(eval_set):
            question = entry["question"]
            gt = entry["ground_truth"]
            e_type = entry["eval_type"]

            print(f"[{i+1}/{len(eval_set)}] Q: {question[:30]}...", end="\r")

            row = {"id": i, "question": question, "expected": str(gt), "type": e_type}

            problematic = False

            for agent in agents:
                try:
                    actual, sql_res, sql_query = agent_inference(agent, question)
                    is_correct = compare_answers_smart(
                        actual, gt, e_type, sql_result=sql_res
                    )
                except Exception as e:
                    actual = f"Error: {e}"
                    sql_res = "Error"
                    sql_query = "Error"
                    is_correct = False

                row[f"{agent.name}_actual"] = actual
                row[f"{agent.name}_sql_query"] = sql_query
                row[f"{agent.name}_sql_result"] = sql_res
                row[f"{agent.name}_correct"] = is_correct

                if is_correct:
                    total_scores[agent.name] += 1
                else:
                    problematic = True

                    # Categorize error
                    if sql_res == "[]" or sql_res == "[(None,)]":
                        error_categories["empty_result_error"].append(i)
                    elif (
                        "Europe, Middle East, and Africa" in question
                        or "歐非中東" in question
                    ):
                        error_categories["region_alias_error"].append(i)
                    elif e_type == "dataframe":
                        error_categories["dataframe_format_mismatch"].append(i)
                    elif (
                        "syntax" in str(actual).lower()
                        or "error" in str(sql_query).lower()
                    ):
                        error_categories["sql_syntax_error"].append(i)
                    else:
                        error_categories["other_error"].append(i)

            row["problematic"] = problematic
            results_matrix.append(row)

            # --- Dynamic Output: Write JSON after each step ---
            try:
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(results_matrix, f, indent=4, ensure_ascii=False)
            except Exception as e:
                print(f"Warning: Failed to write intermediate JSON: {e}")

        print("\nEvaluation Complete.")

        # Calculate Accuracies
        for name in agent_names:
            acc = total_scores[name] / len(eval_set)
            mlflow.log_metric(f"{name}_accuracy", acc)
            print(f"Agent '{name}' Accuracy: {acc:.2%}")

        # Log error categories
        print("\n=== Error Categories ===")
        for category, ids in error_categories.items():
            if ids:
                print(f"{category}: {len(ids)} errors")
                mlflow.log_metric(f"error_{category}_count", len(ids))

        # Create Comparison Table
        df = pd.DataFrame(results_matrix)

        # Save detailed CSV
        csv_path = output_dir / "eval6_comparison.csv"
        df.to_csv(csv_path, index=False)
        mlflow.log_artifact(str(csv_path))

        # Save HTML view
        html_path = output_dir / "eval6_comparison.html"
        df.to_html(html_path)
        mlflow.log_artifact(str(html_path))

        # Save JSON results (Final write)
        df.to_json(json_path, orient="records", indent=4, force_ascii=False)
        mlflow.log_artifact(str(json_path))

        # Log Table to MLflow (for UI exploration)
        try:
            mlflow.log_table(data=df, artifact_file="comparison_table.json")
        except Exception as e:
            print(f"Warning: Failed to log MLflow table: {e}")

        # Filter strictly problematic
        prob_df = df[df["problematic"] == True]
        prob_csv_path = output_dir / "eval6_problematic.csv"
        prob_df.to_csv(prob_csv_path, index=False)
        mlflow.log_artifact(str(prob_csv_path))

        print(f"Saved comparison results to {csv_path}")


if __name__ == "__main__":
    # agents_to_eval = [planned_agent_v5]
    agents_to_eval = [planned_agent_v6]
    run_comparative_evaluation(agents_to_eval, eval_set)
