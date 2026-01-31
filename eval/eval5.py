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
from text2sql.agents.planned_agent.agent_v4 import planned_agent_v4
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types


# MLflow Setup
mlflow.set_tracking_uri("http://localhost:5000")
experiment_name = "Text-to-SQL-Comparison-v5"
mlflow.set_experiment(experiment_name)

PROJECT_DIR = pathlib.Path(__file__).parent.parent.resolve()
DATA_DIR = PROJECT_DIR / "data"
# UPDATE: Use v2 ground truth set
ground_truth_set = DATA_DIR / "eval_set_v2.jsonl"

# Load Eval Set
with open(ground_truth_set, "r", encoding="utf-8") as f:
    eval_set = [json.loads(line) for line in f]

# Global Session Service
session_service = InMemorySessionService()
APP_NAME = "text2sql_eval_v5"
USER_ID = "eval_runner_v5"

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


def compare_answers_smart(actual_text, expected, eval_type, eps=1e-2, sql_result=None):
    try:
        # Prioirity 1: Use sql_result if available as it is more structured
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
            # Expected is typically a string repr of a dict: "{'k': v, ...}"
            # We accept loose matching: keys and values must match loosely
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

            # Convert result to a flat list of items to search
            flat_items = []

            def flatten(data):
                if isinstance(data, (list, tuple)):
                    for x in data:
                        flatten(x)
                elif isinstance(data, dict):
                    for k, v in data.items():
                        flatten(k)
                        flatten(v)
                else:
                    flat_items.append(data)

            flatten(parsed_result)

            # Also flatten expected dict
            # We want to check if every key and value in expected exists in the result
            all_values_found = True
            for k, v in expected_dict.items():
                # Check Key
                key_found = False
                str_k = normalize_string(k)
                for item in flat_items:
                    if normalize_string(item) == str_k:
                        key_found = True
                        break

                # Check Value (numeric tolerance)
                val_found = False
                if isinstance(v, (int, float)):
                    for item in flat_items:
                        try:
                            if isinstance(item, (int, float)) or (
                                isinstance(item, str)
                                and item.replace(".", "", 1).isdigit()
                            ):
                                if abs(float(item) - v) < eps:
                                    val_found = True
                                    break
                        except:
                            pass
                else:
                    str_v = normalize_string(v)
                    for item in flat_items:
                        if normalize_string(item) == str_v:
                            val_found = True
                            break

                if not (key_found and val_found):
                    all_values_found = False
                    break

            return all_values_found

    except Exception as e:
        # print(f"Comparison Error: {e}")
        return False
    return False


# %% Async Agent Wrapper


async def run_agent_once(agent, question):
    session_id = f"eval5_{agent.name}_{int(time.time()*1000)}_{uuid.uuid4().hex[:6]}"

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
    run_name = f"comparison_v5_{int(time.time())}"

    # Output directory
    output_dir = PROJECT_DIR / "eval" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "eval5_comparison.json"

    with mlflow.start_run(run_name=run_name):
        results_matrix = []

        print(
            f"Starting comparison for {len(agents)} agents on {len(eval_set)} questions..."
        )

        agent_names = [a.name for a in agents]
        mlflow.log_param("agents", agent_names)

        total_scores = {name: 0 for name in agent_names}

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

        # Create Comparison Table
        df = pd.DataFrame(results_matrix)

        # Save detailed CSV
        csv_path = output_dir / "eval5_comparison.csv"
        df.to_csv(csv_path, index=False)
        mlflow.log_artifact(str(csv_path))

        # Save HTML view
        html_path = output_dir / "eval5_comparison.html"
        df.to_html(html_path)
        mlflow.log_artifact(str(html_path))

        # Save JSON results (Final write)
        df.to_json(json_path, orient="records", indent=4, force_ascii=False)
        mlflow.log_artifact(str(json_path))

        # Log Table to MLflow (for UI exploration)
        try:
            # Log the entire comparison table to MLflow Artifacts for rich table view
            mlflow.log_table(data=df, artifact_file="comparison_table.json")
        except Exception as e:
            print(f"Warning: Failed to log MLflow table: {e}")

        # Filter strictly problematic
        prob_df = df[df["problematic"] == True]
        prob_csv_path = output_dir / "eval5_problematic.csv"
        prob_df.to_csv(prob_csv_path, index=False)
        mlflow.log_artifact(str(prob_csv_path))

        print(f"Saved comparison results to {csv_path}")


if __name__ == "__main__":
    agents_to_eval = [planned_agent_v4]
    run_comparative_evaluation(agents_to_eval, eval_set)
