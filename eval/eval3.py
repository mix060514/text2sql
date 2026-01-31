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
from text2sql.agents.planned_agent.agent import planned_agent
from text2sql.agents.planned_agent.agent_v2 import planned_agent_v2
from text2sql.agents.planned_agent.agent_v3 import planned_agent_v3
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types


# MLflow Setup
mlflow.set_tracking_uri("http://localhost:5000")
experiment_name = "Text-to-SQL-Comparison"
mlflow.set_experiment(experiment_name)

PROJECT_DIR = pathlib.Path(__file__).parent.parent.resolve()
DATA_DIR = PROJECT_DIR / "data"
ground_truth_set = DATA_DIR / "eval_set.jsonl"

# Load Eval Set
with open(ground_truth_set, "r", encoding="utf-8") as f:
    eval_set = [json.loads(line) for line in f]

# Global Session Service
session_service = InMemorySessionService()
APP_NAME = "text2sql_eval"
USER_ID = "eval_runner"

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


def compare_answers_smart(actual_text, expected, eval_type, eps=1e-2, sql_result=None):
    try:
        if eval_type == "number":
            # 1. Try to find number in sql_result if available
            if sql_result:
                try:
                    # Parse string representation of list like "[(39932364.28,)]"
                    raw_data = ast.literal_eval(sql_result)
                    # Flatten/extract all numbers from the structure
                    candidates = []

                    def extract_from_struct(data):
                        if isinstance(data, (list, tuple)):
                            for item in data:
                                extract_from_struct(item)
                        elif isinstance(data, (int, float)):
                            candidates.append(data)
                        elif isinstance(data, str):
                            # Try to convert string to float if it looks like a number
                            try:
                                candidates.append(float(data))
                            except:
                                pass

                    extract_from_struct(raw_data)

                    expected_val = float(expected)
                    for n in candidates:
                        if abs(n - expected_val) < eps:
                            return True
                except:
                    pass

            # 2. Fallback to extracting from text
            nums = extract_numbers(actual_text)
            expected_val = float(expected)
            for n in nums:
                if abs(n - expected_val) < eps:
                    return True
            return False

        elif eval_type == "string":
            return str(expected).strip().lower() in str(actual_text).strip().lower()

        elif eval_type == "list":
            if not isinstance(expected, (list, tuple)):
                return False
            text_lower = str(actual_text).lower()
            for item in expected:
                if str(item).lower() not in text_lower:
                    return False
            return True

        elif eval_type == "dataframe":
            return False

    except Exception as e:
        # print(f"Comparison Error: {e}")
        return False
    return False


# %% Async Agent Wrapper


async def run_agent_once(agent, question):
    session_id = f"eval_{agent.name}_{int(time.time()*1000)}_{uuid.uuid4().hex[:6]}"

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

    try:
        async for event in runner.run_async(
            user_id=USER_ID, session_id=session_id, new_message=content
        ):
            # Capture tool results
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if (
                        part.function_response
                        and part.function_response.name == "execute_sql"
                    ):
                        # part.function_response.response is a dict {'result': '...'}
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

    return final_text, executed_sql_result


def agent_inference(agent, question):
    return asyncio.run(run_agent_once(agent, question))


# %% Evaluation Loop


def run_comparative_evaluation(agents, eval_set):
    run_name = f"comparison_{int(time.time())}"

    with mlflow.start_run(run_name=run_name):
        results_matrix = (
            []
        )  # List of dicts: {q_idx, question, expected, agent1_res, agent1_ok, agent2_res, agent2_ok, ...}

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
                    actual, sql_res = agent_inference(agent, question)
                    is_correct = compare_answers_smart(
                        actual, gt, e_type, sql_result=sql_res
                    )
                except Exception as e:
                    actual = f"Error: {e}"
                    sql_res = "Error"
                    is_correct = False

                row[f"{agent.name}_actual"] = actual
                row[f"{agent.name}_sql_result"] = sql_res
                row[f"{agent.name}_correct"] = is_correct

                if is_correct:
                    total_scores[agent.name] += 1
                else:
                    problematic = True

            row["problematic"] = problematic
            results_matrix.append(row)

        print("\nEvaluation Complete.")

        # Calculate Accuracies
        for name in agent_names:
            acc = total_scores[name] / len(eval_set)
            mlflow.log_metric(f"{name}_accuracy", acc)
            print(f"Agent '{name}' Accuracy: {acc:.2%}")

        # Create Comparison Table
        df = pd.DataFrame(results_matrix)

        # Create output directory
        output_dir = PROJECT_DIR / "eval" / "results"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save detailed CSV
        csv_path = output_dir / "eval3_comparison.csv"
        df.to_csv(csv_path, index=False)
        mlflow.log_artifact(str(csv_path))

        # Save HTML view for easier reading
        html_path = output_dir / "eval3_comparison.html"
        df.to_html(html_path)
        mlflow.log_artifact(str(html_path))

        # Save JSON results
        json_path = output_dir / "eval3_comparison.json"
        df.to_json(json_path, orient="records", indent=4, force_ascii=False)
        mlflow.log_artifact(str(json_path))

        # Filter strictly problematic questions (where at least one agent failed)
        prob_df = df[df["problematic"] == True]
        prob_csv_path = output_dir / "eval3_problematic.csv"
        prob_df.to_csv(prob_csv_path, index=False)
        mlflow.log_artifact(str(prob_csv_path))

        print(f"Saved comparison results to {csv_path}")
        print(f"Identify problematic questions in {prob_csv_path}")


if __name__ == "__main__":
    agents_to_eval = [planned_agent_v3]
    # To run all agents uncomment the line below
    # agents_to_eval = [planned_agent, planned_agent_v2, planned_agent_v3]
    run_comparative_evaluation(agents_to_eval, eval_set)
