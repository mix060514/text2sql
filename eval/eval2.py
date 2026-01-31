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

# Ignore warnings
import warnings

warnings.filterwarnings("ignore")

# Configure Logging
logging.basicConfig(level=logging.ERROR)

# Setup path to import local src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

# Import Planned Agent components
from text2sql.agents.planned_agent.agent import planned_agent
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types

# MLflow Setup
mlflow.set_tracking_uri("http://localhost:5000")
experiment_name = "Text-to-SQL-Planned-Agent"
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
    """Extracts all numbers from text, handling commas and decimals."""
    # Pattern for numbers: optional negative, digits with optional commas, optional decimal part
    # We remove commas before casting to float
    # Note: capturing groups need care.
    # Matches: "123", "12,345.67", "-0.5"
    matches = re.findall(r"-?\d{1,3}(?:,\d{3})*(?:\.\d+)?|-?\d+(?:\.\d+)?", str(text))
    cleaned = []
    for m in matches:
        try:
            val = float(m.replace(",", ""))
            cleaned.append(val)
        except:
            pass
    return cleaned


def compare_answers_smart(actual_text, expected, eval_type, eps=1e-2):
    """
    Compares the agent's natural language response with ground truth
    using heuristic extraction.
    """
    try:
        if eval_type == "number":
            # Extract numbers from text
            nums = extract_numbers(actual_text)
            expected_val = float(expected)

            # If any number in the response matches the ground truth
            for n in nums:
                if abs(n - expected_val) < eps:
                    return True
            return False

        elif eval_type == "string":
            # Check if relevant string is contained
            return str(expected).strip().lower() in str(actual_text).strip().lower()

        elif eval_type == "list":
            # Check if all expected items are present in the text
            # This is a 'recall' based check.
            if not isinstance(expected, (list, tuple)):
                return False

            text_lower = str(actual_text).lower()
            for item in expected:
                if str(item).lower() not in text_lower:
                    return False
            return True

        elif eval_type == "dataframe":
            # Very hard to check dataframe against text summary unless we parse markdown table
            # For now, return False or try basic checks
            return False

    except Exception as e:
        print(f"Comparison Error: {e}")
        return False

    return False


# %% Async Agent Wrapper


async def run_agent_once(question):
    # Create a unique session ID for this run to ensure no context leakage
    session_id = f"eval_{int(time.time()*1000)}_{uuid.uuid4().hex[:6]}"

    await session_service.create_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=session_id
    )

    runner = Runner(
        agent=planned_agent,
        app_name=APP_NAME,
        session_service=session_service,
    )

    content = types.Content(role="user", parts=[types.Part(text=question)])
    final_text = "No response"

    try:
        async for event in runner.run_async(
            user_id=USER_ID, session_id=session_id, new_message=content
        ):
            if event.is_final_response():
                if event.content and event.content.parts:
                    final_text = event.content.parts[0].text
                elif event.actions and event.actions.escalate:
                    final_text = f"Escalated: {event.error_message}"
                break
    except Exception as e:
        final_text = f"Agent Error: {str(e)}"

    return final_text


def planned_agent_inference(question):
    return asyncio.run(run_agent_once(question))


# %% Evaluation Loop (Modified)


def run_evaluation(inference_func, config, eval_set):
    run_name = (
        f"{config.get('agent', 'planned_agent')}_{config.get('model', 'unknown')}"
    )

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(config)
        score = 0
        latencies = []
        detailed_results = []

        print(f"Starting evaluation for {len(eval_set)} questions...")

        for i, entry in enumerate(eval_set):
            question = entry["question"]
            gt = entry["ground_truth"]
            e_type = entry["eval_type"]

            print(f"[{i+1}/{len(eval_set)}] Q: {question[:50]}...", end="\r")

            start_time = time.time()
            try:
                actual_answer = inference_func(question)
                is_correct = compare_answers_smart(actual_answer, gt, e_type)
            except Exception as e:
                actual_answer = f"Error: {str(e)}"
                is_correct = False
            end_time = time.time()
            latency = end_time - start_time
            latencies.append(latency)

            if is_correct:
                score += 1

            detailed_results.append(
                {
                    "question": question,
                    "expected": gt,
                    "actual": actual_answer,
                    "type": e_type,
                    "is_correct": is_correct,
                    "latency": latency,
                }
            )

        print(f"\nEvaluation Complete.")

        accuracy = score / len(eval_set)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("avg_latency", np.mean(latencies))

        # Save details
        detail_file = "eval2_results.json"
        with open(detail_file, "w", encoding="utf-8") as f:
            json.dump(detailed_results, f, ensure_ascii=False, indent=4)
        mlflow.log_artifact(detail_file)

        print(f"Run '{run_name}' Accuracy: {accuracy:.2%}")
        return accuracy


# %% Run Experiments

if __name__ == "__main__":
    # Define experiment config
    config = {
        "agent": "planned_agent_adk",
        "model": "openai/qwen3-4b-instruct-2507",  # Matching the agent definition
        "strategy": "plan_validate_execute",
    }

    run_evaluation(planned_agent_inference, config, eval_set)
