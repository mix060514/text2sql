# %%
import json
import pathlib
import time
import numpy as np
import mlflow
import pandas as pd

# 指向你的本地 MLflow 伺服器
mlflow.set_tracking_uri("http://localhost:5000")
experiment_name = "Text-to-SQL"
mlflow.set_experiment(experiment_name)

PROJECT_DIR = pathlib.Path(__file__).parent.parent.resolve()
DATA_DIR = PROJECT_DIR / "data"
ground_truth_set = DATA_DIR / "eval_set.jsonl"

# 載入考卷
with open(ground_truth_set, "r", encoding="utf-8") as f:
    eval_set = [json.loads(line) for line in f]


# %%
def compare_answers(actual, expected, eval_type):
    try:
        # 1. 處理數字
        if eval_type == "number":
            return abs(float(actual) - float(expected)) < 1e-2

        # 2. 處理字串
        elif eval_type == "string":
            return str(actual).strip().lower() == str(expected).strip().lower()

        # 3. 處理清單 (修正點)
        elif eval_type == "list":
            # 如果 actual 不是 list 或 tuple，直接判錯，避免迭代錯誤
            if not isinstance(actual, (list, tuple)):
                return False
            return set(actual) == set(expected)

        # 4. 處理 DataFrame (字典)
        elif eval_type == "dataframe":
            if isinstance(actual, str):
                actual = json.loads(actual)
            return actual == expected

    except Exception as e:
        print(f"比對錯誤 [Type: {eval_type}]: {e}")
        return False
    return False


# %%
def run_evaluation(inference_func, config, eval_set):
    run_name = f"{config.get('model', 'unknown')}_{config.get('prompt', 'default')}_{config.get('tool', 'custom')}"
    # run_name = f"{config['model']}_{config['prompt']}_{config['tool']}"

    # 啟動 MLflow Run
    with mlflow.start_run(run_name=run_name):
        # 紀錄方法描述
        mlflow.log_params(config)
        score = 0
        latencies = []
        detailed_results = []

        for entry in eval_set:
            question = entry["question"]
            gt = entry["ground_truth"]
            e_type = entry["eval_type"]

            start_time = time.time()
            try:
                actual_answer = inference_func(question)
                is_correct = compare_answers(actual_answer, gt, e_type)
            except Exception as e:
                actual_answer = f"Error: {str(e)}"
                is_correct = False
            end_time = time.time()
            latency = end_time - start_time
            latencies.append(latency)

            if is_correct:
                score += 1

            # 收集每一題的細節
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

        # 紀錄最終指標
        accuracy = score / len(eval_set)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("avg_latency", np.mean(latencies))
        mlflow.log_metric("p95_latency", np.percentile(latencies, 95))

        detail_file = "eval_results.json"
        with open(detail_file, "w", encoding="utf-8") as f:
            json.dump(detailed_results, f, ensure_ascii=False, indent=4)

        mlflow.log_artifact(detail_file)

        print(f"Run '{run_name}' 完成！準確率: {accuracy:.2%}")
        return accuracy


def my_llm_method(question):
    # time.sleep(0.5) # 模擬思考時間
    return 0.5


# 這是模擬 Google Agent 的函式
def my_google_agent_method(question):
    # time.sleep(1.5) # Agent 通常比較慢
    return 0.5


# 定義你想測試的組合
experiments = [
    {
        "func": my_llm_method,
        "config": {"model": "gpt-4o", "prompt": "zero-shot", "tool": "direct_call"},
    },
    {
        "func": my_google_agent_method,
        "config": {"model": "gemini-1.5", "prompt": "CoT", "tool": "direct_call"},
    },
]

# 跑迴圈
for exp in experiments:
    print(f"正在執行: {exp['config']}")
    run_evaluation(inference_func=exp["func"], config=exp["config"], eval_set=eval_set)
