import pandas as pd
import sqlite3
import json
import pathlib

PROJECT_DIR = pathlib.Path(__file__).parent.parent.resolve()
DATA_DIR = PROJECT_DIR / "data"
EVAL_SET_PATH = DATA_DIR / "eval_set_v3.jsonl"
DB_PATH = DATA_DIR / "global_sales_data.sqlite"
CSV_PATH = DATA_DIR / "global_sales_data_logic_distribution.csv"


def check_mismatch():
    print(f"Checking data mismatch...")
    print(f"Eval Set: {EVAL_SET_PATH}")
    print(f"DB Path: {DB_PATH}")
    print(f"CSV Path: {CSV_PATH}")

    # 1. Load Eval Set
    with open(EVAL_SET_PATH, "r") as f:
        eval_data = [json.loads(line) for line in f]

    # 2. Load CSV
    df_csv = pd.read_csv(CSV_PATH)

    # 3. Connect to DB
    conn = sqlite3.connect(DB_PATH)

    # Queries to check
    targets = [
        {
            "country": "India",
            "question_snippet": "India 訂單數量最多的客戶",
            "sql": """
                SELECT "Customer Name", COUNT(DISTINCT "Order ID") as cnt 
                FROM sales_data 
                WHERE "Country" = 'India' 
                GROUP BY "Customer Name" 
                ORDER BY cnt DESC LIMIT 1
            """,
        },
        {
            "country": "Germany",
            "question_snippet": "Germany 訂單數量最多的客戶",
            "sql": """
                SELECT "Customer Name", COUNT(DISTINCT "Order ID") as cnt 
                FROM sales_data 
                WHERE "Country" = 'Germany' 
                GROUP BY "Customer Name" 
                ORDER BY cnt DESC LIMIT 1
            """,
        },
    ]

    for target in targets:
        print(f"\n--- Checking {target['country']} ---")

        # A. Get Expected from Eval Set
        expected = None
        for entry in eval_data:
            if target["question_snippet"] in entry["question"]:
                expected = entry["ground_truth"]
                break
        print(f"Eval Set Expected: {expected}")

        # B. Get Actual from DB
        try:
            cursor = conn.cursor()
            cursor.execute(target["sql"])
            res = cursor.fetchone()
            print(f"DB Actual: {res}")
        except Exception as e:
            print(f"DB Error: {e}")

        # C. Get Actual from CSV (using pandas to replicate logic)
        try:
            ans = (
                df_csv[df_csv["Country"] == target["country"]]
                .groupby("Customer Name")["Order ID"]
                .count()
                .idxmax()
            )
            count = (
                df_csv[df_csv["Country"] == target["country"]]
                .groupby("Customer Name")["Order ID"]
                .count()
                .max()
            )
            print(f"CSV Actual: {ans} (Count: {count})")
        except Exception as e:
            print(f"CSV Error: {e}")

    conn.close()


if __name__ == "__main__":
    check_mismatch()
