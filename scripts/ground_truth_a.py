import pathlib
import pandas as pd
import json
import os


PROJECT_PATH = pathlib.Path(__file__).parent.parent.resolve()
DATA_PATH = PROJECT_PATH / "data"

# 1. 載入資料並做基礎處理
df = pd.read_csv(DATA_PATH / "global_sales_data_logic_distribution.csv")
df["Order Date"] = pd.to_datetime(df["Order Date"])
df["year"] = df["Order Date"].dt.year

eval_set = []

# --- A. 產生基礎聚合問題 (Number) ---
# 針對每個區域產生總銷售額問題
for region in df["Region"].unique():
    ans = round(df[df["Region"] == region]["Total Revenue"].sum(), 2)
    eval_set.append(
        {
            "question": f"{region} 的總銷售額是多少？",
            "ground_truth": ans,
            "eval_type": "number",
            "difficulty": "simple",
        }
    )

# --- B. 產生清單類問題 (List) ---
# 每個區域銷售量前 3 名的產品
for region in df["Region"].unique():
    ans = (
        df[df["Region"] == region]
        .groupby("Product Name")["Quantity"]
        .sum()
        .nlargest(3)
        .index.tolist()
    )
    eval_set.append(
        {
            "question": f"{region} 銷售量最高的前三個產品是什麼？",
            "ground_truth": ans,
            "eval_type": "list",
            "difficulty": "medium",
        }
    )

# --- C. 產生 DataFrame 類問題 (JSON Dict) ---
# 某產品在各年份的銷售趨勢
for category in df["Product Category"].unique():
    ans = (
        df[df["Product Category"] == category]
        .groupby("year")["Total Revenue"]
        .sum()
        .to_dict()
    )
    eval_set.append(
        {
            "question": f"請列出 {category} 類別在各年份的銷售額總計。",
            "ground_truth": ans,
            "eval_type": "dataframe",
            "difficulty": "hard",
        }
    )

# --- D. 產生篩選問題 (String) ---
# 某個國家中訂單量最大的客戶
for country in df["Country"].sample(5).unique():  # 隨機抽5個國家
    ans = (
        df[df["Country"] == country]
        .groupby("Customer Name")["Order ID"]
        .count()
        .idxmax()
    )
    eval_set.append(
        {
            "question": f"{country} 訂單數量最多的客戶是誰？",
            "ground_truth": ans,
            "eval_type": "string",
            "difficulty": "medium",
        }
    )

# 存入 JSONL 檔案
os.makedirs("data", exist_ok=True)
with open(DATA_PATH / "eval_set.jsonl", "a", encoding="utf-8") as f:
    for entry in eval_set:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"成功產生 {len(eval_set)} 筆測試案例於 data/eval_set.jsonl")
