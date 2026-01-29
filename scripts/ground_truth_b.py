import pathlib
import pandas as pd
import json
import os

PROJECT_PATH = pathlib.Path(__file__).parent.parent.resolve()
DATA_PATH = PROJECT_PATH / "data"

# 1. 載入資料並進行進階時間維度處理
df = pd.read_csv(DATA_PATH / "global_sales_data_logic_distribution.csv")
df["Order Date"] = pd.to_datetime(df["Order Date"])
df["year"] = df["Order Date"].dt.year
df["quarter"] = df["Order Date"].dt.quarter
df["month"] = df["Order Date"].dt.month

eval_set = []

# --- A. 季度比較 (Quarterly Analysis) ---
# 針對特定年度，找出銷售額表現最好的季度
for year in df["year"].unique():
    q_data = df[df["year"] == year].groupby("quarter")["Total Revenue"].sum()
    best_q = int(q_data.idxmax())
    eval_set.append(
        {
            "question": f"{year} 年哪一個季度的總銷售額表現最好？",
            "ground_truth": f"Q{best_q}",
            "eval_type": "string",
            "difficulty": "medium",
        }
    )

# --- B. 年增率與成長幅度 (YoY & Growth) ---
# 這裡計算各產品類別在 2023 到 2024 的成長率
y2023 = df[df["year"] == 2023].groupby("Product Category")["Total Revenue"].sum()
y2024 = df[df["year"] == 2024].groupby("Product Category")["Total Revenue"].sum()
growth_rates = ((y2024 - y2023) / y2023 * 100).round(2)

for category, rate in growth_rates.items():
    eval_set.append(
        {
            "question": f"{category} 類別在 2024 年相對於 2023 年的銷售額年增率 (YoY) 是多少？",
            "ground_truth": rate,
            "eval_type": "number",  # 帶百分比建議用語義比對
            "difficulty": "hard",
        }
    )

# --- C. 長期趨勢 (Monthly/Quarterly Trend) ---
# 針對特定區域產出「月度」分布數據 (DataFrame 類)
for region in df["Region"].unique():
    ans = (
        df[df["Region"] == region]
        .groupby(["year", "month"])["Total Revenue"]
        .sum()
        .reset_index()
    )
    # 轉換成容易比對的字典格式：{(2023, 1): 12345, ...}
    ans_dict = {
        f"{int(y)}-{int(m)}": round(v, 2)
        for y, m, v in zip(ans["year"], ans["month"], ans["Total Revenue"])
    }

    eval_set.append(
        {
            "question": f"請列出 {region} 區域在 2023 至 2024 年每個月的銷售趨勢數據。",
            "ground_truth": ans_dict,
            "eval_type": "dataframe",
            "difficulty": "hard",
        }
    )

# --- D. 複雜篩選：特定時間區間的比較 ---
# 例如：比較兩年內 Q4 的銷售總額成長幅度
q4_2023 = df[(df["year"] == 2023) & (df["quarter"] == 4)]["Total Revenue"].sum()
q4_2024 = df[(df["year"] == 2024) & (df["quarter"] == 4)]["Total Revenue"].sum()
q4_growth = round(q4_2024 - q4_2023, 2)

eval_set.append(
    {
        "question": "2024 年第四季相較於 2023 年第四季，總銷售額增加了多少？",
        "ground_truth": q4_growth,
        "eval_type": "number",
        "difficulty": "hard",
    }
)

# 存入 JSONL 檔案
os.makedirs("data", exist_ok=True)
with open(DATA_PATH / "eval_set.jsonl", "a", encoding="utf-8") as f:
    for entry in eval_set:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"成功產生 {len(eval_set)} 筆進階測試案例於 data/eval_set.jsonl")
