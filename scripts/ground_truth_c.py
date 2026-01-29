import pandas as pd
import json

# 1. 載入並處理時間
df = pd.read_csv("data/global_sales_data_logic_distribution.csv")
df["Order Date"] = pd.to_datetime(df["Order Date"])
df["year"] = df["Order Date"].dt.year

eval_set_advanced = []

# --- 任務 1: 前 5 大客戶購買產品占比 (以 North America 2023 為例) ---
region = "North America"
year = 2023
# 找出前五大客戶名單
top_5_custs = (
    df[(df["Region"] == region) & (df["year"] == year)]
    .groupby("Customer Name")["Total Revenue"]
    .sum()
    .nlargest(5)
    .index.tolist()
)
# 計算這些客戶在各類別的佔比
top_5_data = df[(df["year"] == year) & (df["Customer Name"].isin(top_5_custs))]
dist = (
    top_5_data.groupby(["Customer Name", "Product Category"])["Total Revenue"]
    .sum()
    .unstack(fill_value=0)
)
dist_pct = dist.div(dist.sum(axis=1), axis=0).round(4).to_dict("index")

eval_set_advanced.append(
    {
        "question": f"在 {region} 區域，{year} 年前五大客戶購買各產品類別的金額占比為何？",
        "ground_truth": dist_pct,
        "eval_type": "dataframe",
        "difficulty": "hard",
    }
)

# --- 任務 2: 變動最大的客戶 (2023 vs 2024 總額變動幅度) ---
rev_23 = df[df["year"] == 2023].groupby("Customer Name")["Total Revenue"].sum()
rev_24 = df[df["year"] == 2024].groupby("Customer Name")["Total Revenue"].sum()
# 計算絕對變動幅度 (絕對值)
change = (rev_24 - rev_23).abs().sort_values(ascending=False)
biggest_change_cust = change.index[0]

eval_set_advanced.append(
    {
        "question": "在 2023 到 2024 年間，哪位客戶的購買總額變動幅度（絕對值）最大？",
        "ground_truth": biggest_change_cust,
        "eval_type": "string",
        "difficulty": "hard",
    }
)

# --- 任務 3: 每個產品/類別的最大客戶 ---
category = "Electronics"
top_cust_cat = (
    df[df["Product Category"] == category]
    .groupby("Customer Name")["Total Revenue"]
    .sum()
    .idxmax()
)
eval_set_advanced.append(
    {
        "question": f"產品類別 {category} 的最大購買客戶是誰？",
        "ground_truth": top_cust_cat,
        "eval_type": "string",
        "difficulty": "medium",
    }
)

# --- 任務 4: 哪個產品 ASP 最高 ---
# 注意：ASP 可能因訂單而異，這裡取平均 ASP
highest_asp_prod = df.groupby("Product Name")["ASP"].mean().idxmax()
eval_set_advanced.append(
    {
        "question": "哪一個產品的平均單價 (ASP) 最高？",
        "ground_truth": highest_asp_prod,
        "eval_type": "string",
        "difficulty": "simple",
    }
)

# --- 任務 5: 各個 Region 偏好差異 (每個 Region 銷售額最高的類別) ---
reg_pref = (
    df.groupby(["Region", "Product Category"])["Total Revenue"]
    .sum()
    .unstack()
    .idxmax(axis=1)
    .to_dict()
)
eval_set_advanced.append(
    {
        "question": "各個區域 (Region) 最偏好（銷售額最高）的產品類別分別為何？",
        "ground_truth": reg_pref,
        "eval_type": "dataframe",
        "difficulty": "medium",
    }
)

# 存入 JSONL (使用 'a' 模式追加，不複寫)
with open("data/eval_set.jsonl", "a", encoding="utf-8") as f:
    for entry in eval_set_advanced:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
