import pathlib
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# 設定隨機種子以確保結果可重現
np.random.seed(42)
random.seed(42)

# ==========================================
# 1. 參數設定 (Configuration)
# ==========================================
NUM_ROWS = 30000
PROJECT_PATH = pathlib.Path(__file__).parent.parent.resolve()
DATA_PATH = PROJECT_PATH / "data"
print(f"{PROJECT_PATH =}")
print(f"{DATA_PATH =}")


# 產品清單：包含 (Category, Product Name, Base Price, Weight/Popularity)
# Weight 越高代表該產品賣得越好 (長尾頭部)
products_config = [
    # Electronics
    ("Electronics", "Enterprise Laptop X1", 1200, 50),
    ("Electronics", "Pro Smartphone 15", 900, 40),
    ("Electronics", '4K Monitor 27"', 350, 30),
    ("Electronics", "Wireless Headset", 150, 20),
    ("Electronics", "Docking Station", 200, 15),
    # Office Supplies
    ("Office Supplies", "Ergonomic Chair", 400, 25),
    ("Office Supplies", "Standing Desk", 600, 20),
    ("Office Supplies", "Meeting Whiteboard", 300, 10),
    # Software
    ("Software", "Cloud License (Annual)", 5000, 5),  # 高價但少
    ("Software", "Security Suite", 1500, 15),
    ("Software", "Team Collaboration Tool", 200, 30),
]

# 地區權重：模擬真實市場規模 (NA/APAC 較大)
regions_config = {
    "North America": {"countries": ["United States", "Canada"], "weight": 0.40},
    "EMEA": {
        "countries": ["United Kingdom", "Germany", "France", "Netherlands"],
        "weight": 0.30,
    },
    "APAC": {
        "countries": ["Japan", "China", "Singapore", "Australia", "India"],
        "weight": 0.25,
    },
    "LATAM": {"countries": ["Brazil", "Mexico"], "weight": 0.05},
}

# ==========================================
# 2. 生成邏輯 (Generation Logic)
# ==========================================

# --- A. 產品與價格 (Weighted & Normal Distribution) ---
# 解構配置
prod_cats = [p[0] for p in products_config]
prod_names = [p[1] for p in products_config]
prod_prices = [p[2] for p in products_config]
prod_weights = [p[3] for p in products_config]
# 正規化權重
total_weight = sum(prod_weights)
probs = [w / total_weight for w in prod_weights]

# 依權重隨機選擇產品索引
prod_indices = np.random.choice(len(products_config), size=NUM_ROWS, p=probs)

# 根據索引填入資料
data_category = [prod_cats[i] for i in prod_indices]
data_product = [prod_names[i] for i in prod_indices]
base_prices = np.array([prod_prices[i] for i in prod_indices])

# 價格常態分佈：在基準價格上下浮動 10% (模擬折扣或波動)
# loc=平均值, scale=標準差
data_asp = np.random.normal(loc=base_prices, scale=base_prices * 0.05)
data_asp = np.round(data_asp, 2)

# --- B. 數量 (Long Tail / Log-Normal Distribution) ---
# 使用對數常態分佈模擬長尾：大部分訂單數量小 (1-5)，少數訂單數量極大 (50-100+)
# sigma 控制尾巴長度
qty_dist = np.random.lognormal(mean=1.0, sigma=1.0, size=NUM_ROWS)
data_qty = np.maximum(1, np.round(qty_dist)).astype(int)

# --- C. 日期 (Seasonality) ---
# 模擬季節性：Q4 (10, 11, 12月) 權重較高
start_date = datetime(2023, 1, 1)
date_range_days = 730  # 2 years
all_days = [start_date + timedelta(days=x) for x in range(date_range_days)]

# 為每一天分配權重 (Q4 加權)
day_weights = []
for day in all_days:
    w = 1.0
    if day.month in [10, 11, 12]:
        w = 1.5  # 旺季加權
    # 加上一點隨機雜訊
    day_weights.append(w * random.uniform(0.8, 1.2))

# 正規化日期權重
day_probs = np.array(day_weights) / sum(day_weights)
selected_dates = np.random.choice(all_days, size=NUM_ROWS, p=day_probs)
data_date = [d.strftime("%Y-%m-%d") for d in selected_dates]

# --- D. 客戶 (Pareto Principle / 80-20 Rule) ---
# 生成 1000 個客戶池
company_suffixes = ["Inc.", "Corp.", "Ltd.", "Solutions", "Systems", "Group"]
company_prefixes = [
    "Alpha",
    "Beta",
    "Gamma",
    "Delta",
    "Apex",
    "Summit",
    "Global",
    "Next",
    "Prime",
    "Elite",
]
customers_pool = [
    f"{random.choice(company_prefixes)} {random.choice(company_prefixes)} {random.choice(company_suffixes)}"
    for _ in range(1000)
]

# 分配客戶權重：前 200 個客戶 (VIP) 擁有 80% 的出現機率
customer_weights = np.concatenate(
    [np.full(200, 80 / 200), np.full(800, 20 / 800)]  # VIP  # 一般客戶
)
customer_weights = customer_weights / customer_weights.sum()

data_customer = np.random.choice(customers_pool, size=NUM_ROWS, p=customer_weights)

# --- E. 地區 (Weighted) ---
region_keys = list(regions_config.keys())
region_probs = [regions_config[k]["weight"] for k in region_keys]
selected_regions = np.random.choice(region_keys, size=NUM_ROWS, p=region_probs)

data_country = []
for r in selected_regions:
    # 在該區域內隨機選國家
    c_list = regions_config[r]["countries"]
    data_country.append(random.choice(c_list))

# --- F. 其他欄位 ---
# Order ID
data_order_id = [f"ORD-{202300001 + i}" for i in range(NUM_ROWS)]

# 計算總營收
data_revenue = np.round(data_asp * data_qty, 2)

# ==========================================
# 3. 建立 DataFrame 與輸出
# ==========================================
df = pd.DataFrame(
    {
        "Order ID": data_order_id,
        "Order Date": data_date,
        "Region": selected_regions,
        "Country": data_country,
        "Customer Name": data_customer,
        "Product Category": data_category,
        "Product Name": data_product,
        "ASP": data_asp,
        "Quantity": data_qty,
        "Total Revenue": data_revenue,
    }
)

# 排序一下讓資料看起來稍微順一點 (Optional)
df = df.sort_values(by="Order Date").reset_index(drop=True)

# 儲存
csv_filename = "global_sales_data_logic_distribution.csv"
df.to_csv(DATA_PATH / csv_filename, index=False)
import sqlite3

db_filename = "global_sales_data.sqlite"
with sqlite3.connect(DATA_PATH / db_filename) as conn:
    df.to_sql("sales_data", conn, if_exists="replace", index=False)

print(f"File '{csv_filename}' generated with {NUM_ROWS} rows.")
print("Logic applied: Pareto for Customers, Log-Normal for Qty, Seasonality for Dates.")
print(df.head(10))
