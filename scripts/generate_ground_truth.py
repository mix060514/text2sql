import pathlib
import pandas as pd
import json
import os
import random

PROJECT_PATH = pathlib.Path(__file__).parent.parent.resolve()
DATA_PATH = PROJECT_PATH / "data"


def load_and_process_data():
    df = pd.read_csv(DATA_PATH / "global_sales_data_logic_distribution.csv")
    df["Order Date"] = pd.to_datetime(df["Order Date"])
    df["year"] = df["Order Date"].dt.year
    df["quarter"] = df["Order Date"].dt.quarter
    df["month"] = df["Order Date"].dt.month
    return df


# Region and Country Synonyms / Aliases
ALIASES = {
    # Regions
    "APAC": ["APAC", "亞太區", "亞太地區", "Asia Pacific"],
    "North America": ["North America", "北美", "北美洲", "NA"],
    "LATAM": ["LATAM", "拉美", "南美", "拉丁美洲", "Latin America"],
    "EMEA": ["EMEA", "歐洲中東非洲", "歐非中東", "Europe, Middle East, and Africa"],
    # Countries (Sample)
    "Germany": ["Germany", "德國", "德意志"],
    "China": ["China", "中國", "大陸", "PRC"],
    "United States": ["United States", "美國", "US", "USA"],
    "Japan": ["Japan", "日本"],
    "Canada": ["Canada", "加拿大"],
    "Australia": ["Australia", "澳洲", "澳大利亞"],
    "India": ["India", "印度"],
    "France": ["France", "法國"],
    "United Kingdom": ["United Kingdom", "英國", "UK"],
    "Brazil": ["Brazil", "巴西"],
    "Mexico": ["Mexico", "墨西哥"],
}


def get_aliases(name):
    return ALIASES.get(name, [name])


def generate_questions():
    df = load_and_process_data()
    eval_set = []

    # --- A. 基礎聚合問題 (Number) ---
    # 針對每個區域產生總銷售額問題
    for region in df["Region"].unique():
        ans = round(df[df["Region"] == region]["Total Revenue"].sum(), 2)
        # Generate variations for region name
        for alias in get_aliases(region):
            eval_set.append(
                {
                    "question": f"{alias} 的總銷售額是多少？",
                    "ground_truth": ans,
                    "eval_type": "number",
                    "difficulty": "simple",
                    "tags": ["revenue", "region", "simple"],
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
        for alias in get_aliases(region):
            eval_set.append(
                {
                    "question": f"{alias} 銷售量最高的前三個產品是什麼？",
                    "ground_truth": ans,
                    "eval_type": "list",
                    "difficulty": "medium",
                    "tags": ["top_products", "region", "medium"],
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
                "tags": ["trend", "category", "hard"],
            }
        )

    # --- D. 產生篩選問題 (String) ---
    # 某個國家中訂單量最大的客戶
    selected_countries = ["Canada", "India", "Germany", "Australia", "China"]
    for country in selected_countries:
        if country not in df["Country"].values:
            continue

        ans = (
            df[df["Country"] == country]
            .groupby("Customer Name")["Order ID"]
            .count()
            .idxmax()
        )
        for alias in get_aliases(country):
            eval_set.append(
                {
                    "question": f"{alias} 訂單數量最多的客戶是誰？",
                    "ground_truth": ans,
                    "eval_type": "string",
                    "difficulty": "medium",
                    "tags": ["top_customer", "country", "medium"],
                }
            )

    # --- E. 季度比較 (Quarterly Analysis) ---
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
                "tags": ["quarterly", "top_quarter", "medium"],
            }
        )

    # --- F. 年增率與成長幅度 (YoY & Growth) ---
    # 這裡計算各產品類別在 2023 到 2024 的成長率
    y2023 = df[df["year"] == 2023].groupby("Product Category")["Total Revenue"].sum()
    y2024 = df[df["year"] == 2024].groupby("Product Category")["Total Revenue"].sum()
    growth_rates = ((y2024 - y2023) / y2023 * 100).round(2)

    for category, rate in growth_rates.items():
        eval_set.append(
            {
                "question": f"{category} 類別在 2024 年相對於 2023 年的銷售額年增率 (YoY) 是多少？(請以百分比表示)",
                "ground_truth": rate,
                "eval_type": "number",
                "difficulty": "hard",
                "tags": ["yoy", "growth", "category", "hard"],
            }
        )

    # --- G. 長期趨勢 (Monthly/Quarterly Trend) ---
    # 針對特定區域產出「月度」分布數據 (DataFrame 類)
    for region in df["Region"].unique():
        ans = (
            df[df["Region"] == region]
            .groupby(["year", "month"])["Total Revenue"]
            .sum()
            .reset_index()
        )
        ans_dict = {
            f"{int(y)}-{int(m):02d}": round(v, 2)  # Zero-pad month to match SQL format
            for y, m, v in zip(ans["year"], ans["month"], ans["Total Revenue"])
        }

        for alias in get_aliases(region):
            eval_set.append(
                {
                    "question": f"請列出 {alias} 區域在 2023 至 2024 年每個月的銷售趨勢數據。",
                    "ground_truth": ans_dict,
                    "eval_type": "dataframe",
                    "difficulty": "hard",
                    "tags": ["trend", "monthly", "region", "hard"],
                }
            )

    # --- H. 複雜篩選：特定時間區間的比較 (Q4 2023 vs Q4 2024) ---
    q4_2023 = df[(df["year"] == 2023) & (df["quarter"] == 4)]["Total Revenue"].sum()
    q4_2024 = df[(df["year"] == 2024) & (df["quarter"] == 4)]["Total Revenue"].sum()
    q4_growth = round(q4_2024 - q4_2023, 2)

    eval_set.append(
        {
            "question": "2024 年第四季相較於 2023 年第四季，總銷售額增加了多少？",
            "ground_truth": q4_growth,
            "eval_type": "number",
            "difficulty": "hard",
            "tags": ["comparison", "quarterly", "hard"],
        }
    )

    # --- I. 進階任務 1: 前 5 大客戶購買產品占比 (North America 2023) ---
    region = "North America"
    year = 2023
    top_5_custs = (
        df[(df["Region"] == region) & (df["year"] == year)]
        .groupby("Customer Name")["Total Revenue"]
        .sum()
        .nlargest(5)
        .index.tolist()
    )
    top_5_data = df[(df["year"] == year) & (df["Customer Name"].isin(top_5_custs))]
    dist = (
        top_5_data.groupby(["Customer Name", "Product Category"])["Total Revenue"]
        .sum()
        .unstack(fill_value=0)
    )
    dist_pct = dist.div(dist.sum(axis=1), axis=0).round(4).to_dict("index")

    for alias in get_aliases(region):
        eval_set.append(
            {
                "question": f"在 {alias} 區域，{year} 年前五大客戶購買各產品類別的金額占比為何？",
                "ground_truth": dist_pct,
                "eval_type": "dataframe",
                "difficulty": "hard",
                "tags": ["advanced", "customer_segmentation", "hard"],
            }
        )

    # --- J. 進階任務 2: 變動最大的客戶 (2023 vs 2024 總額變動幅度) ---
    rev_23 = df[df["year"] == 2023].groupby("Customer Name")["Total Revenue"].sum()
    rev_24 = df[df["year"] == 2024].groupby("Customer Name")["Total Revenue"].sum()
    change = (rev_24 - rev_23).abs().sort_values(ascending=False)
    biggest_change_cust = change.index[0]

    eval_set.append(
        {
            "question": "在 2023 到 2024 年間，哪位客戶的購買總額變動幅度（絕對值）最大？",
            "ground_truth": biggest_change_cust,
            "eval_type": "string",
            "difficulty": "hard",
            "tags": ["advanced", "customer_churn", "hard"],
        }
    )

    # --- K. 進階任務 3: 每個產品/類別的最大客戶 (Electronics) ---
    category = "Electronics"
    top_cust_cat = (
        df[df["Product Category"] == category]
        .groupby("Customer Name")["Total Revenue"]
        .sum()
        .idxmax()
    )
    eval_set.append(
        {
            "question": f"產品類別 {category} 的最大購買客戶是誰？",
            "ground_truth": top_cust_cat,
            "eval_type": "string",
            "difficulty": "medium",
            "tags": ["top_customer", "category", "medium"],
        }
    )

    # --- L. 進階任務 4: 哪個產品 ASP 最高 ---
    highest_asp_prod = df.groupby("Product Name")["ASP"].mean().idxmax()
    eval_set.append(
        {
            "question": "哪一個產品的平均單價 (ASP) 最高？",
            "ground_truth": highest_asp_prod,
            "eval_type": "string",
            "difficulty": "simple",
            "tags": ["simple", "product"],
        }
    )

    # --- M. 進階任務 5: 各個 Region 偏好差異 (每個 Region 銷售額最高的類別) ---
    reg_pref = (
        df.groupby(["Region", "Product Category"])["Total Revenue"]
        .sum()
        .unstack()
        .idxmax(axis=1)
        .to_dict()
    )
    eval_set.append(
        {
            "question": "各個區域 (Region) 最偏好（銷售額最高）的產品類別分別為何？",
            "ground_truth": reg_pref,
            "eval_type": "dataframe",
            "difficulty": "medium",
            "tags": ["regional_preference", "medium"],
        }
    )

    return eval_set


if __name__ == "__main__":
    eval_set = generate_questions()

    # Shuffle for randomness if needed, or keep structured
    # random.shuffle(eval_set)

    output_file = DATA_PATH / "eval_set_v2.jsonl"
    os.makedirs(DATA_PATH, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        for entry in eval_set:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"成功產生 {len(eval_set)} 筆測試案例於 {output_file}")
