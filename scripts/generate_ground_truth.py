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
        year_labels = [str(y) for y in sorted(ans.keys())]
        eval_set.append(
            {
                "question": f"請列出 {category} 類別在各年份的銷售額總計。",
                "ground_truth": ans,
                "eval_type": "dataframe",
                "difficulty": "hard",
                "tags": ["trend", "category", "hard"],
                "expected_plot": {
                    "should_plot": True,
                    "chart_type": "bar",
                    "expected_x_order": year_labels,
                    "expected_traces": 1,
                },
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
        month_labels = sorted(ans_dict.keys())

        for alias in get_aliases(region):
            eval_set.append(
                {
                    "question": f"請列出 {alias} 區域在 2023 至 2024 年每個月的銷售趨勢數據。",
                    "ground_truth": ans_dict,
                    "eval_type": "dataframe",
                    "difficulty": "hard",
                    "tags": ["trend", "monthly", "region", "hard"],
                    "expected_plot": {
                        "should_plot": True,
                        "chart_type": "line",
                        "expected_x_order": month_labels,
                        "expected_traces": 1,
                    },
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

    # =========================================================================
    # Plot 專用評估題組 (N ~ Q)
    # =========================================================================

    # --- N. Bar Chart 題組 (~15 題) ---

    # N1: 各區域總銷售額比較
    region_rev = (
        df.groupby("Region")["Total Revenue"].sum().sort_values(ascending=False)
    )
    eval_set.append(
        {
            "question": "請比較各區域的總銷售額。",
            "ground_truth": region_rev.round(2).to_dict(),
            "eval_type": "dataframe",
            "difficulty": "medium",
            "tags": ["revenue", "region", "bar", "plot"],
            "expected_plot": {
                "should_plot": True,
                "chart_type": "bar",
                "expected_x_order": region_rev.index.tolist(),
                "expected_traces": 1,
            },
        }
    )

    # N2-N4: 各產品類別總銷售額 / 銷售量 / 訂單數
    cat_rev = (
        df.groupby("Product Category")["Total Revenue"]
        .sum()
        .sort_values(ascending=False)
    )
    eval_set.append(
        {
            "question": "各產品類別的總銷售額分別是多少？請用圖表比較。",
            "ground_truth": cat_rev.round(2).to_dict(),
            "eval_type": "dataframe",
            "difficulty": "medium",
            "tags": ["revenue", "category", "bar", "plot"],
            "expected_plot": {
                "should_plot": True,
                "chart_type": "bar",
                "expected_x_order": cat_rev.index.tolist(),
                "expected_traces": 1,
            },
        }
    )

    cat_qty = (
        df.groupby("Product Category")["Quantity"]
        .sum()
        .sort_values(ascending=False)
    )
    eval_set.append(
        {
            "question": "各產品類別的總銷售量各是多少？",
            "ground_truth": cat_qty.to_dict(),
            "eval_type": "dataframe",
            "difficulty": "medium",
            "tags": ["quantity", "category", "bar", "plot"],
            "expected_plot": {
                "should_plot": True,
                "chart_type": "bar",
                "expected_x_order": cat_qty.index.tolist(),
                "expected_traces": 1,
            },
        }
    )

    cat_orders = (
        df.groupby("Product Category")["Order ID"]
        .count()
        .sort_values(ascending=False)
    )
    eval_set.append(
        {
            "question": "各產品類別的訂單數量各有多少？",
            "ground_truth": cat_orders.to_dict(),
            "eval_type": "dataframe",
            "difficulty": "medium",
            "tags": ["orders", "category", "bar", "plot"],
            "expected_plot": {
                "should_plot": True,
                "chart_type": "bar",
                "expected_x_order": cat_orders.index.tolist(),
                "expected_traces": 1,
            },
        }
    )

    # N5-N8: 各區域內國家銷售額
    region_country_questions = {
        "EMEA": "EMEA 區域中各國家的銷售額比較如何？",
        "APAC": "APAC 區域各國家的銷售額分別是多少？",
        "North America": "北美區域各國的銷售額比較。",
        "LATAM": "LATAM 區域各國家的銷售額分別是多少？",
    }
    for region, question in region_country_questions.items():
        country_rev = (
            df[df["Region"] == region]
            .groupby("Country")["Total Revenue"]
            .sum()
            .sort_values(ascending=False)
        )
        eval_set.append(
            {
                "question": question,
                "ground_truth": country_rev.round(2).to_dict(),
                "eval_type": "dataframe",
                "difficulty": "medium",
                "tags": ["revenue", "country", "bar", "plot"],
                "expected_plot": {
                    "should_plot": True,
                    "chart_type": "bar",
                    "expected_x_order": country_rev.index.tolist(),
                    "expected_traces": 1,
                },
            }
        )

    # N9-N11: 各類別產品銷售額排名
    cat_product_questions = {
        "Electronics": "Electronics 類別中各產品的銷售額排名為何？",
        "Office Supplies": "Office Supplies 類別各產品的銷售額比較。",
        "Software": "Software 類別中各產品的銷售額分別是多少？",
    }
    for cat, question in cat_product_questions.items():
        prod_rev = (
            df[df["Product Category"] == cat]
            .groupby("Product Name")["Total Revenue"]
            .sum()
            .sort_values(ascending=False)
        )
        eval_set.append(
            {
                "question": question,
                "ground_truth": prod_rev.round(2).to_dict(),
                "eval_type": "dataframe",
                "difficulty": "medium",
                "tags": ["revenue", "product", "bar", "plot"],
                "expected_plot": {
                    "should_plot": True,
                    "chart_type": "bar",
                    "expected_x_order": prod_rev.index.tolist(),
                    "expected_traces": 1,
                },
            }
        )

    # N12-N13: 各年度各季度銷售額
    for year in [2023, 2024]:
        q_rev = (
            df[df["year"] == year]
            .groupby("quarter")["Total Revenue"]
            .sum()
        )
        q_labels = [f"Q{q}" for q in q_rev.index.tolist()]
        q_dict = {f"Q{q}": round(v, 2) for q, v in q_rev.items()}
        eval_set.append(
            {
                "question": f"{year} 年各季度的銷售額分別是多少？",
                "ground_truth": q_dict,
                "eval_type": "dataframe",
                "difficulty": "medium",
                "tags": ["quarterly", "revenue", "bar", "plot"],
                "expected_plot": {
                    "should_plot": True,
                    "chart_type": "bar",
                    "expected_x_order": q_labels,
                    "expected_traces": 1,
                },
            }
        )

    # N14: 銷售額前 5 名產品
    top5_rev = (
        df.groupby("Product Name")["Total Revenue"].sum().nlargest(5)
    )
    eval_set.append(
        {
            "question": "銷售額前五名的產品分別是哪些？請用圖表呈現。",
            "ground_truth": top5_rev.round(2).to_dict(),
            "eval_type": "dataframe",
            "difficulty": "medium",
            "tags": ["top_products", "revenue", "bar", "plot"],
            "expected_plot": {
                "should_plot": True,
                "chart_type": "bar",
                "expected_x_order": top5_rev.index.tolist(),
                "expected_traces": 1,
            },
        }
    )

    # N15: 銷售量前 5 名產品
    top5_qty = (
        df.groupby("Product Name")["Quantity"].sum().nlargest(5)
    )
    eval_set.append(
        {
            "question": "銷售量前五名的產品是哪些？",
            "ground_truth": top5_qty.to_dict(),
            "eval_type": "dataframe",
            "difficulty": "medium",
            "tags": ["top_products", "quantity", "bar", "plot"],
            "expected_plot": {
                "should_plot": True,
                "chart_type": "bar",
                "expected_x_order": top5_qty.index.tolist(),
                "expected_traces": 1,
            },
        }
    )

    # --- O. Line Chart 題組 (~11 題) ---

    # Helper: generate month labels for a date range
    def month_labels_for_range(data):
        grouped = (
            data.groupby(["year", "month"])["Total Revenue"]
            .sum()
            .reset_index()
            .sort_values(["year", "month"])
        )
        return [
            f"{int(y)}-{int(m):02d}"
            for y, m in zip(grouped["year"], grouped["month"])
        ]

    # O1-O3: 各類別月銷售趨勢 2023-2024
    for category in df["Product Category"].unique():
        cat_data = df[df["Product Category"] == category]
        m_labels = month_labels_for_range(cat_data)
        monthly = (
            cat_data.groupby(["year", "month"])["Total Revenue"]
            .sum()
            .reset_index()
            .sort_values(["year", "month"])
        )
        ans_dict = {
            f"{int(y)}-{int(m):02d}": round(v, 2)
            for y, m, v in zip(monthly["year"], monthly["month"], monthly["Total Revenue"])
        }
        eval_set.append(
            {
                "question": f"請呈現 {category} 類別在 2023 至 2024 年的月銷售額趨勢。",
                "ground_truth": ans_dict,
                "eval_type": "dataframe",
                "difficulty": "hard",
                "tags": ["trend", "monthly", "category", "line", "plot"],
                "expected_plot": {
                    "should_plot": True,
                    "chart_type": "line",
                    "expected_x_order": m_labels,
                    "expected_traces": 1,
                },
            }
        )

    # O4: 全公司季度銷售趨勢 2023-2024
    quarterly = (
        df.groupby(["year", "quarter"])["Total Revenue"]
        .sum()
        .reset_index()
        .sort_values(["year", "quarter"])
    )
    q_labels = [
        f"{int(y)} Q{int(q)}"
        for y, q in zip(quarterly["year"], quarterly["quarter"])
    ]
    q_ans = {
        f"{int(y)} Q{int(q)}": round(v, 2)
        for y, q, v in zip(quarterly["year"], quarterly["quarter"], quarterly["Total Revenue"])
    }
    eval_set.append(
        {
            "question": "請呈現 2023 至 2024 年各季度的總銷售額趨勢。",
            "ground_truth": q_ans,
            "eval_type": "dataframe",
            "difficulty": "medium",
            "tags": ["trend", "quarterly", "line", "plot"],
            "expected_plot": {
                "should_plot": True,
                "chart_type": "line",
                "expected_x_order": q_labels,
                "expected_traces": 1,
            },
        }
    )

    # O5-O8: 各區域季度銷售趨勢
    for region in df["Region"].unique():
        r_data = df[df["Region"] == region]
        r_quarterly = (
            r_data.groupby(["year", "quarter"])["Total Revenue"]
            .sum()
            .reset_index()
            .sort_values(["year", "quarter"])
        )
        rq_labels = [
            f"{int(y)} Q{int(q)}"
            for y, q in zip(r_quarterly["year"], r_quarterly["quarter"])
        ]
        rq_ans = {
            f"{int(y)} Q{int(q)}": round(v, 2)
            for y, q, v in zip(
                r_quarterly["year"], r_quarterly["quarter"], r_quarterly["Total Revenue"]
            )
        }
        eval_set.append(
            {
                "question": f"請呈現 {region} 區域 2023 至 2024 年各季度的銷售額趨勢。",
                "ground_truth": rq_ans,
                "eval_type": "dataframe",
                "difficulty": "hard",
                "tags": ["trend", "quarterly", "region", "line", "plot"],
                "expected_plot": {
                    "should_plot": True,
                    "chart_type": "line",
                    "expected_x_order": rq_labels,
                    "expected_traces": 1,
                },
            }
        )

    # O9-O11: 特定國家月銷售趨勢
    country_line_questions = {
        "United States": "請呈現美國 (United States) 2023 至 2024 年的月銷售額趨勢。",
        "China": "請呈現中國 (China) 2023 至 2024 年的月銷售額趨勢。",
        "Germany": "請呈現德國 (Germany) 2023 至 2024 年的月銷售額趨勢。",
    }
    for country, question in country_line_questions.items():
        c_data = df[df["Country"] == country]
        c_monthly = (
            c_data.groupby(["year", "month"])["Total Revenue"]
            .sum()
            .reset_index()
            .sort_values(["year", "month"])
        )
        cm_labels = [
            f"{int(y)}-{int(m):02d}"
            for y, m in zip(c_monthly["year"], c_monthly["month"])
        ]
        cm_ans = {
            f"{int(y)}-{int(m):02d}": round(v, 2)
            for y, m, v in zip(
                c_monthly["year"], c_monthly["month"], c_monthly["Total Revenue"]
            )
        }
        eval_set.append(
            {
                "question": question,
                "ground_truth": cm_ans,
                "eval_type": "dataframe",
                "difficulty": "hard",
                "tags": ["trend", "monthly", "country", "line", "plot"],
                "expected_plot": {
                    "should_plot": True,
                    "chart_type": "line",
                    "expected_x_order": cm_labels,
                    "expected_traces": 1,
                },
            }
        )

    # --- P. Pie Chart 題組 (~8 題) ---

    # P1: 各區域銷售額占比
    region_share = (
        df.groupby("Region")["Total Revenue"].sum().sort_values(ascending=False)
    )
    eval_set.append(
        {
            "question": "各區域的銷售額占比分別是多少？請用圓餅圖呈現。",
            "ground_truth": {
                k: round(v / region_share.sum() * 100, 2)
                for k, v in region_share.items()
            },
            "eval_type": "dataframe",
            "difficulty": "medium",
            "tags": ["share", "region", "pie", "plot"],
            "expected_plot": {
                "should_plot": True,
                "chart_type": "pie",
                "expected_x_order": region_share.index.tolist(),
                "expected_traces": 1,
            },
        }
    )

    # P2: 各產品類別銷售額占比
    cat_share = (
        df.groupby("Product Category")["Total Revenue"]
        .sum()
        .sort_values(ascending=False)
    )
    eval_set.append(
        {
            "question": "各產品類別的銷售額占比分別是多少？請用圓餅圖表示。",
            "ground_truth": {
                k: round(v / cat_share.sum() * 100, 2)
                for k, v in cat_share.items()
            },
            "eval_type": "dataframe",
            "difficulty": "medium",
            "tags": ["share", "category", "pie", "plot"],
            "expected_plot": {
                "should_plot": True,
                "chart_type": "pie",
                "expected_x_order": cat_share.index.tolist(),
                "expected_traces": 1,
            },
        }
    )

    # P3: 各區域訂單數占比
    region_orders = (
        df.groupby("Region")["Order ID"].count().sort_values(ascending=False)
    )
    eval_set.append(
        {
            "question": "各區域的訂單數量占比為何？",
            "ground_truth": {
                k: round(v / region_orders.sum() * 100, 2)
                for k, v in region_orders.items()
            },
            "eval_type": "dataframe",
            "difficulty": "medium",
            "tags": ["share", "region", "orders", "pie", "plot"],
            "expected_plot": {
                "should_plot": True,
                "chart_type": "pie",
                "expected_x_order": region_orders.index.tolist(),
                "expected_traces": 1,
            },
        }
    )

    # P4-P5: 特定年份各類別占比
    for year in [2023, 2024]:
        yr_cat = (
            df[df["year"] == year]
            .groupby("Product Category")["Total Revenue"]
            .sum()
            .sort_values(ascending=False)
        )
        eval_set.append(
            {
                "question": f"{year} 年各產品類別的銷售額占比分別是多少？",
                "ground_truth": {
                    k: round(v / yr_cat.sum() * 100, 2)
                    for k, v in yr_cat.items()
                },
                "eval_type": "dataframe",
                "difficulty": "medium",
                "tags": ["share", "category", "yearly", "pie", "plot"],
                "expected_plot": {
                    "should_plot": True,
                    "chart_type": "pie",
                    "expected_x_order": yr_cat.index.tolist(),
                    "expected_traces": 1,
                },
            }
        )

    # P6-P8: 區域內國家占比
    region_pie_questions = {
        "APAC": "APAC 區域中各國家的銷售額占比分別是多少？",
        "EMEA": "EMEA 區域中各國家的銷售額占比為何？",
        "North America": "北美區域各國的銷售額占比如何？",
    }
    for region, question in region_pie_questions.items():
        rc_share = (
            df[df["Region"] == region]
            .groupby("Country")["Total Revenue"]
            .sum()
            .sort_values(ascending=False)
        )
        eval_set.append(
            {
                "question": question,
                "ground_truth": {
                    k: round(v / rc_share.sum() * 100, 2)
                    for k, v in rc_share.items()
                },
                "eval_type": "dataframe",
                "difficulty": "medium",
                "tags": ["share", "country", "region", "pie", "plot"],
                "expected_plot": {
                    "should_plot": True,
                    "chart_type": "pie",
                    "expected_x_order": rc_share.index.tolist(),
                    "expected_traces": 1,
                },
            }
        )

    # --- Q. No-Plot 反例題組 (~16 題) ---

    # Q1-Q4: 單一聚合值
    total_2024 = round(df[df["year"] == 2024]["Total Revenue"].sum(), 2)
    eval_set.append(
        {
            "question": "2024 年的總銷售額是多少？",
            "ground_truth": total_2024,
            "eval_type": "number",
            "difficulty": "simple",
            "tags": ["revenue", "yearly", "no_plot"],
            "expected_plot": {"should_plot": False},
        }
    )

    elec_rev = round(
        df[df["Product Category"] == "Electronics"]["Total Revenue"].sum(), 2
    )
    eval_set.append(
        {
            "question": "Electronics 類別的總銷售額是多少？",
            "ground_truth": elec_rev,
            "eval_type": "number",
            "difficulty": "simple",
            "tags": ["revenue", "category", "no_plot"],
            "expected_plot": {"should_plot": False},
        }
    )

    total_2023 = round(df[df["year"] == 2023]["Total Revenue"].sum(), 2)
    eval_set.append(
        {
            "question": "2023 年的總銷售額有多少？",
            "ground_truth": total_2023,
            "eval_type": "number",
            "difficulty": "simple",
            "tags": ["revenue", "yearly", "no_plot"],
            "expected_plot": {"should_plot": False},
        }
    )

    sw_rev = round(
        df[df["Product Category"] == "Software"]["Total Revenue"].sum(), 2
    )
    eval_set.append(
        {
            "question": "Software 類別的總銷售額為何？",
            "ground_truth": sw_rev,
            "eval_type": "number",
            "difficulty": "simple",
            "tags": ["revenue", "category", "no_plot"],
            "expected_plot": {"should_plot": False},
        }
    )

    # Q5-Q8: Top-1 查詢
    top_region = (
        df.groupby("Region")["Total Revenue"].sum().idxmax()
    )
    eval_set.append(
        {
            "question": "銷售額最高的區域是哪一個？",
            "ground_truth": top_region,
            "eval_type": "string",
            "difficulty": "simple",
            "tags": ["top", "region", "no_plot"],
            "expected_plot": {"should_plot": False},
        }
    )

    top_product = (
        df.groupby("Product Name")["Total Revenue"].sum().idxmax()
    )
    eval_set.append(
        {
            "question": "銷售額最高的產品是什麼？",
            "ground_truth": top_product,
            "eval_type": "string",
            "difficulty": "simple",
            "tags": ["top", "product", "no_plot"],
            "expected_plot": {"should_plot": False},
        }
    )

    top_order_customer = (
        df.groupby("Customer Name")["Order ID"].count().idxmax()
    )
    eval_set.append(
        {
            "question": "訂單數量最多的客戶是誰？",
            "ground_truth": top_order_customer,
            "eval_type": "string",
            "difficulty": "simple",
            "tags": ["top", "customer", "no_plot"],
            "expected_plot": {"should_plot": False},
        }
    )

    top_country = (
        df.groupby("Country")["Total Revenue"].sum().idxmax()
    )
    eval_set.append(
        {
            "question": "銷售額最高的國家是哪一個？",
            "ground_truth": top_country,
            "eval_type": "string",
            "difficulty": "simple",
            "tags": ["top", "country", "no_plot"],
            "expected_plot": {"should_plot": False},
        }
    )

    # Q9-Q12: 特定值查詢
    laptop_asp = round(
        df[df["Product Name"] == "Enterprise Laptop X1"]["ASP"].mean(), 2
    )
    eval_set.append(
        {
            "question": "Enterprise Laptop X1 的平均單價 (ASP) 是多少？",
            "ground_truth": laptop_asp,
            "eval_type": "number",
            "difficulty": "simple",
            "tags": ["asp", "product", "no_plot"],
            "expected_plot": {"should_plot": False},
        }
    )

    us_orders = int(
        df[df["Country"] == "United States"]["Order ID"].count()
    )
    eval_set.append(
        {
            "question": "美國 (United States) 總共有多少筆訂單？",
            "ground_truth": us_orders,
            "eval_type": "number",
            "difficulty": "simple",
            "tags": ["orders", "country", "no_plot"],
            "expected_plot": {"should_plot": False},
        }
    )

    japan_qty = int(
        df[df["Country"] == "Japan"]["Quantity"].sum()
    )
    eval_set.append(
        {
            "question": "日本 (Japan) 的總銷售量是多少？",
            "ground_truth": japan_qty,
            "eval_type": "number",
            "difficulty": "simple",
            "tags": ["quantity", "country", "no_plot"],
            "expected_plot": {"should_plot": False},
        }
    )

    cloud_rev = round(
        df[df["Product Name"] == "Cloud License (Annual)"]["Total Revenue"].sum(), 2
    )
    eval_set.append(
        {
            "question": "Cloud License (Annual) 的總銷售額是多少？",
            "ground_truth": cloud_rev,
            "eval_type": "number",
            "difficulty": "simple",
            "tags": ["revenue", "product", "no_plot"],
            "expected_plot": {"should_plot": False},
        }
    )

    # Q13-Q16: YoY / 成長率 (結果只有 1 個數字)
    total_2023_all = df[df["year"] == 2023]["Total Revenue"].sum()
    total_2024_all = df[df["year"] == 2024]["Total Revenue"].sum()
    total_yoy = round((total_2024_all - total_2023_all) / total_2023_all * 100, 2)
    eval_set.append(
        {
            "question": "2024 年相較 2023 年的總銷售額年增率是多少？(以百分比表示)",
            "ground_truth": total_yoy,
            "eval_type": "number",
            "difficulty": "hard",
            "tags": ["yoy", "growth", "no_plot"],
            "expected_plot": {"should_plot": False},
        }
    )

    na_2023 = df[(df["Region"] == "North America") & (df["year"] == 2023)][
        "Total Revenue"
    ].sum()
    na_2024 = df[(df["Region"] == "North America") & (df["year"] == 2024)][
        "Total Revenue"
    ].sum()
    na_yoy = round((na_2024 - na_2023) / na_2023 * 100, 2)
    eval_set.append(
        {
            "question": "North America 區域 2024 年相較 2023 年的銷售額年增率為何？",
            "ground_truth": na_yoy,
            "eval_type": "number",
            "difficulty": "hard",
            "tags": ["yoy", "growth", "region", "no_plot"],
            "expected_plot": {"should_plot": False},
        }
    )

    apac_2023 = df[(df["Region"] == "APAC") & (df["year"] == 2023)][
        "Total Revenue"
    ].sum()
    apac_2024 = df[(df["Region"] == "APAC") & (df["year"] == 2024)][
        "Total Revenue"
    ].sum()
    apac_yoy = round((apac_2024 - apac_2023) / apac_2023 * 100, 2)
    eval_set.append(
        {
            "question": "APAC 區域 2024 年相較 2023 年的銷售額成長率是多少百分比？",
            "ground_truth": apac_yoy,
            "eval_type": "number",
            "difficulty": "hard",
            "tags": ["yoy", "growth", "region", "no_plot"],
            "expected_plot": {"should_plot": False},
        }
    )

    emea_2023 = df[(df["Region"] == "EMEA") & (df["year"] == 2023)][
        "Total Revenue"
    ].sum()
    emea_2024 = df[(df["Region"] == "EMEA") & (df["year"] == 2024)][
        "Total Revenue"
    ].sum()
    emea_yoy = round((emea_2024 - emea_2023) / emea_2023 * 100, 2)
    eval_set.append(
        {
            "question": "EMEA 區域 2024 年對比 2023 年的銷售額年增率是多少？",
            "ground_truth": emea_yoy,
            "eval_type": "number",
            "difficulty": "hard",
            "tags": ["yoy", "growth", "region", "no_plot"],
            "expected_plot": {"should_plot": False},
        }
    )

    return eval_set


if __name__ == "__main__":
    eval_set = generate_questions()

    output_file = DATA_PATH / "eval_set_v3.jsonl"
    os.makedirs(DATA_PATH, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        for entry in eval_set:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # Print statistics
    plot_q = [e for e in eval_set if "expected_plot" in e]
    plot_yes = [e for e in plot_q if e["expected_plot"].get("should_plot")]
    plot_no = [e for e in plot_q if not e["expected_plot"].get("should_plot")]

    print(f"成功產生 {len(eval_set)} 筆測試案例於 {output_file}")
    print(f"  Plot 題: {len(plot_q)} (should_plot=True: {len(plot_yes)}, no-plot: {len(plot_no)})")
    for ct in ["bar", "line", "pie"]:
        n = sum(1 for e in plot_yes if e["expected_plot"].get("chart_type") == ct)
        print(f"    {ct}: {n}")
