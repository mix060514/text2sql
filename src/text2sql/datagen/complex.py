"""Complex-query golden seeds: CTEs, window functions, cross-year comparisons, cohorts.

Each template carries its SQL and an independent pandas implementation on the source CSV
(not the SQLite file). A seed is accepted only when both produce the same rows in the same
order. Ties are broken by metric DESC, name ASC in both implementations.
"""
import math
import re
from functools import lru_cache

import pandas as pd

from text2sql.agents.planned_agent import tools

from .templates import MONTH, PROJECT, QTR, REV, YEAR, _entities, load_legacy

CSV = PROJECT / "data" / "global_sales_data_logic_distribution.csv"
R = "Total Revenue"
BOUNDARY = "只能換說法；必須問同一個指標、分組、名次、時間範圍、門檻與計算方式，不得新增或刪除條件。"
WEEKDAYS = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]


@lru_cache(maxsize=1)
def frame() -> pd.DataFrame:
    df = pd.read_csv(CSV)
    date = pd.to_datetime(df["Order Date"])
    df["year"] = date.dt.year.astype(str)
    df["q"] = date.dt.quarter
    df["ym"] = date.dt.strftime("%Y-%m")
    df["mm"] = date.dt.month
    df["wd"] = date.dt.dayofweek  # Monday = 0
    return df


def _rows(table: pd.DataFrame) -> list[list]:
    return [[None if (isinstance(v, float) and math.isnan(v)) else (v.item() if hasattr(v, "item") else v)
             for v in row] for row in table.itertuples(index=False)]


def _top(table, group, metric, name, n=1):
    """Top-n per group by metric DESC, name ASC."""
    ordered = table.sort_values([group, metric, name], ascending=[True, False, True])
    return ordered.groupby(group, sort=True).head(n)


def _yoy(table, keys):
    pivot = table.pivot_table(index=keys, columns="year", values=R, aggfunc="sum")
    return (pivot["2024"] - pivot["2023"]) * 100.0 / pivot["2023"]


# ---------- pandas references (one per template) ----------

def p_region_top_product(df, year):
    t = df[df.year == year].groupby(["Region", "Product Name"], as_index=False)[R].sum()
    t = _top(t, "Region", R, "Product Name")
    return _rows(t.assign(v=t[R].round(2))[["Region", "Product Name", "v"]])


def p_category_top3_quantity(df):
    t = df.groupby(["Product Category", "Product Name"], as_index=False)["Quantity"].sum()
    t = _top(t, "Product Category", "Quantity", "Product Name", 3)
    t["rank"] = t.groupby("Product Category").cumcount() + 1
    return _rows(t[["Product Category", "rank", "Product Name", "Quantity"]])


def p_region_mom(df, region, year):
    # January compares with the previous December (run 20261107-153104: the SQL judge flagged a
    # within-year LAG that left January empty although December data exists).
    t = df[df.Region == region].groupby("ym")[R].sum().sort_index()
    growth = ((t - t.shift(1)) * 100.0 / t.shift(1)).round(2)
    keep = [m.startswith(year) for m in t.index]
    return _rows(pd.DataFrame({"m": t.index[keep], "g": growth.values[keep]}))


def p_cumulative(df, year):
    t = df[df.year == year].groupby("ym")[R].sum().sort_index().cumsum()
    return _rows(pd.DataFrame({"m": t.index, "c": t.round(2).values}))


def p_country_decline(df):
    yoy = _yoy(df, ["Country"]).round(2)
    t = yoy[yoy < 0].reset_index(name="y").sort_values(["y", "Country"])
    return _rows(t[["Country", "y"]])


def p_all_categories(df, year):
    n = df[df.year == year].groupby("Customer Name")["Product Category"].nunique()
    return [[int((n == 3).sum())]]


def p_retained(df):
    years = df.groupby("Customer Name")["year"].nunique()
    return [[int((years == 2).sum())]]


def p_new_customers(df, year):
    first = df.groupby("Customer Name")["year"].min()
    names = sorted(first[first == year].index)
    return [[name] for name in names]


def p_aov_region(df):
    t = df.groupby("Region").agg(rev=(R, "sum"), n=("Order ID", "count"))
    t["aov"] = t.rev / t.n
    t = t.reset_index().sort_values(["aov", "Region"], ascending=[False, True])
    return _rows(t.assign(aov=t.aov.round(2))[["Region", "aov"]])


def p_region_share_of_category(df, year, category):
    t = df[(df.year == year) & (df["Product Category"] == category)].groupby("Region", as_index=False)[R].sum()
    t["s"] = t[R] * 100.0 / t[R].sum()
    t = t.sort_values(["s", "Region"], ascending=[False, True])
    return _rows(t.assign(s=t.s.round(2))[["Region", "s"]])


def p_region_yoy(df):
    t = _yoy(df, ["Region"]).reset_index(name="y").sort_values(["y", "Region"], ascending=[False, True])
    return _rows(t.assign(y=t.y.round(2))[["Region", "y"]])


def p_top_customer_concentration(df, n):
    t = df.groupby("Customer Name")[R].sum().reset_index().sort_values([R, "Customer Name"], ascending=[False, True])
    return [[round(float(t.head(n)[R].sum() * 100.0 / df[R].sum()), 2)]]


def p_large_order_share(df, qty):
    t = df.assign(big=df[R].where(df.Quantity >= qty, 0)).groupby("Region")[["big", R]].sum()
    t["s"] = t.big * 100.0 / t[R]
    t = t.reset_index().sort_values(["s", "Region"], ascending=[False, True])
    return _rows(t.assign(s=t.s.round(2))[["Region", "s"]])


def p_best_month_each_year(df):
    t = df.groupby(["year", "ym"], as_index=False)[R].sum()
    t = _top(t, "year", R, "ym")
    return _rows(t.assign(v=t[R].round(2))[["year", "ym", "v"]])


def p_asp_change(df, n):
    pivot = df.pivot_table(index="Product Name", columns="year", values="ASP", aggfunc="mean")
    t = (pivot["2024"] - pivot["2023"]).reset_index(name="d").sort_values(["d", "Product Name"], ascending=[False, True])
    return _rows(t.head(n).assign(d=lambda x: x.d.round(2))[["Product Name", "d"]])


def p_pareto(df, pct):
    t = df.groupby("Product Name")[R].sum().reset_index().sort_values([R, "Product Name"], ascending=[False, True])
    share = (t[R].cumsum() * 100.0 / t[R].sum()).reset_index(drop=True)
    return [[int(share[share >= pct].index[0]) + 1]]


def p_multi_region_customers(df, k):
    n = df.groupby("Customer Name")["Region"].nunique()
    return [[int((n >= k).sum())]]


def p_region_category_matrix(df, year):
    t = df[df.year == year].groupby(["Region", "Product Category"], as_index=False)[R].sum()
    t = t.sort_values(["Region", "Product Category"])
    return _rows(t.assign(v=t[R].round(2))[["Region", "Product Category", "v"]])


def p_category_best_month(df, year):
    t = df[df.year == year].groupby(["Product Category", "ym"], as_index=False)[R].sum()
    t = _top(t, "Product Category", R, "ym")
    return _rows(t.assign(v=t[R].round(2))[["Product Category", "ym", "v"]])


def p_category_quarter_yoy(df, category):
    sub = df[df["Product Category"] == category]
    yoy = _yoy(sub, ["q"]).reset_index(name="y").sort_values("q")
    return _rows(pd.DataFrame({"q": "Q" + yoy.q.astype(str), "y": yoy.y.round(2)}))


def p_weekday(df):
    t = df.groupby("wd")[R].sum().sort_index()
    return _rows(pd.DataFrame({"d": [WEEKDAYS[i] for i in t.index], "v": t.round(2).values}))


def p_region_top_customers_orders(df, region, year, n):
    sub = df[(df.Region == region) & (df.year == year)]
    t = sub.groupby("Customer Name").agg(rev=(R, "sum"), n=("Order ID", "count")).reset_index()
    t = t.sort_values(["rev", "Customer Name"], ascending=[False, True]).head(n)
    return _rows(t.assign(rev=t.rev.round(2))[["Customer Name", "rev", "n"]])


def p_avg_qty_per_order(df):
    t = df.groupby("Product Category")["Quantity"].mean().reset_index()
    t = t.sort_values(["Quantity", "Product Category"], ascending=[False, True])
    return _rows(t.assign(Quantity=t.Quantity.round(2)))


def p_countries_above_avg(df):
    t = df.groupby("Country")[R].sum()
    above = t[t > t.mean()].reset_index().sort_values([R, "Country"], ascending=[False, True])
    return _rows(above.assign(v=above[R].round(2))[["Country", "v"]])


def p_h2_vs_h1(df, year):
    sub = df[df.year == year].assign(half=lambda x: (x.mm > 6).map({False: "h1", True: "h2"}))
    t = sub.pivot_table(index="Region", columns="half", values=R, aggfunc="sum")
    t["g"] = (t.h2 - t.h1) * 100.0 / t.h1
    t = t.reset_index().sort_values(["g", "Region"], ascending=[False, True])
    return _rows(t.assign(g=t.g.round(2))[["Region", "g"]])


def p_country_rank_in_region(df, country):
    region = df.loc[df.Country == country, "Region"].iloc[0]
    t = df[df.Region == region].groupby("Country")[R].sum().reset_index()
    t = t.sort_values([R, "Country"], ascending=[False, True]).reset_index(drop=True)
    return [[int(t.index[t.Country == country][0]) + 1]]


def p_nth_customer(df, year, n):
    t = df[df.year == year].groupby("Customer Name")[R].sum().reset_index()
    t = t.sort_values([R, "Customer Name"], ascending=[False, True]).reset_index(drop=True)
    return [[t.loc[n - 1, "Customer Name"]]]


def p_months_beat_last_year(df):
    t = df.pivot_table(index="mm", columns="year", values=R, aggfunc="sum")
    months = [f"{m:02d}" for m in t.index[t["2024"] > t["2023"]]]
    return [[m] for m in months]


def p_high_price_orders(df, price):
    t = df[df.ASP > price].groupby("Product Category").size().reset_index(name="n")
    return _rows(t.sort_values(["n", "Product Category"], ascending=[False, True]))


def p_revenue_per_customer_region(df):
    t = df.groupby("Region").agg(rev=(R, "sum"), c=("Customer Name", "nunique"))
    t["v"] = t.rev / t.c
    t = t.reset_index().sort_values(["v", "Region"], ascending=[False, True])
    return _rows(t.assign(v=t.v.round(2))[["Region", "v"]])


def p_country_top_quantity_product(df, region):
    t = df[df.Region == region].groupby(["Country", "Product Name"], as_index=False)["Quantity"].sum()
    t = _top(t, "Country", "Quantity", "Product Name")
    return _rows(t[["Country", "Product Name", "Quantity"]])


def p_product_growth_amount(df, n):
    pivot = df.pivot_table(index="Product Name", columns="year", values=R, aggfunc="sum")
    t = (pivot["2024"] - pivot["2023"]).reset_index(name="d").sort_values(["d", "Product Name"], ascending=[False, True])
    return _rows(t.head(n).assign(d=lambda x: x.d.round(2))[["Product Name", "d"]])


def p_customer_category_mix(df, customer):
    t = df[df["Customer Name"] == customer].groupby("Product Category", as_index=False)[R].sum()
    t["s"] = t[R] * 100.0 / t[R].sum()
    t = t.sort_values(["s", "Product Category"], ascending=[False, True])
    return _rows(t.assign(s=t.s.round(2))[["Product Category", "s"]])


def p_quarter_max_qoq(df):
    t = df.assign(k=df.year + "-Q" + df.q.astype(str)).groupby("k")[R].sum().sort_index()
    g = ((t - t.shift(1)) * 100.0 / t.shift(1)).dropna().reset_index(name="g")
    g = g.sort_values(["g", "k"], ascending=[False, True]).head(1)
    return _rows(g.assign(g=g.g.round(2)))


# ---------- SQL ----------

SQL = {
    "region_top_product": f"""WITH t AS (SELECT Region AS region, "Product Name" AS product_name, SUM({REV}) AS revenue
FROM sales_data WHERE {YEAR} = '{{year}}' GROUP BY 1, 2), r AS (SELECT *, ROW_NUMBER() OVER (PARTITION BY region
ORDER BY revenue DESC, product_name ASC) AS rn FROM t)
SELECT region, product_name, ROUND(revenue, 2) AS total_revenue FROM r WHERE rn = 1 ORDER BY region""",
    "category_top3_quantity": """WITH t AS (SELECT "Product Category" AS product_category, "Product Name" AS product_name,
SUM(Quantity) AS total_quantity FROM sales_data GROUP BY 1, 2), r AS (SELECT *, ROW_NUMBER() OVER (PARTITION BY
product_category ORDER BY total_quantity DESC, product_name ASC) AS rank FROM t)
SELECT product_category, rank, product_name, total_quantity FROM r WHERE rank <= 3 ORDER BY product_category, rank""",
    "region_mom": f"""WITH m AS (SELECT {MONTH} AS month, SUM({REV}) AS revenue FROM sales_data
WHERE Region = '{{region}}' GROUP BY month),
g AS (SELECT month, revenue, LAG(revenue) OVER (ORDER BY month) AS previous_month_revenue FROM m)
SELECT month, ROUND((revenue - previous_month_revenue) * 100.0 / previous_month_revenue, 2) AS mom_growth_pct
FROM g WHERE substr(month, 1, 4) = '{{year}}' ORDER BY month""",
    "cumulative": f"""WITH m AS (SELECT {MONTH} AS month, SUM({REV}) AS revenue FROM sales_data WHERE {YEAR} = '{{year}}'
GROUP BY month) SELECT month, ROUND(SUM(revenue) OVER (ORDER BY month), 2) AS cumulative_revenue FROM m ORDER BY month""",
    "country_decline": f"""WITH a AS (SELECT Country AS country, SUM(CASE WHEN {YEAR} = '2023' THEN {REV} END) AS r23,
SUM(CASE WHEN {YEAR} = '2024' THEN {REV} END) AS r24 FROM sales_data GROUP BY Country)
SELECT country, ROUND((r24 - r23) * 100.0 / r23, 2) AS yoy_pct FROM a WHERE r24 < r23 ORDER BY yoy_pct ASC, country ASC""",
    "all_categories": f"""SELECT COUNT(*) AS customer_count FROM (SELECT "Customer Name" FROM sales_data WHERE {YEAR} = '{{year}}'
GROUP BY "Customer Name" HAVING COUNT(DISTINCT "Product Category") = 3)""",
    "retained": f"""SELECT COUNT(*) AS customer_count FROM (SELECT "Customer Name" FROM sales_data
GROUP BY "Customer Name" HAVING SUM({YEAR} = '2023') > 0 AND SUM({YEAR} = '2024') > 0)""",
    "new_customers": f"""SELECT "Customer Name" AS customer_name FROM sales_data GROUP BY "Customer Name"
HAVING MIN({YEAR}) = '{{year}}' ORDER BY customer_name""",
    "aov_region": f"""SELECT Region AS region, ROUND(SUM({REV}) / COUNT("Order ID"), 2) AS avg_order_value FROM sales_data
GROUP BY Region ORDER BY SUM({REV}) / COUNT("Order ID") DESC, region ASC""",
    "region_share_of_category": f"""SELECT Region AS region, ROUND(SUM({REV}) * 100.0 / SUM(SUM({REV})) OVER (), 2) AS share_pct
FROM sales_data WHERE {YEAR} = '{{year}}' AND "Product Category" = '{{category}}' GROUP BY Region
ORDER BY SUM({REV}) DESC, region ASC""",
    "region_yoy": f"""WITH a AS (SELECT Region AS region, SUM(CASE WHEN {YEAR} = '2023' THEN {REV} END) AS r23,
SUM(CASE WHEN {YEAR} = '2024' THEN {REV} END) AS r24 FROM sales_data GROUP BY Region)
SELECT region, ROUND((r24 - r23) * 100.0 / r23, 2) AS yoy_pct FROM a ORDER BY (r24 - r23) / r23 DESC, region ASC""",
    "top_customer_concentration": f"""WITH c AS (SELECT "Customer Name" AS customer_name, SUM({REV}) AS revenue FROM sales_data
GROUP BY 1 ORDER BY revenue DESC, customer_name ASC LIMIT {{n}})
SELECT ROUND((SELECT SUM(revenue) FROM c) * 100.0 / (SELECT SUM({REV}) FROM sales_data), 2) AS share_pct""",
    "large_order_share": f"""SELECT Region AS region, ROUND(SUM(CASE WHEN Quantity >= {{qty}} THEN {REV} ELSE 0 END) * 100.0
/ SUM({REV}), 2) AS share_pct FROM sales_data GROUP BY Region
ORDER BY SUM(CASE WHEN Quantity >= {{qty}} THEN {REV} ELSE 0 END) / SUM({REV}) DESC, region ASC""",
    "best_month_each_year": f"""WITH m AS (SELECT {YEAR} AS year, {MONTH} AS month, SUM({REV}) AS revenue FROM sales_data
GROUP BY 1, 2), r AS (SELECT *, ROW_NUMBER() OVER (PARTITION BY year ORDER BY revenue DESC, month ASC) AS rn FROM m)
SELECT year, month, ROUND(revenue, 2) AS total_revenue FROM r WHERE rn = 1 ORDER BY year""",
    "asp_change": f"""WITH a AS (SELECT "Product Name" AS product_name, AVG(CASE WHEN {YEAR} = '2023' THEN ASP END) AS a23,
AVG(CASE WHEN {YEAR} = '2024' THEN ASP END) AS a24 FROM sales_data GROUP BY 1)
SELECT product_name, ROUND(a24 - a23, 2) AS asp_change FROM a ORDER BY a24 - a23 DESC, product_name ASC LIMIT {{n}}""",
    "pareto": f"""WITH p AS (SELECT "Product Name" AS product_name, SUM({REV}) AS revenue FROM sales_data GROUP BY 1),
c AS (SELECT ROW_NUMBER() OVER (ORDER BY revenue DESC, product_name ASC) AS product_count,
SUM(revenue) OVER (ORDER BY revenue DESC, product_name ASC) * 100.0 / SUM(revenue) OVER () AS cumulative_share_pct FROM p)
SELECT MIN(product_count) AS product_count FROM c WHERE cumulative_share_pct >= {{pct}}""",
    "multi_region_customers": """SELECT COUNT(*) AS customer_count FROM (SELECT "Customer Name" FROM sales_data
GROUP BY "Customer Name" HAVING COUNT(DISTINCT Region) >= {k})""",
    "region_category_matrix": f"""SELECT Region AS region, "Product Category" AS product_category, ROUND(SUM({REV}), 2)
AS total_revenue FROM sales_data WHERE {YEAR} = '{{year}}' GROUP BY 1, 2 ORDER BY region, product_category""",
    "category_best_month": f"""WITH m AS (SELECT "Product Category" AS product_category, {MONTH} AS month, SUM({REV}) AS revenue
FROM sales_data WHERE {YEAR} = '{{year}}' GROUP BY 1, 2), r AS (SELECT *, ROW_NUMBER() OVER (PARTITION BY product_category
ORDER BY revenue DESC, month ASC) AS rn FROM m)
SELECT product_category, month, ROUND(revenue, 2) AS total_revenue FROM r WHERE rn = 1 ORDER BY product_category""",
    "category_quarter_yoy": f"""WITH q AS (SELECT {QTR} AS qn, SUM(CASE WHEN {YEAR} = '2023' THEN {REV} END) AS r23,
SUM(CASE WHEN {YEAR} = '2024' THEN {REV} END) AS r24 FROM sales_data WHERE "Product Category" = '{{category}}' GROUP BY qn)
SELECT 'Q' || qn AS quarter, ROUND((r24 - r23) * 100.0 / r23, 2) AS yoy_pct FROM q ORDER BY qn""",
    "weekday": f"""SELECT CASE CAST(strftime('%w', "Order Date") AS INTEGER) WHEN 1 THEN '星期一' WHEN 2 THEN '星期二'
WHEN 3 THEN '星期三' WHEN 4 THEN '星期四' WHEN 5 THEN '星期五' WHEN 6 THEN '星期六' ELSE '星期日' END AS weekday,
ROUND(SUM({REV}), 2) AS total_revenue FROM sales_data GROUP BY weekday
ORDER BY (CAST(strftime('%w', "Order Date") AS INTEGER) + 6) % 7""",
    "region_top_customers_orders": f"""SELECT "Customer Name" AS customer_name, ROUND(SUM({REV}), 2) AS total_revenue,
COUNT("Order ID") AS order_count FROM sales_data WHERE Region = '{{region}}' AND {YEAR} = '{{year}}'
GROUP BY 1 ORDER BY SUM({REV}) DESC, customer_name ASC LIMIT {{n}}""",
    "avg_qty_per_order": """SELECT "Product Category" AS product_category, ROUND(AVG(Quantity), 2) AS avg_quantity
FROM sales_data GROUP BY 1 ORDER BY AVG(Quantity) DESC, product_category ASC""",
    "countries_above_avg": f"""WITH c AS (SELECT Country AS country, SUM({REV}) AS revenue FROM sales_data GROUP BY 1)
SELECT country, ROUND(revenue, 2) AS total_revenue FROM c WHERE revenue > (SELECT AVG(revenue) FROM c)
ORDER BY revenue DESC, country ASC""",
    "h2_vs_h1": f"""WITH h AS (SELECT Region AS region,
SUM(CASE WHEN CAST(strftime('%m', "Order Date") AS INTEGER) <= 6 THEN {REV} END) AS h1,
SUM(CASE WHEN CAST(strftime('%m', "Order Date") AS INTEGER) > 6 THEN {REV} END) AS h2
FROM sales_data WHERE {YEAR} = '{{year}}' GROUP BY Region)
SELECT region, ROUND((h2 - h1) * 100.0 / h1, 2) AS growth_pct FROM h ORDER BY (h2 - h1) / h1 DESC, region ASC""",
    "country_rank_in_region": f"""WITH c AS (SELECT Country AS country, SUM({REV}) AS revenue FROM sales_data
WHERE Region = (SELECT Region FROM sales_data WHERE Country = '{{country}}' LIMIT 1) GROUP BY Country),
r AS (SELECT country, ROW_NUMBER() OVER (ORDER BY revenue DESC, country ASC) AS rank FROM c)
SELECT rank FROM r WHERE country = '{{country}}'""",
    "nth_customer": f"""SELECT customer_name FROM (SELECT "Customer Name" AS customer_name, ROW_NUMBER() OVER
(ORDER BY SUM({REV}) DESC, "Customer Name" ASC) AS rn FROM sales_data WHERE {YEAR} = '{{year}}' GROUP BY 1)
WHERE rn = {{n}}""",
    "months_beat_last_year": f"""SELECT strftime('%m', "Order Date") AS month FROM sales_data GROUP BY month
HAVING SUM(CASE WHEN {YEAR} = '2024' THEN {REV} END) > SUM(CASE WHEN {YEAR} = '2023' THEN {REV} END) ORDER BY month""",
    "high_price_orders": """SELECT "Product Category" AS product_category, COUNT("Order ID") AS order_count FROM sales_data
WHERE ASP > {price} GROUP BY 1 ORDER BY order_count DESC, product_category ASC""",
    "revenue_per_customer_region": f"""SELECT Region AS region, ROUND(SUM({REV}) / COUNT(DISTINCT "Customer Name"), 2)
AS revenue_per_customer FROM sales_data GROUP BY Region
ORDER BY SUM({REV}) / COUNT(DISTINCT "Customer Name") DESC, region ASC""",
    "country_top_quantity_product": """WITH t AS (SELECT Country AS country, "Product Name" AS product_name,
SUM(Quantity) AS total_quantity FROM sales_data WHERE Region = '{region}' GROUP BY 1, 2),
r AS (SELECT *, ROW_NUMBER() OVER (PARTITION BY country ORDER BY total_quantity DESC, product_name ASC) AS rn FROM t)
SELECT country, product_name, total_quantity FROM r WHERE rn = 1 ORDER BY country""",
    "product_growth_amount": f"""WITH a AS (SELECT "Product Name" AS product_name,
SUM(CASE WHEN {YEAR} = '2024' THEN {REV} END) - SUM(CASE WHEN {YEAR} = '2023' THEN {REV} END) AS diff
FROM sales_data GROUP BY 1) SELECT product_name, ROUND(diff, 2) AS revenue_increase FROM a
ORDER BY diff DESC, product_name ASC LIMIT {{n}}""",
    "customer_category_mix": f"""SELECT "Product Category" AS product_category,
ROUND(SUM({REV}) * 100.0 / SUM(SUM({REV})) OVER (), 2) AS share_pct FROM sales_data
WHERE "Customer Name" = '{{customer}}' GROUP BY 1 ORDER BY SUM({REV}) DESC, product_category ASC""",
    "quarter_max_qoq": f"""WITH q AS (SELECT {YEAR} || '-Q' || {QTR} AS quarter, SUM({REV}) AS revenue FROM sales_data
GROUP BY quarter), g AS (SELECT quarter, (revenue - LAG(revenue) OVER (ORDER BY quarter)) * 100.0
/ LAG(revenue) OVER (ORDER BY quarter) AS growth FROM q)
SELECT quarter, ROUND(growth, 2) AS qoq_growth_pct FROM g WHERE growth IS NOT NULL
ORDER BY growth DESC, quarter ASC LIMIT 1""",
}


def _c(*pairs):
    return [{"name": n, "type": "concept", "meaning": m} for n, m in pairs]


def _lit(name, *words):
    return {"name": name, "type": "entity", "any_of": list(words)}


TOP = {3: ["3", "三"], 5: ["5", "五"], 10: ["10", "十"], 2: ["第二", "第 2", "第2", "亞軍"]}

# (key, eval_type, metric, chart x column or None, [(params, question, extra literal entities)], concepts)
SPECS = [
    ("region_top_product", "dataframe", "amount", None,
     [({"year": "2024"}, "2024 年各區域銷售額最高的產品分別是什麼？金額是多少？", [])],
     _c(("分組", "每個區域各取一個產品"), ("指標", "銷售金額"))),
    ("category_top3_quantity", "dataframe", "quantity", None,
     [({}, "每個產品類別裡，銷售量前三名的產品分別是哪些？", [_lit("名次", *TOP[3])])],
     _c(("分組", "每個類別內各自排名"), ("指標", "銷售數量"))),
    ("region_mom", "dataframe", "amount", None,
     [({"region": "APAC", "year": "2024"}, "APAC 在 2024 年每個月的銷售額比前一個月成長了多少百分比？", [])],
     _c(("指標", "月增率（與前一個月相比的百分比）"), ("時間粒度", "按月"))),
    ("cumulative", "dataframe", "amount", "month",
     [({"year": "2023"}, "2023 年從一月起逐月累計的銷售額是多少？請用折線圖呈現。", [])],
     _c(("指標", "逐月累計的銷售額（不是單月）"), ("圖表", "折線圖"))),
    ("country_decline", "dataframe", "amount", None,
     [({}, "哪些國家 2024 年的銷售額比 2023 年下滑？各下滑了多少百分比？", [])],
     _c(("條件", "只列出銷售額下滑的國家"), ("指標", "年增率百分比"))),
    ("all_categories", "number", None, None,
     [({"year": "2024"}, "2024 年三種產品類別都有買過的客戶有幾位？", [_lit("類別數", "三", "3", "所有", "全部", "每一種", "每種")])],
     _c(("對象", "客戶人數，不是訂單數"), ("條件", "同一年內買齊所有產品類別"))),
    ("retained", "number", None, None,
     [({}, "2023 年和 2024 年都有下過單的客戶共有多少位？", [])],
     _c(("對象", "客戶人數"), ("條件", "兩年都有購買"))),
    ("new_customers", "list", None, None,
     [({"year": "2024"}, "哪些客戶是 2024 年才第一次下單的新客戶？", [])],
     _c(("條件", "首次購買在該年（之前沒買過）"), ("對象", "客戶名單"))),
    ("aov_region", "dataframe", "amount", None,
     [({}, "各區域平均每筆訂單的金額是多少？由高到低排列。", [])],
     _c(("指標", "平均每筆訂單金額（總營收除以訂單筆數）"), ("排序", "由高到低"))),
    ("region_share_of_category", "dataframe", "amount", None,
     [({"year": "2024", "category": "Software"}, "2024 年 Software 的銷售額中，各區域各占多少百分比？", [])],
     _c(("指標", "該類別銷售額中各區域的占比"),)),
    ("region_yoy", "dataframe", "amount", "region",
     [({}, "各區域 2024 年相較 2023 年的銷售額年增率是多少？請由高到低用長條圖比較。", [])],
     _c(("指標", "各區域年增率百分比"), ("圖表", "長條圖"), ("排序", "由高到低"))),
    ("top_customer_concentration", "number", "amount", None,
     [({"n": 10}, "前十大客戶的銷售額加起來占全部銷售額的百分之幾？", [_lit("名次", *TOP[10])])],
     _c(("指標", "前 N 大客戶合計銷售額占總銷售額的比例"),)),
    ("large_order_share", "dataframe", None, None,
     [({"qty": 10}, "各區域中，數量 10 件以上的大訂單貢獻了多少比例的銷售額？", [_lit("門檻", "10", "十")])],
     _c(("條件", "單筆數量達門檻的訂單"), ("指標", "這些訂單的銷售額占該區域銷售額的比例"))),
    ("best_month_each_year", "dataframe", "amount", None,
     [({}, "2023 年和 2024 年各自銷售額最高的是哪個月？金額多少？", [])],
     _c(("分組", "每年各取一個月份"), ("指標", "銷售金額"))),
    ("asp_change", "dataframe", "amount", None,
     [({"n": 3}, "哪三個產品 2024 年的平均單價比 2023 年漲最多？各漲了多少？", [_lit("名次", *TOP[3])])],
     _c(("指標", "平均單價 ASP 的年度差額"),)),
    ("pareto", "number", None, None,
     [({"pct": 80}, "產品按銷售額由高到低累加，至少要幾個產品才能達到總銷售額的 80%？",
       [_lit("門檻", "80%", "80 %", "八成", "百分之八十")])],
     _c(("指標", "累計銷售額占比達門檻所需的產品個數"),)),
    ("multi_region_customers", "number", None, None,
     [({"k": 4}, "有多少位客戶在全部四個區域都下過單？", [_lit("區域數", "四", "4", "全部", "所有", "每個", "全區", "各個")])],
     _c(("對象", "客戶人數"), ("條件", "在所有區域都有購買"))),
    ("region_category_matrix", "dataframe", "amount", None,
     [({"year": "2023"}, "2023 年每個區域在各產品類別的銷售額分別是多少？", [])],
     _c(("分組", "區域 × 產品類別交叉"), ("指標", "銷售金額"))),
    ("category_best_month", "dataframe", "amount", None,
     [({"year": "2024"}, "2024 年每個產品類別銷售額最高的月份分別是哪個月？", [])],
     _c(("分組", "每個類別各取一個月份"),)),
    ("category_quarter_yoy", "dataframe", "amount", None,
     [({"category": "Electronics"}, "Electronics 在 2024 年每一季與 2023 年同一季相比，銷售額成長率各是多少？", [])],
     _c(("指標", "同季比較的年增率百分比"), ("時間粒度", "按季"))),
    ("weekday", "dataframe", "amount", "weekday",
     [({}, "星期一到星期日，每一天的總銷售額各是多少？請用長條圖比較。", [])],
     _c(("分組", "按星期幾"), ("圖表", "長條圖"))),
    ("region_top_customers_orders", "dataframe", None, None,
     [({"region": "EMEA", "year": "2024", "n": 5}, "EMEA 在 2024 年銷售額前五大的客戶是誰？他們各下了幾筆訂單？",
       [_lit("名次", *TOP[5])])],
     _c(("排名依據", "銷售金額"), ("附加欄位", "每位客戶的訂單筆數"))),
    ("avg_qty_per_order", "dataframe", "quantity", None,
     [({}, "各產品類別平均每筆訂單買幾件？", [])],
     _c(("指標", "每筆訂單的平均數量"), ("分組", "按產品類別"))),
    ("countries_above_avg", "dataframe", "amount", None,
     [({}, "哪些國家的總銷售額高於所有國家的平均值？", [])],
     _c(("條件", "高於各國平均"), ("指標", "銷售金額"))),
    ("h2_vs_h1", "dataframe", "amount", None,
     [({"year": "2024"}, "2024 年下半年相比上半年，各區域的銷售額成長了多少百分比？", [])],
     _c(("比較", "下半年對上半年"), ("指標", "成長百分比"))),
    ("country_rank_in_region", "number", "amount", None,
     [({"country": "Japan"}, "Japan 的銷售額在同一個區域的國家裡排第幾名？", [])],
     _c(("指標", "在所屬區域內的銷售額名次"),)),
    ("nth_customer", "string", "amount", None,
     [({"year": "2023", "n": 2}, "2023 年銷售額第二高的客戶是誰？", [_lit("名次", *TOP[2])])],
     _c(("對象", "單一客戶"),)),
    ("months_beat_last_year", "list", "amount", None,
     [({}, "2024 年有哪些月份的銷售額超過 2023 年同一個月？", [])],
     _c(("條件", "與去年同月相比較高"), ("對象", "月份清單"))),
    ("high_price_orders", "dataframe", "count", None,
     [({"price": 500}, "單價超過 500 的訂單，在各產品類別各有幾筆？", [_lit("門檻", "500", "五百")])],
     _c(("條件", "單價超過門檻"), ("指標", "訂單筆數"))),
    ("revenue_per_customer_region", "dataframe", "amount", None,
     [({}, "各區域平均每位客戶貢獻多少銷售額？", [])],
     _c(("指標", "區域銷售額除以該區域客戶人數"),)),
    ("country_top_quantity_product", "dataframe", "quantity", None,
     [({"region": "LATAM"}, "LATAM 每個國家銷售量最高的產品分別是什麼？", [])],
     _c(("分組", "區域內每個國家各取一個產品"), ("指標", "銷售數量"))),
    ("product_growth_amount", "dataframe", "amount", None,
     [({"n": 3}, "2024 年比 2023 年銷售額增加最多的三個產品是哪些？各增加多少？", [_lit("名次", *TOP[3])])],
     _c(("指標", "銷售額增加的金額，不是百分比"),)),
    ("customer_category_mix", "dataframe", "amount", None,
     [({"customer": "Global Summit Solutions"}, "Global Summit Solutions 的消費金額中，各產品類別各占多少比例？",
       [_lit("客戶", "Global Summit Solutions")])],
     _c(("指標", "該客戶各類別的金額占比"),)),
    ("quarter_max_qoq", "dataframe", "amount", None,
     [({}, "2023 到 2024 年間，哪一季的銷售額季增率最高？是多少？", [])],
     _c(("指標", "與前一季相比的成長率"), ("對象", "單一季度"))),
]

REFERENCE = {name[2:]: fn for name, fn in globals().items() if name.startswith("p_")}


def _slug(params):
    return "-".join(re.sub(r"[^A-Za-z0-9]+", "", str(v)) for v in params.values())


def _close(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=0.011)
    return str(a) == str(b)


def compare_rows(rows: list, reference: list) -> list[str]:
    """Row-by-row, order-sensitive comparison with rounding tolerance."""
    if len(rows) != len(reference):
        return [f"row count: SQL {len(rows)} vs pandas {len(reference)}"]
    problems = []
    for i, (got, want) in enumerate(zip(rows, reference)):
        if len(got) != len(want) or not all(_close(a, b) for a, b in zip(got, want)):
            problems.append(f"row {i}: SQL {list(got)} vs pandas {list(want)}")
    return problems


def build_complex_seeds(db_path=tools.DB_PATH) -> tuple[list[dict], list[dict]]:
    """Every complex seed with its SQL result checked against the pandas reference."""
    _, aliases = load_legacy()
    df = frame()
    seeds, problems = [], []
    for key, eval_type, metric, x, instances, concepts in SPECS:
        for params, question, extra in instances:
            seed_id = "X-" + key + (f"-{_slug(params)}" if params else "")
            sql = SQL[key].format(**params)
            result = tools.read_sales(sql, db_path)
            reference = REFERENCE[key](df, **params)
            mismatch = compare_rows(result["rows"], reference)
            if mismatch:
                problems.append({"seed_id": seed_id, "problem": "; ".join(mismatch[:3])})
            if not result["rows"]:
                problems.append({"seed_id": seed_id, "problem": "empty result"})
            entity_params = {k: v for k, v in params.items() if k in ("region", "country", "category", "product")}
            keep = _entities({"params": entity_params, "question": question}, aliases) + extra
            plot = {"should_plot": True, "chart_type": "line" if key == "cumulative" else "bar",
                    "x_from_column": x} if x else None
            seed = {
                "seed_id": seed_id, "kind": "golden", "label": "allow", "category": "sales_query",
                "question": question, "sql": sql, "eval_type": eval_type, "expected_plot": plot,
                "keep": keep + concepts, "boundary": BOUNDARY, "difficulty": "complex",
                "tags": ["complex", key], "metric": metric, "template": "X_" + key,
                "reference_rows": reference, "oracle": "sql+pandas(csv)", "sql_check": True,
                "seed_source": "SQL 先行：Claude 先寫 SQL 與 pandas 對照實作，再寫出對應題目（2026-11-07）",
            }
            seeds.append(seed)
    return seeds, problems



