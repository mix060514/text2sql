"""Golden seeds from the legacy question templates, with a second, independent oracle.

The legacy generator (scripts/generate_ground_truth.py) expands 17 template groups into 132
questions whose answers pandas computes from the source CSV; 54 of them only swap a region or
country alias. Here every template instance carries its own SQL. A seed is accepted only when
the SQL result (through tools.read_sales on SQLite) matches the legacy pandas answer.

Aliases are not separate seeds: they become entity alternatives the rewriter may use, so one
seed per (template, entity) replaces the alias copies. `select_seeds` then keeps a bounded number
of instances per template, spreading entities so breadth survives the reduction.
"""
from collections import Counter
import importlib.util
import json
import math
import re
from pathlib import Path

from text2sql.agents.planned_agent import tools

PROJECT = Path(__file__).resolve().parents[3]
LEGACY_SCRIPT = PROJECT / "scripts" / "generate_ground_truth.py"

REV = '"Total Revenue"'
YEAR = "strftime('%Y', \"Order Date\")"
QTR = "((CAST(strftime('%m', \"Order Date\") AS INTEGER) + 2) / 3)"
MONTH = "strftime('%Y-%m', \"Order Date\")"

CATEGORY_ALIASES = {
    "Electronics": ["Electronics", "電子產品", "電子類", "3C"],
    "Office Supplies": ["Office Supplies", "辦公用品", "文具"],
    "Software": ["Software", "軟體"],
}
REVENUE = ("指標", "銷售金額（營收）")


def _t(tid, sql, eval_type, instances, concepts, *, x=None, exclude=None):
    return {"id": tid, "sql": sql, "eval_type": eval_type, "instances": instances,
            "concepts": concepts, "x": x, "exclude": exclude}


# Metric kind per template (prefix before "_"): amount, quantity or count of orders.
METRIC = {"B": "quantity", "N3": "quantity", "N15": "quantity", "Q11": "quantity",
          "D": "count", "N4": "count", "P3": "count", "Q7": "count", "Q10": "count"}


def _per(kind, values, question, **extra):
    return [{"params": {kind: v}, "question": question.format(**{kind: v}), **extra} for v in values]


REGIONS = ["EMEA", "North America", "APAC", "LATAM"]
CATEGORIES = ["Electronics", "Office Supplies", "Software"]

TEMPLATES = [
    _t("A_region_revenue", "SELECT ROUND(SUM(" + REV + "), 2) AS total_revenue FROM sales_data WHERE Region = '{region}'",
       "number", _per("region", REGIONS, "{region} 的總銷售額是多少？"), [REVENUE]),
    _t("B_region_top3_quantity",
       "SELECT \"Product Name\" AS product_name, SUM(Quantity) AS total_quantity FROM sales_data WHERE Region = '{region}' "
       "GROUP BY \"Product Name\" ORDER BY total_quantity DESC, product_name ASC LIMIT 3",
       "list", _per("region", REGIONS, "{region} 銷售量最高的前三個產品是什麼？", topn=3),
       [("指標", "銷售數量，不是金額"), ("排名", "前三名產品")]),
    _t("C_category_yearly_revenue",
       f"SELECT {YEAR} AS year, ROUND(SUM({REV}), 2) AS total_revenue FROM sales_data WHERE \"Product Category\" = '{{category}}' "
       "GROUP BY year ORDER BY year",
       "dataframe", _per("category", CATEGORIES, "請列出 {category} 類別在各年份的銷售額總計。"),
       [REVENUE, ("分組", "按年份")], x="year"),
    _t("D_country_top_order_customer",
       "SELECT \"Customer Name\" AS customer_name, COUNT(\"Order ID\") AS order_count FROM sales_data WHERE Country = '{country}' "
       "GROUP BY \"Customer Name\" ORDER BY order_count DESC, customer_name ASC LIMIT 1",
       "string", _per("country", ["Canada", "India", "Germany", "Australia", "China"], "{country} 訂單數量最多的客戶是誰？"),
       [("指標", "訂單筆數，不是金額"), ("對象", "單一客戶")]),
    _t("E_year_best_quarter",
       f"SELECT 'Q' || {QTR} AS quarter, ROUND(SUM({REV}), 2) AS total_revenue FROM sales_data WHERE {YEAR} = '{{year}}' "
       "GROUP BY quarter ORDER BY SUM(" + REV + ") DESC, quarter ASC LIMIT 1",
       "string", _per("year", [2023, 2024], "{year} 年哪一個季度的總銷售額表現最好？"),
       [REVENUE, ("對象", "表現最好的單一季度")]),
    _t("F_category_yoy",
       f"SELECT ROUND((SUM(CASE WHEN {YEAR} = '2024' THEN {REV} END) - SUM(CASE WHEN {YEAR} = '2023' THEN {REV} END)) * 100.0 "
       f"/ SUM(CASE WHEN {YEAR} = '2023' THEN {REV} END), 2) AS yoy_pct FROM sales_data WHERE \"Product Category\" = '{{category}}'",
       "number", _per("category", CATEGORIES, "{category} 類別在 2024 年相對於 2023 年的銷售額年增率 (YoY) 是多少？(請以百分比表示)"),
       [("指標", "銷售額年增率（百分比）")]),
    _t("G_region_monthly_trend",
       f"SELECT {MONTH} AS month, ROUND(SUM({REV}), 2) AS total_revenue FROM sales_data WHERE Region = '{{region}}' "
       "GROUP BY month ORDER BY month",
       "dataframe", _per("region", REGIONS, "請列出 {region} 區域在 2023 至 2024 年每個月的銷售趨勢數據。"),
       [REVENUE, ("時間粒度", "按月")], x="month"),
    _t("H_q4_revenue_increase",
       f"SELECT ROUND(SUM(CASE WHEN {YEAR} = '2024' AND {QTR} = 4 THEN {REV} ELSE 0 END) - "
       f"SUM(CASE WHEN {YEAR} = '2023' AND {QTR} = 4 THEN {REV} ELSE 0 END), 2) AS revenue_increase FROM sales_data",
       "number", [{"params": {}, "question": "2024 年第四季相較於 2023 年第四季，總銷售額增加了多少？", "quarter": 4}],
       [("指標", "總銷售額的差額（金額，不是百分比）")]),
    _t("I_na_top5_customer_category_share",
       f"WITH top AS (SELECT \"Customer Name\" AS c FROM sales_data WHERE Region = 'North America' AND {YEAR} = '2023' "
       f"GROUP BY c ORDER BY SUM({REV}) DESC, c ASC LIMIT 5), mix AS (SELECT \"Customer Name\" AS customer_name, "
       f"\"Product Category\" AS product_category, SUM({REV}) AS revenue FROM sales_data WHERE {YEAR} = '2023' "
       "AND \"Customer Name\" IN (SELECT c FROM top) GROUP BY 1, 2) SELECT customer_name, product_category, "
       "ROUND(revenue / SUM(revenue) OVER (PARTITION BY customer_name), 4) AS share FROM mix ORDER BY customer_name, product_category",
       "dataframe", [{"params": {"region": "North America"},
                      "question": "在 North America 區域，2023 年前五大客戶購買各產品類別的金額占比為何？"}],
       [("指標", "各產品類別的金額占比")],
       exclude="題目說「在 North America 區域」，但舊答案的占比包含這些客戶在其他地區的購買（487/494 位客戶跨地區），題意與答案不一致"),
    _t("J_biggest_change_customer",
       f"WITH a AS (SELECT \"Customer Name\" AS c, SUM(CASE WHEN {YEAR} = '2023' THEN {REV} END) AS r23, "
       f"SUM(CASE WHEN {YEAR} = '2024' THEN {REV} END) AS r24 FROM sales_data GROUP BY c) "
       "SELECT c AS customer_name, ROUND(ABS(r24 - r23), 2) AS abs_change FROM a WHERE r23 IS NOT NULL AND r24 IS NOT NULL "
       "ORDER BY ABS(r24 - r23) DESC, customer_name ASC LIMIT 1",
       "string", [{"params": {}, "question": "在 2023 到 2024 年間，哪位客戶的購買總額變動幅度（絕對值）最大？"}],
       [("指標", "購買總額變動的絕對值")]),
    _t("K_category_top_customer",
       f"SELECT \"Customer Name\" AS customer_name, ROUND(SUM({REV}), 2) AS total_revenue FROM sales_data "
       f"WHERE \"Product Category\" = '{{category}}' GROUP BY \"Customer Name\" ORDER BY SUM({REV}) DESC, customer_name ASC LIMIT 1",
       "string", _per("category", ["Electronics"], "產品類別 {category} 的最大購買客戶是誰？"), [REVENUE]),
    _t("L_top_asp_product",
       "SELECT \"Product Name\" AS product_name, ROUND(AVG(ASP), 2) AS avg_asp FROM sales_data GROUP BY \"Product Name\" "
       "ORDER BY AVG(ASP) DESC, product_name ASC LIMIT 1",
       "string", [{"params": {}, "question": "哪一個產品的平均單價 (ASP) 最高？"}], [("指標", "平均單價 ASP")]),
    _t("M_region_top_category",
       f"WITH t AS (SELECT Region AS region, \"Product Category\" AS product_category, SUM({REV}) AS revenue FROM sales_data "
       "GROUP BY 1, 2), r AS (SELECT region, product_category, ROW_NUMBER() OVER (PARTITION BY region ORDER BY revenue DESC, "
       "product_category ASC) AS rn FROM t) SELECT region, product_category FROM r WHERE rn = 1 ORDER BY region",
       "dataframe", [{"params": {}, "question": "各個區域 (Region) 最偏好（銷售額最高）的產品類別分別為何？"}],
       [REVENUE, ("分組", "每個區域各一個類別")]),
    _t("N1_region_revenue_bar",
       f"SELECT Region AS region, ROUND(SUM({REV}), 2) AS total_revenue FROM sales_data GROUP BY Region "
       f"ORDER BY SUM({REV}) DESC, region ASC",
       "dataframe", [{"params": {}, "question": "請比較各區域的總銷售額。"}], [REVENUE, ("分組", "按區域")], x="region"),
    _t("N2_category_revenue_bar",
       f"SELECT \"Product Category\" AS product_category, ROUND(SUM({REV}), 2) AS total_revenue FROM sales_data "
       f"GROUP BY \"Product Category\" ORDER BY SUM({REV}) DESC, product_category ASC",
       "dataframe", [{"params": {}, "question": "各產品類別的總銷售額分別是多少？請用圖表比較。"}],
       [REVENUE, ("分組", "按產品類別"), ("圖表", "要圖表比較")], x="product_category"),
    _t("N3_category_quantity_bar",
       "SELECT \"Product Category\" AS product_category, SUM(Quantity) AS total_quantity FROM sales_data "
       "GROUP BY \"Product Category\" ORDER BY total_quantity DESC, product_category ASC",
       "dataframe", [{"params": {}, "question": "各產品類別的總銷售量各是多少？"}],
       [("指標", "銷售數量"), ("分組", "按產品類別")], x="product_category"),
    _t("N4_category_orders_bar",
       "SELECT \"Product Category\" AS product_category, COUNT(\"Order ID\") AS order_count FROM sales_data "
       "GROUP BY \"Product Category\" ORDER BY order_count DESC, product_category ASC",
       "dataframe", [{"params": {}, "question": "各產品類別的訂單數量各有多少？"}],
       [("指標", "訂單筆數"), ("分組", "按產品類別")], x="product_category"),
    _t("N5_region_country_revenue_bar",
       f"SELECT Country AS country, ROUND(SUM({REV}), 2) AS total_revenue FROM sales_data WHERE Region = '{{region}}' "
       f"GROUP BY Country ORDER BY SUM({REV}) DESC, country ASC",
       "dataframe", [
           {"params": {"region": "EMEA"}, "question": "EMEA 區域中各國家的銷售額比較如何？"},
           {"params": {"region": "APAC"}, "question": "APAC 區域各國家的銷售額分別是多少？"},
           {"params": {"region": "North America"}, "question": "北美區域各國的銷售額比較。"},
           {"params": {"region": "LATAM"}, "question": "LATAM 區域各國家的銷售額分別是多少？"}],
       [REVENUE, ("分組", "區域內按國家")], x="country"),
    _t("N9_category_product_revenue_bar",
       f"SELECT \"Product Name\" AS product_name, ROUND(SUM({REV}), 2) AS total_revenue FROM sales_data "
       f"WHERE \"Product Category\" = '{{category}}' GROUP BY \"Product Name\" ORDER BY SUM({REV}) DESC, product_name ASC",
       "dataframe", [
           {"params": {"category": "Electronics"}, "question": "Electronics 類別中各產品的銷售額排名為何？"},
           {"params": {"category": "Office Supplies"}, "question": "Office Supplies 類別各產品的銷售額比較。"},
           {"params": {"category": "Software"}, "question": "Software 類別中各產品的銷售額分別是多少？"}],
       [REVENUE, ("分組", "類別內按產品")], x="product_name"),
    _t("N12_year_quarter_revenue_bar",
       f"SELECT 'Q' || {QTR} AS quarter, ROUND(SUM({REV}), 2) AS total_revenue FROM sales_data WHERE {YEAR} = '{{year}}' "
       "GROUP BY quarter ORDER BY quarter",
       "dataframe", _per("year", [2023, 2024], "{year} 年各季度的銷售額分別是多少？"),
       [REVENUE, ("分組", "按季")], x="quarter"),
    _t("N14_top5_product_revenue_bar",
       f"SELECT \"Product Name\" AS product_name, ROUND(SUM({REV}), 2) AS total_revenue FROM sales_data "
       f"GROUP BY \"Product Name\" ORDER BY SUM({REV}) DESC, product_name ASC LIMIT 5",
       "dataframe", [{"params": {}, "question": "銷售額前五名的產品分別是哪些？請用圖表呈現。", "topn": 5}],
       [REVENUE, ("圖表", "要圖表")], x="product_name"),
    _t("N15_top5_product_quantity_bar",
       "SELECT \"Product Name\" AS product_name, SUM(Quantity) AS total_quantity FROM sales_data "
       "GROUP BY \"Product Name\" ORDER BY total_quantity DESC, product_name ASC LIMIT 5",
       "dataframe", [{"params": {}, "question": "銷售量前五名的產品是哪些？", "topn": 5}],
       [("指標", "銷售數量")], x="product_name"),
    _t("O1_category_monthly_trend",
       f"SELECT {MONTH} AS month, ROUND(SUM({REV}), 2) AS total_revenue FROM sales_data WHERE \"Product Category\" = '{{category}}' "
       "GROUP BY month ORDER BY month",
       "dataframe", _per("category", CATEGORIES, "請呈現 {category} 類別在 2023 至 2024 年的月銷售額趨勢。"),
       [REVENUE, ("時間粒度", "按月"), ("圖表", "趨勢")], x="month"),
    _t("O4_quarterly_trend",
       f"SELECT {YEAR} || '-Q' || {QTR} AS quarter, ROUND(SUM({REV}), 2) AS total_revenue FROM sales_data "
       "GROUP BY quarter ORDER BY quarter",
       "dataframe", [{"params": {}, "question": "請呈現 2023 至 2024 年各季度的總銷售額趨勢。"}],
       [REVENUE, ("時間粒度", "按季，跨兩年"), ("圖表", "趨勢")], x="quarter"),
    _t("O5_region_quarterly_trend",
       f"SELECT {YEAR} || '-Q' || {QTR} AS quarter, ROUND(SUM({REV}), 2) AS total_revenue FROM sales_data "
       "WHERE Region = '{region}' GROUP BY quarter ORDER BY quarter",
       "dataframe", _per("region", REGIONS, "請呈現 {region} 區域 2023 至 2024 年各季度的銷售額趨勢。"),
       [REVENUE, ("時間粒度", "按季，跨兩年"), ("圖表", "趨勢")], x="quarter"),
    _t("O9_country_monthly_trend",
       f"SELECT {MONTH} AS month, ROUND(SUM({REV}), 2) AS total_revenue FROM sales_data WHERE Country = '{{country}}' "
       "GROUP BY month ORDER BY month",
       "dataframe", [
           {"params": {"country": "United States"}, "question": "請呈現美國 (United States) 2023 至 2024 年的月銷售額趨勢。"},
           {"params": {"country": "China"}, "question": "請呈現中國 (China) 2023 至 2024 年的月銷售額趨勢。"},
           {"params": {"country": "Germany"}, "question": "請呈現德國 (Germany) 2023 至 2024 年的月銷售額趨勢。"}],
       [REVENUE, ("時間粒度", "按月"), ("圖表", "趨勢")], x="month"),
    _t("P1_region_share_pie",
       f"SELECT Region AS region, ROUND(SUM({REV}) * 100.0 / (SELECT SUM({REV}) FROM sales_data), 2) AS share_pct "
       f"FROM sales_data GROUP BY Region ORDER BY SUM({REV}) DESC, region ASC",
       "dataframe", [{"params": {}, "question": "各區域的銷售額占比分別是多少？請用圓餅圖呈現。"}],
       [("指標", "銷售額占比"), ("分組", "按區域"), ("圖表", "圓餅圖")], x="region"),
    _t("P2_category_share_pie",
       f"SELECT \"Product Category\" AS product_category, ROUND(SUM({REV}) * 100.0 / (SELECT SUM({REV}) FROM sales_data), 2) "
       f"AS share_pct FROM sales_data GROUP BY \"Product Category\" ORDER BY SUM({REV}) DESC, product_category ASC",
       "dataframe", [{"params": {}, "question": "各產品類別的銷售額占比分別是多少？請用圓餅圖表示。"}],
       [("指標", "銷售額占比"), ("分組", "按產品類別"), ("圖表", "圓餅圖")], x="product_category"),
    _t("P3_region_order_share_pie",
       "SELECT Region AS region, ROUND(COUNT(\"Order ID\") * 100.0 / (SELECT COUNT(\"Order ID\") FROM sales_data), 2) AS share_pct "
       "FROM sales_data GROUP BY Region ORDER BY COUNT(\"Order ID\") DESC, region ASC",
       "dataframe", [{"params": {}, "question": "各區域的訂單數量占比為何？"}],
       [("指標", "訂單筆數占比"), ("分組", "按區域")], x="region"),
    _t("P4_year_category_share_pie",
       f"SELECT \"Product Category\" AS product_category, ROUND(SUM({REV}) * 100.0 / (SELECT SUM({REV}) FROM sales_data "
       f"WHERE {YEAR} = '{{year}}'), 2) AS share_pct FROM sales_data WHERE {YEAR} = '{{year}}' GROUP BY \"Product Category\" "
       f"ORDER BY SUM({REV}) DESC, product_category ASC",
       "dataframe", _per("year", [2023, 2024], "{year} 年各產品類別的銷售額占比分別是多少？"),
       [("指標", "銷售額占比"), ("分組", "按產品類別")], x="product_category"),
    _t("P6_region_country_share_pie",
       f"SELECT Country AS country, ROUND(SUM({REV}) * 100.0 / (SELECT SUM({REV}) FROM sales_data WHERE Region = '{{region}}'), 2) "
       f"AS share_pct FROM sales_data WHERE Region = '{{region}}' GROUP BY Country ORDER BY SUM({REV}) DESC, country ASC",
       "dataframe", [
           {"params": {"region": "APAC"}, "question": "APAC 區域中各國家的銷售額占比分別是多少？"},
           {"params": {"region": "EMEA"}, "question": "EMEA 區域中各國家的銷售額占比為何？"},
           {"params": {"region": "North America"}, "question": "北美區域各國的銷售額占比如何？"}],
       [("指標", "銷售額占比"), ("分組", "區域內按國家")], x="country"),
    _t("Q1_year_revenue",
       f"SELECT ROUND(SUM({REV}), 2) AS total_revenue FROM sales_data WHERE {YEAR} = '{{year}}'",
       "number", [{"params": {"year": 2024}, "question": "2024 年的總銷售額是多少？"},
                  {"params": {"year": 2023}, "question": "2023 年的總銷售額有多少？"}], [REVENUE]),
    _t("Q2_category_revenue",
       f"SELECT ROUND(SUM({REV}), 2) AS total_revenue FROM sales_data WHERE \"Product Category\" = '{{category}}'",
       "number", [{"params": {"category": "Electronics"}, "question": "Electronics 類別的總銷售額是多少？"},
                  {"params": {"category": "Software"}, "question": "Software 類別的總銷售額為何？"}], [REVENUE]),
    _t("Q5_top_region",
       f"SELECT Region AS region FROM sales_data GROUP BY Region ORDER BY SUM({REV}) DESC, region ASC LIMIT 1",
       "string", [{"params": {}, "question": "銷售額最高的區域是哪一個？"}], [REVENUE]),
    _t("Q6_top_product",
       f"SELECT \"Product Name\" AS product_name FROM sales_data GROUP BY \"Product Name\" ORDER BY SUM({REV}) DESC, product_name ASC LIMIT 1",
       "string", [{"params": {}, "question": "銷售額最高的產品是什麼？"}], [REVENUE]),
    _t("Q7_top_order_customer",
       "SELECT \"Customer Name\" AS customer_name FROM sales_data GROUP BY \"Customer Name\" "
       "ORDER BY COUNT(\"Order ID\") DESC, customer_name ASC LIMIT 1",
       "string", [{"params": {}, "question": "訂單數量最多的客戶是誰？"}], [("指標", "訂單筆數")]),
    _t("Q8_top_country",
       f"SELECT Country AS country FROM sales_data GROUP BY Country ORDER BY SUM({REV}) DESC, country ASC LIMIT 1",
       "string", [{"params": {}, "question": "銷售額最高的國家是哪一個？"}], [REVENUE]),
    _t("Q9_product_asp",
       "SELECT ROUND(AVG(ASP), 2) AS avg_asp FROM sales_data WHERE \"Product Name\" = '{product}'",
       "number", [{"params": {"product": "Enterprise Laptop X1"}, "question": "Enterprise Laptop X1 的平均單價 (ASP) 是多少？"}],
       [("指標", "平均單價 ASP")]),
    _t("Q10_country_orders",
       "SELECT COUNT(\"Order ID\") AS order_count FROM sales_data WHERE Country = '{country}'",
       "number", [{"params": {"country": "United States"}, "question": "美國 (United States) 總共有多少筆訂單？"}],
       [("指標", "訂單筆數")]),
    _t("Q11_country_quantity",
       "SELECT SUM(Quantity) AS total_quantity FROM sales_data WHERE Country = '{country}'",
       "number", [{"params": {"country": "Japan"}, "question": "日本 (Japan) 的總銷售量是多少？"}], [("指標", "銷售數量")]),
    _t("Q12_product_revenue",
       f"SELECT ROUND(SUM({REV}), 2) AS total_revenue FROM sales_data WHERE \"Product Name\" = '{{product}}'",
       "number", [{"params": {"product": "Cloud License (Annual)"}, "question": "Cloud License (Annual) 的總銷售額是多少？"}],
       [REVENUE]),
    _t("Q13_total_yoy",
       f"SELECT ROUND((SUM(CASE WHEN {YEAR} = '2024' THEN {REV} END) - SUM(CASE WHEN {YEAR} = '2023' THEN {REV} END)) * 100.0 "
       f"/ SUM(CASE WHEN {YEAR} = '2023' THEN {REV} END), 2) AS yoy_pct FROM sales_data",
       "number", [{"params": {}, "question": "2024 年相較 2023 年的總銷售額年增率是多少？(以百分比表示)"}],
       [("指標", "總銷售額年增率（百分比）")]),
    _t("Q14_region_yoy",
       f"SELECT ROUND((SUM(CASE WHEN {YEAR} = '2024' THEN {REV} END) - SUM(CASE WHEN {YEAR} = '2023' THEN {REV} END)) * 100.0 "
       f"/ SUM(CASE WHEN {YEAR} = '2023' THEN {REV} END), 2) AS yoy_pct FROM sales_data WHERE Region = '{{region}}'",
       "number", [
           {"params": {"region": "North America"}, "question": "North America 區域 2024 年相較 2023 年的銷售額年增率為何？"},
           {"params": {"region": "APAC"}, "question": "APAC 區域 2024 年相較 2023 年的銷售額成長率是多少百分比？"},
           {"params": {"region": "EMEA"}, "question": "EMEA 區域 2024 年對比 2023 年的銷售額年增率是多少？"}],
       [("指標", "銷售額年增率（百分比）")]),
]


# ---------- legacy reference ----------

def load_legacy():
    """Import the untouched legacy generator by path; returns (entries, aliases)."""
    spec = importlib.util.spec_from_file_location("legacy_generate_ground_truth", LEGACY_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    entries = json.loads(json.dumps(module.generate_questions(), ensure_ascii=False, default=str))
    return entries, module.ALIASES


# ---------- answer comparison ----------

def _label(value) -> str:
    return re.sub(r"^(\d{4}) Q(\d)$", r"\1-Q\2", str(value))  # legacy "2023 Q1" -> convention "2023-Q1"


def _close(a, b) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=0.011)
    return a == b


def compare_answers(eval_type: str, columns: list, rows: list, reference, reference_x=None, sql_x=None) -> list[str]:
    """Return mismatches between the SQL result and the legacy pandas answer (empty = agree)."""
    problems = []
    if eval_type in ("number", "string"):
        if len(rows) != 1:
            return [f"expected one row, got {len(rows)}"]
        if not _close(rows[0][0], reference):
            problems.append(f"SQL {rows[0][0]!r} != pandas {reference!r}")
    elif eval_type == "list":
        got = [row[0] for row in rows]
        if got != reference:
            problems.append(f"SQL {got} != pandas {reference}")
    elif isinstance(reference, dict) and reference and all(isinstance(v, dict) for v in reference.values()):
        got: dict = {}
        for a, b, value in rows:
            got.setdefault(str(a), {})[str(b)] = value
        for outer, inner in reference.items():
            for key, value in inner.items():
                if not _close(got.get(outer, {}).get(key, 0), value):
                    problems.append(f"{outer}/{key}: SQL {got.get(outer, {}).get(key, 0)} != pandas {value}")
        if set(got) != set(reference):
            problems.append(f"row keys differ: SQL {sorted(got)} vs pandas {sorted(reference)}")
    elif isinstance(reference, dict):
        got = {_label(row[0]): row[1] for row in rows}
        ref = {_label(k): v for k, v in reference.items()}
        if set(got) != set(ref):
            problems.append(f"keys differ: SQL-only {sorted(set(got) - set(ref))}, pandas-only {sorted(set(ref) - set(got))}")
        for key in set(got) & set(ref):
            if not _close(got[key], ref[key]):
                problems.append(f"{key}: SQL {got[key]} != pandas {ref[key]}")
    else:
        problems.append(f"unsupported reference shape for {eval_type}")
    if reference_x is not None and sql_x is not None and [_label(x) for x in reference_x] != [_label(x) for x in sql_x]:
        problems.append(f"x order: SQL {sql_x} vs pandas {reference_x}")
    return problems


# ---------- seed construction ----------

def _entities(instance: dict, aliases: dict) -> list[dict]:
    groups = []
    for kind, value in instance["params"].items():
        if kind in ("region", "country"):
            names = list(dict.fromkeys([value, *aliases.get(value, [])]))
            if value == "North America":
                names += ["北美區", "北美地區"]
            groups.append({"name": "地區" if kind == "region" else "國家", "type": "entity", "any_of": names})
        elif kind == "category":
            groups.append({"name": "產品類別", "type": "entity", "any_of": CATEGORY_ALIASES[value]})
        elif kind == "product":
            groups.append({"name": "產品", "type": "entity", "any_of": [value]})
    for year in sorted(set(re.findall(r"20\d\d", instance["question"]))):
        groups.append({"name": f"年份 {year}", "type": "entity", "any_of": [year]})
    if instance.get("topn"):
        n = instance["topn"]
        groups.append({"name": "名次", "type": "entity", "any_of": [str(n), {3: "三", 5: "五"}[n]]})
    if instance.get("quarter") == 4:
        groups.append({"name": "季度", "type": "entity", "any_of": ["第四季", "Q4", "第 4 季", "第4季"]})
    return groups


def build_instances(db_path=tools.DB_PATH) -> tuple[list[dict], list[dict]]:
    """Every template instance with SQL answer, pandas reference and comparison result."""
    legacy, aliases = load_legacy()
    by_question = {entry["question"]: entry for entry in legacy}
    built, problems = [], []
    for template in TEMPLATES:
        for instance in template["instances"]:
            ref = by_question.get(instance["question"])
            slug = "-".join(re.sub(r"[^A-Za-z0-9]+", "", str(v)) for v in instance["params"].values())
            seed_id = template["id"].split("_")[0] + (f"-{slug}" if slug else "")
            if ref is None:
                problems.append({"seed_id": seed_id, "problem": "legacy question not found"})
                continue
            sql = template["sql"].format(**instance["params"])
            result = tools.read_sales(sql, db_path)
            sql_x = None
            if template["x"]:
                index = result["columns"].index(template["x"])
                sql_x = [str(row[index]) for row in result["rows"]]
            ref_plot = ref.get("expected_plot")
            mismatches = compare_answers(template["eval_type"], result["columns"], result["rows"], ref["ground_truth"],
                                         (ref_plot or {}).get("expected_x_order"), sql_x)
            plot = None
            if ref_plot is not None:
                plot = {k: v for k, v in ref_plot.items() if k != "expected_x_order"}
                if plot.get("should_plot"):
                    plot["x_from_column"] = template["x"]
            concepts = [{"name": n, "type": "concept", "meaning": m} for n, m in template["concepts"]]
            seed = {
                "seed_id": seed_id, "kind": "golden", "label": "allow", "category": "sales_query",
                "question": instance["question"], "sql": sql, "eval_type": template["eval_type"],
                "expected_plot": plot, "keep": _entities(instance, aliases) + concepts,
                "boundary": "只能換說法；必須問同一個指標、分組、名次、時間範圍與圖表需求，不得新增或刪除條件。",
                "difficulty": ref.get("difficulty"), "tags": ref.get("tags", []),
                "metric": METRIC.get(template["id"].split("_")[0], "amount"),
                "template": template["id"], "reference_answer": ref["ground_truth"], "oracle": "sql+pandas",
            }
            if template["exclude"]:
                seed["exclude"] = template["exclude"]
            if mismatches:
                problems.append({"seed_id": seed_id, "problem": "; ".join(mismatches)})
            built.append(seed)
    return built, problems


def select_seeds(seeds: list[dict], per_template: int | None = 1) -> list[dict]:
    """Keep up to N instances per template, choosing the least-used entities first."""
    usable = [s for s in seeds if "exclude" not in s]
    if per_template is None:
        return usable
    used: Counter = Counter()
    chosen = []
    for template in dict.fromkeys(s["template"] for s in usable):
        pool = [s for s in usable if s["template"] == template]
        for _ in range(min(per_template, len(pool))):
            key = lambda s: (sum(used[e] for e in _entity_values(s)), pool.index(s))
            pick = min(pool, key=key)
            pool.remove(pick)
            chosen.append(pick)
            used.update(_entity_values(pick))
    return chosen


def _entity_values(seed: dict) -> list[str]:
    return [g["any_of"][0] for g in seed["keep"] if g["type"] == "entity" and not g["name"].startswith("年份")]
