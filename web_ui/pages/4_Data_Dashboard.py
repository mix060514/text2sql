import streamlit as st
import pandas as pd
import altair as alt
import pathlib

st.set_page_config(page_title="資料儀表板", page_icon="📈", layout="wide")

st.title("📈 資料儀表板")
st.markdown("問答系統查詢的同一份銷售資料（單一資料表 `sales_data`），可用來對照對話查詢的答案。")

# Paths
project_root = pathlib.Path(__file__).parent.parent.parent.resolve()
csv_path = project_root / "data" / "global_sales_data_logic_distribution.csv"


# Load Data
@st.cache_data
def load_data():
    if not csv_path.exists():
        st.error("找不到銷售資料檔。")
        return pd.DataFrame()

    df = pd.read_csv(csv_path)
    # Convert dates
    df["Order Date"] = pd.to_datetime(df["Order Date"])
    df["Year"] = df["Order Date"].dt.year
    df["Month"] = df["Order Date"].dt.month_name()
    df["Month_Num"] = df["Order Date"].dt.month
    df["Month_Year"] = df["Order Date"].dt.to_period("M").astype(str)
    return df


with st.spinner("載入資料中…"):
    df = load_data()

if df.empty:
    st.stop()

# --- STATISTICS ---
total_rows = len(df)
total_revenue = df["Total Revenue"].sum()
years_covered = sorted(df["Year"].unique().tolist())
formatted_years = ", ".join(map(str, years_covered))

st.subheader("資料概覽")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("訂單筆數", f"{total_rows:,}")
with col2:
    st.metric("總銷售額", f"${total_revenue:,.0f}")
with col3:
    st.metric("涵蓋年份", formatted_years)

st.divider()

# --- VISUALIZATIONS ---
st.subheader("圖表")

# Row 1: Region & Category
c1, c2 = st.columns(2)

with c1:
    st.markdown("### 各地區銷售額（依年份）")
    # Grouped Bar Chart (Unstacked)
    # Determine revenue per region per year
    region_year_group = (
        df.groupby(["Region", "Year"])["Total Revenue"].sum().reset_index()
    )

    chart_region = (
        alt.Chart(region_year_group)
        .mark_bar()
        .encode(
            x=alt.X(
                "Year:O", title=None, axis=None
            ),  # Hide X axis for cleaner look in group
            y=alt.Y("Total Revenue", title="銷售額", axis=alt.Axis(format="$.2s")),
            color=alt.Color("Year:O", title="年份"),
            column=alt.Column(
                "Region", title="地區", header=alt.Header(titleOrient="bottom", labelOrient="bottom")
            ),
            tooltip=[alt.Tooltip("Region", title="地區"), alt.Tooltip("Year:O", title="年份"),
                     alt.Tooltip("Total Revenue", title="銷售額", format="$,.0f")],
        )
        .properties(width=80)  # Adjust width per group
        .interactive()
    )
    st.altair_chart(chart_region)

with c2:
    st.markdown("### 各產品類別銷售額占比")
    # Pie chart
    cat_group = df.groupby("Product Category")["Total Revenue"].sum().reset_index()

    cat_group["Percentage"] = (
        cat_group["Total Revenue"] / cat_group["Total Revenue"].sum()
    )

    chart_cat = (
        alt.Chart(cat_group)
        .mark_arc(innerRadius=50)
        .encode(
            theta=alt.Theta("Total Revenue", type="quantitative"),
            color=alt.Color("Product Category", type="nominal", title="產品類別"),
            tooltip=[
                alt.Tooltip("Product Category", type="nominal", title="產品類別"),
                alt.Tooltip("Total Revenue", format="$,.0f", type="quantitative", title="銷售額"),
                alt.Tooltip(
                    "Percentage",
                    format=".1%",
                    title="占比",
                ),
            ],
        )
        .interactive()
    )
    st.altair_chart(chart_cat, use_container_width=True)

# Row 2: Top Products & Trend
c3, c4 = st.columns(2)

with c3:
    st.markdown("### 銷售額前 10 名產品")
    # Top 10 Products
    top_products = (
        df.groupby("Product Name")["Total Revenue"].sum().nlargest(10).reset_index()
    )

    chart_top = (
        alt.Chart(top_products)
        .mark_bar()
        .encode(
            x=alt.X("Total Revenue", title="銷售額", axis=alt.Axis(format="$.2s")),
            y=alt.Y("Product Name", sort="-x", title="產品"),
            color=alt.Color("Total Revenue", legend=None),
            tooltip=[alt.Tooltip("Product Name", title="產品"), alt.Tooltip("Total Revenue", title="銷售額", format="$,.0f")],
        )
        .interactive()
    )
    st.altair_chart(chart_top, use_container_width=True)

with c4:
    st.markdown("### 每月銷售額趨勢")
    # Line chart of monthly revenue
    # Group by Month_Year
    monthly_trend = (
        df.groupby(["Year", "Month_Num", "Month_Year"])["Total Revenue"]
        .sum()
        .reset_index()
    )
    monthly_trend = monthly_trend.sort_values(by=["Year", "Month_Num"])

    chart_trend = (
        alt.Chart(monthly_trend)
        .mark_line(point=True)
        .encode(
            x=alt.X("Month_Year", title="月份"),
            y=alt.Y("Total Revenue", title="銷售額", axis=alt.Axis(format="$.2s")),
            tooltip=[alt.Tooltip("Month_Year", title="月份"), alt.Tooltip("Total Revenue", title="銷售額", format="$,.0f")],
        )
        .interactive()
    )
    st.altair_chart(chart_trend, use_container_width=True)
