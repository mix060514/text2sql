import streamlit as st
import pandas as pd
import altair as alt
import pathlib

st.set_page_config(page_title="Data Dashboard", page_icon="📈", layout="wide")

st.title("📈 Data Dashboard")

# Paths
project_root = pathlib.Path(__file__).parent.parent.parent.resolve()
csv_path = project_root / "data" / "global_sales_data_logic_distribution.csv"


# Load Data
@st.cache_data
def load_data():
    if not csv_path.exists():
        st.error(f"Data file not found at: {csv_path}")
        return pd.DataFrame()

    df = pd.read_csv(csv_path)
    # Convert dates
    df["Order Date"] = pd.to_datetime(df["Order Date"])
    df["Year"] = df["Order Date"].dt.year
    df["Month"] = df["Order Date"].dt.month_name()
    df["Month_Num"] = df["Order Date"].dt.month
    df["Month_Year"] = df["Order Date"].dt.to_period("M").astype(str)
    return df


with st.spinner("Loading data..."):
    df = load_data()

if df.empty:
    st.stop()

# --- STATISTICS ---
total_rows = len(df)
total_revenue = df["Total Revenue"].sum()
years_covered = sorted(df["Year"].unique().tolist())
formatted_years = ", ".join(map(str, years_covered))

st.subheader("Quick Stats")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Rows", f"{total_rows:,}")
with col2:
    st.metric("Total Revenue", f"${total_revenue:,.0f}")
with col3:
    st.metric("Years Covered", formatted_years)

st.divider()

# --- VISUALIZATIONS ---
st.subheader("Visualizations")

# Row 1: Region & Category
c1, c2 = st.columns(2)

with c1:
    st.markdown("### Sales by Region (Grouped by Year)")
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
            y=alt.Y("Total Revenue", title="Revenue", axis=alt.Axis(format="$.2s")),
            color=alt.Color("Year:O", title="Year"),
            column=alt.Column(
                "Region", header=alt.Header(titleOrient="bottom", labelOrient="bottom")
            ),
            tooltip=["Region", "Year", alt.Tooltip("Total Revenue", format="$,.0f")],
        )
        .properties(width=80)  # Adjust width per group
        .interactive()
    )
    st.altair_chart(chart_region)

with c2:
    st.markdown("### Sales by Product Category")
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
            color=alt.Color("Product Category", type="nominal"),
            tooltip=[
                alt.Tooltip("Product Category", type="nominal"),
                alt.Tooltip("Total Revenue", format="$,.0f", type="quantitative"),
                alt.Tooltip(
                    "Percentage",
                    format=".1%",
                    title="Percentage",
                ),
            ],
        )
        .interactive()
    )
    st.altair_chart(chart_cat, use_container_width=True)

# Row 2: Top Products & Trend
c3, c4 = st.columns(2)

with c3:
    st.markdown("### Top 10 Products by Revenue")
    # Top 10 Products
    top_products = (
        df.groupby("Product Name")["Total Revenue"].sum().nlargest(10).reset_index()
    )

    chart_top = (
        alt.Chart(top_products)
        .mark_bar()
        .encode(
            x=alt.X("Total Revenue", title="Revenue", axis=alt.Axis(format="$.2s")),
            y=alt.Y("Product Name", sort="-x", title="Product"),
            color=alt.Color("Total Revenue", legend=None),
            tooltip=["Product Name", alt.Tooltip("Total Revenue", format="$,.0f")],
        )
        .interactive()
    )
    st.altair_chart(chart_top, use_container_width=True)

with c4:
    st.markdown("### Monthly Sales Trend")
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
            x=alt.X("Month_Year", title="Month"),
            y=alt.Y("Total Revenue", title="Revenue", axis=alt.Axis(format="$.2s")),
            tooltip=["Month_Year", alt.Tooltip("Total Revenue", format="$,.0f")],
        )
        .interactive()
    )
    st.altair_chart(chart_trend, use_container_width=True)
