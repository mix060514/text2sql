# %%
import pandas as pd
import matplotlib.pyplot as plt
import pathlib

# %%
PROJECT_PATH = pathlib.Path(__file__).parent.parent.resolve()
DATA_PATH = PROJECT_PATH / "data"
print(f"{PROJECT_PATH =}")
print(f"{DATA_PATH =}")
# %%
df0 = pd.read_csv(DATA_PATH / "global_sales_data_logic_distribution.csv")
df0.columns = [col.lower().replace(" ", "_") for col in df0.columns]
df0.rename(
    columns={
        "order_id": "order_id",
        "order_date": "date",
        "region": "region",
        "country": "country",
        "customer_name": "customer",
        "product_category": "category",
        "product_name": "name",
        "asp": "asp",
        "quantity": "qty",
        "total_revenue": "rev",
    },
    inplace=True,
)
df0["date"] = pd.to_datetime(df0["date"], format="%Y-%m-%d")
df0["month"] = df0["date"].dt.month
df0["year"] = df0["date"].dt.year
print(df0.head())

# %%
df0.groupby(["month"])["qty"].sum()

# %%
df0.groupby(["year"])["qty"].sum()
