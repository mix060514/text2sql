from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from text2sql.agents.planned_agent.tools import check_sql_syntax, execute_sql

# ------------------------------------------------------------------------------
# Subagent: Manages Schema & SQL
# ------------------------------------------------------------------------------

# Database Schema Definition
SCHEMA_DESCRIPTION = """
    **Database Schema:**
    Table: `sales_data`
    Columns:
    - `Order ID` (TEXT): Unique identifier for each order.
    - `Order Date` (TEXT): Format 'YYYY-MM-DD'. Use `strftime('%Y', "Order Date")` for year extraction.
    - `Region` (TEXT): Geographic region.
      * Examples: 'North America', 'EMEA', 'APAC', 'LATAM'
    - `Country` (TEXT): Country of the order.
      * Examples: 'United States', 'Mexico', 'United Kingdom', 'Germany', 'China', 'India', 'Australia', 'Canada', 'Brazil'
    - `Customer Name` (TEXT): Name of the customer company.
    - `Product Category` (TEXT): Category of the product.
      * Examples: 'Electronics', 'Software', 'Office Supplies'
    - `Product Name` (TEXT): Specific name of the product.
    - `ASP` (REAL): Average Selling Price per unit.
    - `Quantity` (INTEGER): Number of units sold.
    - `Total Revenue` (REAL): Total sales amount (ASP * Quantity).
"""

sql_subagent = LlmAgent(
    model=LiteLlm(
        model="openai/qwen3-4b-instruct-2507",
        api_key="aaa",
        api_base="http://localhost:8081",
    ),
    name="sql_specialist",
    description="A disciplined SQL specialist agent that manages schema and executes queries. Use this agent for any data retrieval needs.",
    instruction=f"""You are a precise SQL specialist using SQLite.

    {SCHEMA_DESCRIPTION}

    **Key Definitions for User Questions:**
    - "銷售額" (Sales / Revenue) -> Use `SUM("Total Revenue")`
    - "銷售量" (Sales Volume / Quantity) -> Use `SUM("Quantity")`
    - "訂單數量" (Order Count) -> Use `COUNT("Order ID")`
    - "ASP" / "單價" -> Use `ASP` column.

    **SQL Examples:**
    1. Total sales in North America:
       SELECT SUM("Total Revenue") FROM sales_data WHERE Region = 'North America';

    2. Top 3 products by QUANTITY ("銷售量") in North America:
       SELECT "Product Name", SUM(Quantity) as total_qty FROM sales_data WHERE Region = 'North America' GROUP BY "Product Name" ORDER BY total_qty DESC LIMIT 3;

    3. Customer with the most ORDERS ("訂單數量") in Canada:
       SELECT "Customer Name", COUNT("Order ID") as order_count FROM sales_data WHERE Country = 'Canada' GROUP BY "Customer Name" ORDER BY order_count DESC LIMIT 1;
    
    4. Which product has the highest ASP? (Return the Name):
       SELECT "Product Name" FROM sales_data ORDER BY ASP DESC LIMIT 1;

    5. Top 1 Product Category by Revenue for EACH Region:
       WITH Ranked AS (
           SELECT Region, "Product Category", SUM("Total Revenue") as revenue,
                  RANK() OVER (PARTITION BY Region ORDER BY SUM("Total Revenue") DESC) as rnk
           FROM sales_data
           GROUP BY Region, "Product Category"
       )
       SELECT Region, "Product Category", revenue FROM Ranked WHERE rnk = 1;

    **Workflow:**
    1. **Analyze**: Understand the data request. Distinguish between 'Quantity' (items) and 'Order Count' (orders).
    2. **Plan & Validate**: Write the SQL query. MUST use `check_sql_syntax` to verify.
    3. **Execute**: If valid, use `execute_sql`.
    4. **Report**: Return the SQL used and the raw results (rows) to the caller.
    
    Do NOT chat excessively. Just return the data and SQL.
    """,
    tools=[check_sql_syntax, execute_sql],
)

# ------------------------------------------------------------------------------
# Root Agent: Dispatcher & Responder
# ------------------------------------------------------------------------------

planned_agent_v4 = LlmAgent(
    model=LiteLlm(
        model="openai/qwen3-4b-instruct-2507",
        api_key="aaa",
        api_base="http://localhost:8081",
    ),
    name="planned_agent_v4",
    description="Root agent that dispatches SQL tasks to a subagent and formats the final answer.",
    instruction="""You are a helpful assistant.
    
    Your goal is to answer user questions using the database.
    
    **Workflow:**
    1. **Dispatch**: Delegate data retrieval to `sql_specialist`.
       - Be very specific about "Order Count" vs "Quantity" based on the user's Chinese terms.
       
    2. **Answer**: Once you receive the result (SQL + Data) from the subagent:
       - Summarize the answer in **Traditional Chinese (繁體中文)**.
       - You MUST include the SQL query used (provided by the subagent) in your answer context.
       - Provide the answer clearly based on the returned data.
       - If the result is a LIST of items, format them clearly.
       
    If the subagent returns an error, explain it to the user.
    """,
    sub_agents=[sql_subagent],
)
