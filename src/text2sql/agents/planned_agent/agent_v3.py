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
      * Examples: 'United States', 'Mexico', 'United Kingdom', 'France', 'Germany', 'China', 'Japan', 'South Korea', 'India', 'Australia', 'Canada', 'Brazil'
    - `Customer Name` (TEXT): Name of the customer company.
      * Examples: 'Elite Elite Systems', 'Apex Apex Ltd.', 'Summit Beta Solutions', 'Global Next Corp.', 'Delta Global Systems'
    - `Product Category` (TEXT): Category of the product.
      * Examples: 'Electronics', 'Software', 'Office Supplies'
    - `Product Name` (TEXT): Specific name of the product.
      * Examples: 'Docking Station', '4K Monitor 27"', 'Pro Smartphone 15', 'Enterprise Laptop X1', 'Team Collaboration Tool', 'Cloud License (Annual)'
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
    instruction=f"""You are a precise SQL specialist.
    
    {SCHEMA_DESCRIPTION}
    
    **SQL Examples:**
    1. Total sales in North America:
       SELECT SUM("Total Revenue") FROM sales_data WHERE Region = 'North America';

    2. Top 3 products by quantity in North America:
       SELECT "Product Name", SUM(Quantity) as total_qty FROM sales_data WHERE Region = 'North America' GROUP BY "Product Name" ORDER BY total_qty DESC LIMIT 3;

    3. Monthly sales trend for LATAM in 2023:
       SELECT strftime('%Y-%m', "Order Date") as month, SUM("Total Revenue") FROM sales_data WHERE Region = 'LATAM' AND strftime('%Y', "Order Date") = '2023' GROUP BY month ORDER BY month;

    **Workflow:**
    1. **Analyze**: Understand the data request.
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

planned_agent_v3 = LlmAgent(
    model=LiteLlm(
        model="openai/qwen3-4b-instruct-2507",
        api_key="aaa",
        api_base="http://localhost:8081",
    ),
    name="planned_agent_v3",
    description="Root agent that dispatches SQL tasks to a subagent and formats the final answer.",
    instruction="""You are a helpful assistant.
    
    Your goal is to answer user questions using the database.
    
    **Workflow:**
    1. **Dispatch**: When you receive a user question, DO NOT define the schema or write SQL yourself. 
       Instead, delegate the task to the `sql_specialist` subagent.
       Pass a clear description of the data requirement to it.
       
    2. **Answer**: Once you receive the result (SQL + Data) from the subagent:
       - Summarize the answer in **Traditional Chinese (繁體中文)**.
       - You MUST include the SQL query used (provided by the subagent) in your answer context.
       - Provide the answer clearly based on the returned data.
       
    If the subagent returns an error, explain it to the user.
    """,
    sub_agents=[sql_subagent],
)
