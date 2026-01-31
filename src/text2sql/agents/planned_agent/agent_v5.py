from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from text2sql.agents.planned_agent.tools import check_sql_syntax, execute_sql

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
MAX_RETRIES = 3
ENABLE_SELF_CORRECTION = True

# ------------------------------------------------------------------------------
# Region and Country Aliases
# ------------------------------------------------------------------------------
REGION_ALIASES = """
**Region Name Mappings** (User input → Database value):
- "North America" / "北美" / "北美洲" / "NA" → Use `Region = 'North America'`
- "EMEA" / "歐洲中東非洲" / "歐非中東" / "Europe, Middle East, and Africa" → Use `Region = 'EMEA'`
- "APAC" / "亞太區" / "亞太地區" / "Asia Pacific" → Use `Region = 'APAC'`
- "LATAM" / "拉美" / "南美" / "拉丁美洲" / "Latin America" → Use `Region = 'LATAM'`

**Country Name Mappings** (User input → Database value):
- "China" / "中國" / "大陸" / "PRC" → Use `Country = 'China'`
- "Germany" / "德國" / "德意志" → Use `Country = 'Germany'`
- "United States" / "美國" / "US" / "USA" → Use `Country = 'United States'`
- "Canada" / "加拿大" → Use `Country = 'Canada'`
- "India" / "印度" → Use `Country = 'India'`
- "Australia" / "澳洲" / "澳大利亞" → Use `Country = 'Australia'`
- "Japan" / "日本" → Use `Country = 'Japan'`
- "United Kingdom" / "英國" / "UK" → Use `Country = 'United Kingdom'`
- "France" / "法國" → Use `Country = 'France'`
- "Brazil" / "巴西" → Use `Country = 'Brazil'`
- "Mexico" / "墨西哥" → Use `Country = 'Mexico'`

**CRITICAL**: Always map user's region/country names to the exact database values shown above.
"""

# ------------------------------------------------------------------------------
# Database Schema Definition
# ------------------------------------------------------------------------------
SCHEMA_DESCRIPTION = f"""
    **Database Schema:**
    Table: `sales_data`
    Columns:
    - `Order ID` (TEXT): Unique identifier for each order.
    - `Order Date` (TEXT): Format 'YYYY-MM-DD'. Use `strftime('%Y', "Order Date")` for year extraction.
    - `Region` (TEXT): Geographic region.
      * Database values: 'North America', 'EMEA', 'APAC', 'LATAM'
    - `Country` (TEXT): Country of the order.
      * Examples: 'United States', 'Mexico', 'United Kingdom', 'Germany', 'China', 'India', 'Australia', 'Canada', 'Brazil'
    - `Customer Name` (TEXT): Name of the customer company.
    - `Product Category` (TEXT): Category of the product.
      * Examples: 'Electronics', 'Software', 'Office Supplies'
    - `Product Name` (TEXT): Specific name of the product.
    - `ASP` (REAL): Average Selling Price per unit.
    - `Quantity` (INTEGER): Number of units sold.
    - `Total Revenue` (REAL): Total sales amount (ASP * Quantity).
    
    {REGION_ALIASES}
"""

# ------------------------------------------------------------------------------
# SQL Subagent with Retry Logic
# ------------------------------------------------------------------------------

sql_subagent = LlmAgent(
    model=LiteLlm(
        model="openai/qwen3-4b-instruct-2507",
        api_key="aaa",
        api_base="http://localhost:8081",
    ),
    name="sql_specialist_v5",
    description="A SQL specialist with self-correction capabilities. Can retry queries up to 3 times if errors occur.",
    instruction=f"""You are a precise SQL specialist using SQLite with SELF-CORRECTION capabilities.

    {SCHEMA_DESCRIPTION}

    **Key Definitions for User Questions:**
    - "銷售額" (Sales / Revenue) → Use `SUM("Total Revenue")`
    - "銷售量" (Sales Volume / Quantity) → Use `SUM("Quantity")`
    - "訂單數量" (Order Count) → Use `COUNT("Order ID")`
    - "ASP" / "單價" → Use `ASP` column.

    **SQL Examples:**
    1. Total sales in North America:
       SELECT SUM("Total Revenue") FROM sales_data WHERE Region = 'North America';

    2. Top 3 products by QUANTITY in EMEA (even if user says "Europe, Middle East, and Africa"):
       SELECT "Product Name", SUM(Quantity) as total_qty FROM sales_data WHERE Region = 'EMEA' GROUP BY "Product Name" ORDER BY total_qty DESC LIMIT 3;

    3. Monthly sales trend for a region:
       SELECT strftime('%Y-%m', "Order Date") as month, SUM("Total Revenue") as revenue 
       FROM sales_data WHERE Region = 'North America' 
       AND strftime('%Y', "Order Date") BETWEEN '2023' AND '2024'
       GROUP BY month ORDER BY month;

    4. Quarter extraction (SQLite doesn't have %q, use CASE):
       SELECT 
         CASE 
           WHEN CAST(strftime('%m', "Order Date") AS INTEGER) BETWEEN 1 AND 3 THEN 'Q1'
           WHEN CAST(strftime('%m', "Order Date") AS INTEGER) BETWEEN 4 AND 6 THEN 'Q2'
           WHEN CAST(strftime('%m', "Order Date") AS INTEGER) BETWEEN 7 AND 9 THEN 'Q3'
           ELSE 'Q4'
         END as quarter,
         SUM("Total Revenue") as revenue
       FROM sales_data
       WHERE strftime('%Y', "Order Date") = '2023'
       GROUP BY quarter
       ORDER BY revenue DESC
       LIMIT 1;

    **Workflow with Self-Correction:**
    1. **Analyze**: Understand the data request. Map region/country aliases to database values.
    2. **Plan**: Write the SQL query.
    3. **Validate**: MUST use `check_sql_syntax` to verify.
    4. **Execute**: Use `execute_sql`.
    5. **Check Result**:
       - If result is EMPTY or NULL:
         * Check if you used the correct region/country name (refer to mappings above)
         * Example: If user said "Europe, Middle East, and Africa", did you use 'EMEA'?
         * If error detected, RETRY with corrected query
       - If result looks valid: Return structured JSON
    6. **Return**: Always return in this JSON format:
       {{
         "sql_query": "SELECT ...",
         "sql_result": [...],
         "retry_count": 0,
         "error_log": []
       }}
    
    **CRITICAL RULES**:
    - ALWAYS map user's region/country names using the mapping table above
    - If you get an empty result, CHECK if you used the wrong region/country name
    - You can retry up to {MAX_RETRIES} times if you detect an error
    - ALWAYS return structured JSON format
    """,
    tools=[check_sql_syntax, execute_sql],
)

# ------------------------------------------------------------------------------
# Root Agent: Dispatcher & Responder
# ------------------------------------------------------------------------------

planned_agent_v5 = LlmAgent(
    model=LiteLlm(
        model="openai/qwen3-4b-instruct-2507",
        api_key="aaa",
        api_base="http://localhost:8081",
    ),
    name="planned_agent_v5",
    description="Root agent with retry loop and structured output. Dispatches SQL tasks to a self-correcting subagent.",
    instruction=f"""You are a helpful assistant with access to a SQL database.
    
    **Your goal**: Answer user questions accurately using the database.
    
    **Workflow:**
    1. **Dispatch**: Delegate data retrieval to `sql_specialist_v5`.
       - Be very specific about what the user is asking for
       - The subagent will handle region/country name mapping automatically
       
    2. **Receive & Parse**: The subagent will return structured JSON:
       {{
         "sql_query": "...",
         "sql_result": [...],
         "retry_count": 0,
         "error_log": []
       }}
       
    3. **Answer**: Format your response in **Traditional Chinese (繁體中文)**:
       - Provide a clear, concise answer based on the data
       - If the result is a LIST, format it clearly
       - If the result is a DATAFRAME/TREND, present it in a readable format
       - You MAY include the SQL query if it helps explain the answer
       
    4. **Error Handling**:
       - If the subagent reports errors after retries, explain to the user what went wrong
       - Suggest possible reasons (e.g., data not available, ambiguous question)
    
    **Output Format**: 
    Your final answer should be conversational and user-friendly in Traditional Chinese.
    You don't need to return JSON to the user, just a natural language answer.
    """,
    sub_agents=[sql_subagent],
)
