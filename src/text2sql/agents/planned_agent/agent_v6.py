from google.adk.agents import LlmAgent, LoopAgent, SequentialAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.tool_context import ToolContext
from src.text2sql.agents.planned_agent.tools import check_sql_syntax, execute_sql

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
MAX_RETRIES = 3

# ------------------------------------------------------------------------------
# State Keys for LoopAgent
# ------------------------------------------------------------------------------
STATE_SQL_QUERY = "sql_query"
STATE_SQL_RESULT = "sql_result"
STATE_RETRY_COUNT = "retry_count"


# ------------------------------------------------------------------------------
# Exit Loop Tool
# ------------------------------------------------------------------------------
def exit_retry_loop(tool_context: ToolContext):
    """Call this function when SQL execution succeeds with valid results."""
    print(f" [Tool Call] exit_retry_loop triggered by {tool_context.agent_name}")
    tool_context.actions.escalate = True
    tool_context.actions.skip_summarization = True
    return {
        "status": "success",
        "message": "SQL execution successful, exiting retry loop",
    }


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
# STEP 1: Initial SQL Generator (runs ONCE to initialize state)
# ------------------------------------------------------------------------------
initial_sql_generator = LlmAgent(
    model=LiteLlm(
        model="openai/qwen3-4b-instruct-2507",
        api_key="aaa",
        api_base="http://localhost:8081",
    ),
    name="initial_sql_generator_v6",
    description="Generates the initial SQL query",
    include_contents="none",
    instruction=f"""You are a SQL query generator.

{SCHEMA_DESCRIPTION}

**Key Definitions:**
- "銷售額" (Sales / Revenue) → Use `SUM("Total Revenue")`
- "銷售量" (Sales Volume / Quantity) → Use `SUM("Quantity")`
- "訂單數量" (Order Count) → Use `COUNT("Order ID")`

**Your Task:**
1. Read the user's question
2. Map region/country names using the mapping table above
3. Generate a syntactically correct SQLite query
4. Use `check_sql_syntax` tool to verify
5. Output ONLY the SQL query as plain text

**CRITICAL**: 
- ALWAYS map region/country aliases correctly
- Return ONLY the SQL query, no explanations
""",
    tools=[check_sql_syntax],
    output_key=STATE_SQL_QUERY,
)

# ------------------------------------------------------------------------------
# STEP 2a: SQL Executor (inside loop)
# ------------------------------------------------------------------------------
sql_executor_in_loop = LlmAgent(
    model=LiteLlm(
        model="openai/qwen3-4b-instruct-2507",
        api_key="aaa",
        api_base="http://localhost:8081",
    ),
    name="sql_executor_v6",
    description="Executes SQL and validates results",
    include_contents="none",
    instruction=f"""You are a SQL executor and validator.

**SQL Query to Execute:**
{{{{sql_query}}}}

**Your Task:**
1. Execute the SQL using `execute_sql` tool
2. Check the result:
   - If result is NOT empty (has data): Call `exit_retry_loop` tool to stop the loop
   - If result is EMPTY: Output "RETRY" to trigger another iteration

**How to Exit Loop:**
When you get valid results, you MUST call the `exit_retry_loop` tool.

Output either:
- Call `exit_retry_loop` if successful
- "RETRY" if empty (to trigger retry)
""",
    tools=[execute_sql, exit_retry_loop],
    output_key=STATE_SQL_RESULT,
)

# ------------------------------------------------------------------------------
# STEP 2b: SQL Refiner (inside loop, only runs if executor said RETRY)
# ------------------------------------------------------------------------------
sql_refiner_in_loop = LlmAgent(
    model=LiteLlm(
        model="openai/qwen3-4b-instruct-2507",
        api_key="aaa",
        api_base="http://localhost:8081",
    ),
    name="sql_refiner_v6",
    description="Refines SQL query based on empty results",
    include_contents="none",
    instruction=f"""You are a SQL query refiner.

{SCHEMA_DESCRIPTION}

**Previous SQL Query:**
{{{{sql_query}}}}

**Previous Result:**
{{{{sql_result}}}}

**Your Task:**
Analyze why the previous query returned empty results.

IF the result says "RETRY":
1. Check if region/country mapping is correct
2. Generate a corrected SQL query
3. Output ONLY the corrected SQL query

ELSE (result is not "RETRY", meaning we have data):
Output the same SQL query (don't change it)

Output ONLY the SQL query.
""",
    tools=[check_sql_syntax],
    output_key=STATE_SQL_QUERY,  # Overwrites the SQL query for next iteration
)

# ------------------------------------------------------------------------------
# STEP 2: Refinement Loop
# ------------------------------------------------------------------------------
refinement_loop = LoopAgent(
    name="sql_refinement_loop_v6",
    description=f"Retry loop for SQL execution (max {MAX_RETRIES} attempts)",
    sub_agents=[
        sql_executor_in_loop,  # Execute first
        sql_refiner_in_loop,  # Then refine if needed
    ],
    max_iterations=MAX_RETRIES,
)

# ------------------------------------------------------------------------------
# STEP 3: Response Formatter
# ------------------------------------------------------------------------------
response_formatter = LlmAgent(
    model=LiteLlm(
        model="openai/qwen3-4b-instruct-2507",
        api_key="aaa",
        api_base="http://localhost:8081",
    ),
    name="response_formatter_v6",
    description="Formats the final answer in Traditional Chinese",
    include_contents="none",
    instruction=f"""You are a helpful assistant formatting SQL query results.

**SQL Query:**
{{{{sql_query}}}}

**SQL Result:**
{{{{sql_result}}}}

**Your Task:**
Format a clear answer in **Traditional Chinese (繁體中文)**:
- Summarize the data in a user-friendly way
- If the result contains data, present it clearly
- If there was an error or empty result, explain what might be wrong

**Output Format**: Natural language answer in Traditional Chinese.
""",
)

# ------------------------------------------------------------------------------
# STEP 4: Sequential Pipeline (Initialize → Loop → Format)
# ------------------------------------------------------------------------------
planned_agent_v6 = SequentialAgent(
    name="planned_agent_v6",
    description="Text-to-SQL agent with LoopAgent-based retry mechanism",
    sub_agents=[
        initial_sql_generator,  # Step 1: Initialize state
        refinement_loop,  # Step 2: Retry loop
        response_formatter,  # Step 3: Format response
    ],
)
