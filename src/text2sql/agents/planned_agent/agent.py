from typing import List, Dict, Any, Optional
import os

from pydantic import BaseModel, Field
from google.adk.agents import LlmAgent, LoopAgent, SequentialAgent
from google.adk.tools import ToolContext
from google.adk.models.lite_llm import LiteLlm

from .tools import check_sql_syntax, execute_sql

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
MAX_RETRIES = 3
ENABLE_SELF_CORRECTION = True
LLAMA_CPP_MODEL = "openai/qwen3-4b-instruct-2507"
# LLAMA_CPP_MODEL = "openai/qwen3-14b-Q4_K_M"
LLAMA_CPP_API_KEY = "aaa"
LLAMA_CPP_API_BASE = os.getenv("LLAMA_CPP_API_BASE", "http://localhost:8081")


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
SCHEMA_DESCRIPTION = """
    **Database Schema Information:**
    
    [CRITICAL] TABLE NAME: `sales_data` (DO NOT use 'sales', 'orders', or any other name)
    
    **Columns in `sales_data`:**
    - `Order ID` (TEXT): Unique transaction ID.
      * RULE: For "Order Count" (訂單數), usage: `COUNT(DISTINCT "Order ID")`.
      
    - `Order Date` (TEXT): Format 'YYYY-MM-DD'.
    
    - `Region` (TEXT):
      * Values: 'North America', 'EMEA', 'APAC', 'LATAM'
      
    - `Country` (TEXT): e.g., 'United States', 'Germany', 'China', 'India'.
    - `Customer Name` (TEXT): Name of the customer company.
    - `Product Category` (TEXT): e.g., 'Electronics', 'Software'.
    - `Product Name` (TEXT): Specific product name.

    - `ASP` (REAL): Average Selling Price (Unit Price).
    
    - `Quantity` (INTEGER): Units sold.
      * RULE: For "Sales Volume" (銷售量), usage: `SUM("Quantity")`.
      
    - `Total Revenue` (REAL): Sales amount.
      * RULE: For "Revenue" (銷售額), usage: `SUM("Total Revenue")`.
"""


region_country_check_agent = LlmAgent(
    model=LiteLlm(
        model=LLAMA_CPP_MODEL,
        api_key=LLAMA_CPP_API_KEY,
        api_base=LLAMA_CPP_API_BASE,
    ),
    name="region_country_check",
    description="A region and country check agent that checks the region and country from the user question is in database or not.",
    instruction=f"""You are a region and country check agent that checks the region and country from the user question.
    
    {REGION_ALIASES}

    return what region and country should use in sql query, if not related with region and country, return None""
    """,
    output_key="region_country",
)


sql_gen_agent = LlmAgent(
    model=LiteLlm(
        model=LLAMA_CPP_MODEL,
        api_key=LLAMA_CPP_API_KEY,
        api_base=LLAMA_CPP_API_BASE,
    ),
    name="sql_gen",
    description="A SQL specialist with structured output and self-correction capabilities.",
    instruction=f"""You are a precise SQL specialist using SQLite.

    {SCHEMA_DESCRIPTION}

    **CRITICAL CONCEPT MAPPING (MUST FOLLOW):**
    
    1. **"訂單數量" (Order Count / How many orders)**
       - Concept: Counting unique transactions.
       - Keyword matches: 訂單數, 幾筆, 多少單, Transaction count.
       - SQL Action: `COUNT(DISTINCT "Order ID")`
       - ❌ WRONG: SUM("Quantity")
       
    2. **"銷售量" (Sales Volume / How many units)**
       - Concept: Summing physical items sold.
       - Keyword matches: 銷售量, 賣出多少個, 銷量, Units sold.
       - SQL Action: `SUM("Quantity")`
       - ❌ WRONG: COUNT("Order ID")

    3. **"銷售額" (Revenue / Sales Amount)**
       - Concept: Total money earned.
       - SQL Action: `SUM("Total Revenue")`
       
    4. **"YoY Growth Rate %" (Year over Year Growth Rate %)**
       - Concept: Year over year growth rate, the xx% not 0.xx.
       - SQL Action: `SELECT 
        ((SUM(CASE WHEN strftime('%Y', "Order Date") = '2024' THEN "Total Revenue" ELSE 0 END) - 
          SUM(CASE WHEN strftime('%Y', "Order Date") = '2023' THEN "Total Revenue" ELSE 0 END)) * 100.0 / 
          NULLIF(SUM(CASE WHEN strftime('%Y', "Order Date") = '2023' THEN "Total Revenue" ELSE 0 END), 0)) 
        as yoy_growth_pct`
       - Output: YoY growth rate.

    **SQL Examples:**
    1. [Revenue] Total sales in North America:
       SELECT SUM("Total Revenue") FROM sales_data WHERE Region = 'North America';

    2. [Sales Volume] Top 3 products by QUANTITY (銷售量) in EMEA:
       SELECT "Product Name", SUM("Quantity") as total_qty FROM sales_data WHERE Region = 'EMEA' GROUP BY "Product Name" ORDER BY total_qty DESC LIMIT 3;
       
    3. [Order Count] How many orders (訂單數量) in 2023?
       SELECT COUNT(DISTINCT "Order ID") FROM sales_data WHERE strftime('%Y', "Order Date") = '2023';
       
    the country or region should be used: {{region_country?}}
    critic_result: {{critic_result?}}
    """,
    output_key="sql_query",
)

sql_gen_sequential_agent = SequentialAgent(
    name="sql_gen_sequential",
    description="agent that gen sql query for further use",
    sub_agents=[region_country_check_agent, sql_gen_agent],
)

check_sql_agent = LlmAgent(
    model=LiteLlm(
        model=LLAMA_CPP_MODEL,
        api_key=LLAMA_CPP_API_KEY,
        api_base=LLAMA_CPP_API_BASE,
    ),
    name="check_sql",
    description="A SQL checker that checks the SQL query for errors.",
    instruction="""You are a SQL checker that checks the SQL query for errors.
    use the tool check_sql_syntax to use sql parser to check the SQL query for errors.
    
    {sql_query}
    """,
    tools=[check_sql_syntax],
)

execute_sql_agent = LlmAgent(
    model=LiteLlm(
        model=LLAMA_CPP_MODEL,
        api_key=LLAMA_CPP_API_KEY,
        api_base=LLAMA_CPP_API_BASE,
    ),
    name="execute_sql",
    description="A SQL executor that executes the SQL query against the SQLite database.",
    instruction="""You are a SQL executor that executes the SQL query against the SQLite database.
    execute this:
    {sql_query}
    """,
    tools=[execute_sql],
    include_contents="none",
    output_key="query_result",
)


def exit_loop(tool_context: ToolContext):
    """Call this function ONLY when the critique indicates no further changes are needed, signaling the iterative process should end."""
    #  print(f"  [Tool Call] exit_loop triggered by {tool_context.agent_name}")
    tool_context.actions.escalate = True
    tool_context.actions.skip_summarization = True
    # Return empty dict as tools should typically return JSON-serializable output
    return {}


critic_agent = LlmAgent(
    model=LiteLlm(
        model=LLAMA_CPP_MODEL,
        api_key=LLAMA_CPP_API_KEY,
        api_base=LLAMA_CPP_API_BASE,
    ),
    name="critic",
    description="You are a Constructive Critic AI reviewing query result whether it can answer the user question (typically 2-6 sentences).",
    instruction="""You are a critic that checks the SQL query for errors.

    query: {sql_query}
    query_result: {query_result}
    is query correct and can answer user question?
    if yes, you must call exit_loop tool.
    if no, return what should be fixed and give the feedback to the next round sql gen agent.
    """,
    output_key="critic_result",
    tools=[exit_loop],
)


sql_critic_sequential_agent = SequentialAgent(
    name="sql_critic",
    description="A SQL critic that checks the SQL query for errors. execute sql and check is the result can answer the user question, if not, return the error message and the corrected sql query",
    sub_agents=[check_sql_agent, execute_sql_agent, critic_agent],
)


get_data_agent = LoopAgent(
    name="get_data_agent",
    sub_agents=[sql_gen_sequential_agent, sql_critic_sequential_agent],
    max_iterations=MAX_RETRIES,
)

answer_agent = LlmAgent(
    model=LiteLlm(
        model=LLAMA_CPP_MODEL,
        api_key=LLAMA_CPP_API_KEY,
        api_base=LLAMA_CPP_API_BASE,
    ),
    name="answer",
    #  description="你是一個商業專家，回答使用者的問題",
    #  instruction="根據查詢的結果，回答使用者的問題, 請用繁體中文回答總結，但是產品類別、產品名稱資訊，請用英文回答。",
    description="你是一個商業專家，回答使用者的問題",
    instruction="""
    請根據查詢結果 {query_result}，以商業專家的口吻回答使用者問題。

    **回答規範**：
    1. **語言**：核心敘述使用**繁體中文**，但「產品類別」與「產品名稱」請保留**英文原文**。
    2. **拒絕幻覺**：**嚴禁**列出結果中不存在的欄位（若結果只有金額，不要強行加上「產品名稱」）。

    **數字呈現規則 (重要)**：
    針對金額或銷售量等大數字，請**同時**提供以下兩種格式：
    1. **精確數值**：務必加上千分位符號 (e.g., 25,286,700.15)。
    2. **概略縮寫**：使用 K/M/B 進行縮寫，並加上波浪號 (e.g., ~25.3M)。
    
    **回答範例**：
    - 正確：「亞太區的總銷售額為 25,286,700.15 (~25.3M)。」
    - 錯誤：「銷售額是 25286700。」(未格式化)
    - 錯誤：「產品名稱：N/A，金額：25M。」(出現不存在的欄位)
    """,
    output_key="final_answer",
)


query_and_answer_agent = SequentialAgent(
    name="query_and_answer_agent",
    sub_agents=[get_data_agent, answer_agent],
)


# ------------------------------------------------------------------------------
# Root Agent: Dispatcher & Responder with Structured Output (V7)
# ------------------------------------------------------------------------------


root_agent = LlmAgent(
    model=LiteLlm(
        model=LLAMA_CPP_MODEL,
        api_key=LLAMA_CPP_API_KEY,
        api_base=LLAMA_CPP_API_BASE,
    ),
    name="root_agent",
    description="Root agent that delegates SQL query tasks to query_and_answer_agent.",
    instruction="""You are a helpful assistant that delegates SQL database queries to a specialized sub-agent.

**Your only job**: Delegate the user's question to `query_and_answer_agent` and let it handle everything.

**Workflow:**
1. When user asks a question about data, immediately transfer to `query_and_answer_agent`
2. The sub-agent will:
   - Generate SQL query
   - Execute and validate the query
   - Return the final answer
3. After the sub-agent completes, return its answer to the user

**CRITICAL**: Do NOT try to answer data questions yourself. Always delegate to the sub-agent.
    """,
    sub_agents=[query_and_answer_agent],
)
