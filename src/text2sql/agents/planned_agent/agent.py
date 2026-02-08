from typing import List, Dict, Any, Optional
import os

from pydantic import BaseModel, Field
from google.adk.agents import LlmAgent, LoopAgent, SequentialAgent
from google.adk.tools import ToolContext
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.lite_llm import LiteLlm

from .tools import check_sql_syntax, execute_sql, render_chart
from text2sql.config import (
    MAX_RETRIES,
    LLAMA_CPP_MODEL,
    LLAMA_CPP_API_KEY,
    LLAMA_CPP_API_BASE,
)

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


# ------------------------------------------------------------------------------
# Callback to capture user question
# ------------------------------------------------------------------------------
def capture_user_question(callback_context: CallbackContext):
    """Captures user's question into session state before agent runs."""
    if callback_context.user_content and callback_context.user_content.parts:
        user_text = callback_context.user_content.parts[0].text
        if user_text:
            callback_context.state["user_question"] = user_text
    return None  # Continue with agent execution


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

    5. **[CRITICAL] Time Series Queries (Quarterly/Monthly Trends)**:
       When the query involves "Year + Quarter" or "Year + Month" trends for visualization:
       - MUST create a COMBINED column as the time axis (e.g., "Period" or "YearQuarter")
       - DO NOT output Year and Quarter as separate columns
       - This ensures proper X-axis ordering in charts
       
       **Correct Example (Year + Quarter trend):**
       ```sql
       SELECT 
           strftime('%Y', "Order Date") || '-' || 
           CASE 
               WHEN strftime('%m', "Order Date") IN ('01','02','03') THEN 'Q1'
               WHEN strftime('%m', "Order Date") IN ('04','05','06') THEN 'Q2'
               WHEN strftime('%m', "Order Date") IN ('07','08','09') THEN 'Q3'
               ELSE 'Q4' 
           END AS Period,
           SUM("Total Revenue") AS Revenue
       FROM sales_data
       WHERE strftime('%Y', "Order Date") IN ('2023', '2024')
       GROUP BY Period
       ORDER BY Period;
       ```
       
       **Correct Example (Year + Month trend):**
       ```sql
       SELECT 
           strftime('%Y-%m', "Order Date") AS Period,
           SUM("Total Revenue") AS Revenue
       FROM sales_data
       GROUP BY Period
       ORDER BY Period;
       ```
       
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
    query_result: {raw_query_result}
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
    description="你是一個商業專家，回答使用者的問題",
    instruction="""
    請根據查詢結果，以商業專家的口吻回答使用者問題。

    使用者原始問題: {user_question}

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
    
    query_result: {raw_query_result}
    """,
    output_key="answer",
)

query_and_answer_agent = SequentialAgent(
    name="query_and_answer",
    description="Agent that queries data and answers the question",
    sub_agents=[get_data_agent, answer_agent],
)


# ------------------------------------------------------------------------------
# Plot Decision Agent
# ------------------------------------------------------------------------------


class PlotDecision(BaseModel):
    """Structured output for plot decision"""

    should_plot: bool = Field(description="Whether to create a chart visualization")
    chart_type: str | None = Field(
        description="Chart type: 'bar', 'line', 'pie', or null if should_plot is false"
    )
    x_column: str | None = Field(
        description="Name of the column to use as X-axis (labels/categories)"
    )
    y_column: str | None = Field(
        description="Name of the column to use as Y-axis (values/metrics)"
    )
    color_column: str | None = Field(
        description="Name of the column to use for grouping/coloring (optional). E.g. 'Year' or 'Region' for comparisons."
    )
    reason: str = Field(description="Brief reason for the decision")


def ensure_chart_context(callback_context: Any, **kwargs: Any) -> None:
    """Ensure necessary context variables exist in state to prevent prompt formatting errors."""
    state = callback_context.state
    if "column_names" not in state:
        state["column_names"] = []
    if "raw_query_result" not in state:
        state["raw_query_result"] = []
    if "user_question" not in state:
        state["user_question"] = ""
    return None


plot_decision_agent = LlmAgent(
    model=LiteLlm(
        model=LLAMA_CPP_MODEL,
        api_key=LLAMA_CPP_API_KEY,
        api_base=LLAMA_CPP_API_BASE,
    ),
    name="plot_decision",
    before_agent_callback=ensure_chart_context,
    description="Decides whether to visualize query results with a chart",
    instruction="""You are a visualization decision agent. Based on the user's question and query results, decide:
1. Whether a chart would be helpful (should_plot: true/false)
2. If yes, which chart type is most appropriate
3. Select columns for X-axis, Y-axis, and optional Color grouping (hue).

**Available Data:**
- Columns: {column_names}
- Preview: {raw_query_result}

**CRITICAL: When to plot?**
- **YES**: Comparison of multiple items, trends over time, or distribution/proportions.
- **NO**: Single values (e.g. "Total Sales", "Top 1 Product"), specific lookup (e.g. "Who is..."), or Lists where visualization adds no value.

**Column Selection Rules:**
- **x_column**: The categorical dimension (e.g., Month, Region, Product).
- **y_column**: The numeric metric (e.g., Revenue, Count).
- **color_column**: Use this if there is a SECOND categorical dimension (e.g. comparing trends across 'Years' or 'Regions').

**Chart Type Guidelines:**
- **pie**: Proportions (e.g. "佔比").
- **line**: Time trends (e.g. "趨勢").
- **bar**: Comparisons/Rankings (e.g. "比較").

**Examples:**
- "各區域銷售額佔比" 
  -> plot: true, type: "pie", x: "Region", y: "Total Revenue"
- "2023-2024 月銷售趨勢" (Columns: Month, Revenue, Year)
  -> plot: true, type: "line", x: "Month", y: "Revenue", color: "Year"
- "北美的總銷售額是多少"
  -> plot: false (Single value)
- "銷售量最高的產品是哪個"
  -> plot: false (Finding the top one)

User question: {user_question}
""",
    output_schema=PlotDecision,
    output_key="plot_decision",
)


def render_chart_tool(tool_context: ToolContext) -> dict:
    """Tool to render chart based on plot_decision and query_result in session state."""
    import json

    plot_decision = tool_context.state.get("plot_decision", {})
    raw_query_result = tool_context.state.get("raw_query_result")

    chart_type = plot_decision.get("chart_type", "bar")
    x_col = plot_decision.get("x_column")
    y_col = plot_decision.get("y_column")
    color_col = plot_decision.get("color_column")
    column_names = tool_context.state.get("column_names", [])

    if not raw_query_result:
        return {"success": False, "error": "No query result available to render."}

    if not column_names:
        # Fallback if no column names captured (unlikely with new execute_sql)
        return {"success": False, "error": "Missing column names for charting."}

    # Convert list of tuples to list of dicts for Plotly
    # e.g. columns=["Region", "Sales"], result=[("NA", 100)] -> [{"Region": "NA", "Sales": 100}]
    data_list = []
    try:
        for row in raw_query_result:
            row_dict = {}
            for i, col in enumerate(column_names):
                if i < len(row):
                    row_dict[col] = row[i]
            data_list.append(row_dict)

        result = render_chart(
            chart_type=chart_type,
            data=data_list,
            x_col=x_col,
            y_col=y_col,
            color_col=color_col,
        )

        if result.get("success"):
            tool_context.state["chart_image"] = result["image"]

            # Store chart data for eval validation
            tool_context.state["chart_data"] = {
                "chart_type": chart_type,
                "x_column": x_col,
                "y_column": y_col,
                "color_column": color_col,
                "x_values": [row.get(x_col) for row in data_list],
                "y_values": [row.get(y_col) for row in data_list],
                "data_rows": len(data_list),
            }

            # Return a summary to the LLM to save context tokens
            return {"success": True, "message": "Chart generated and stored in state."}
        return result
    except Exception as e:
        return {"success": False, "error": f"Chart rendering internal error: {str(e)}"}


plot_render_agent = LlmAgent(
    model=LiteLlm(
        model=LLAMA_CPP_MODEL,
        api_key=LLAMA_CPP_API_KEY,
        api_base=LLAMA_CPP_API_BASE,
    ),
    name="plot_render",
    description="Renders charts when visualization is needed",
    instruction="""If plot_decision indicates should_plot is true, call render_chart_tool to create the chart.
    
    plot_decision: {plot_decision}
    
    If should_plot is false, just respond that no chart is needed.
    """,
    tools=[render_chart_tool],
    output_key="chart_result",
)


# ------------------------------------------------------------------------------
# Root Agent: Dispatcher & Responder with Structured Output (V7)
# ------------------------------------------------------------------------------

# Sequential agent that handles the full workflow: query -> answer -> plot decision -> render
full_workflow_agent = SequentialAgent(
    name="full_workflow",
    description="Complete workflow: query data, answer question, decide if chart needed, render chart",
    sub_agents=[query_and_answer_agent, plot_decision_agent, plot_render_agent],
)


root_agent = LlmAgent(
    model=LiteLlm(
        model=LLAMA_CPP_MODEL,
        api_key=LLAMA_CPP_API_KEY,
        api_base=LLAMA_CPP_API_BASE,
    ),
    name="root_agent",
    before_agent_callback=capture_user_question,
    description="Root agent that delegates SQL query tasks and visualization.",
    instruction="""You are a helpful assistant that delegates SQL database queries to a specialized sub-agent.

**Your only job**: Delegate the user's question to `full_workflow` and let it handle everything.

**Workflow:**
1. When user asks a question about data, immediately transfer to `full_workflow`
2. The sub-agent will:
   - Generate and execute SQL query
   - Provide a text answer
   - Decide if visualization is helpful
   - Render a chart if needed
3. After the sub-agent completes, return its answer to the user

**CRITICAL**: Do NOT try to answer data questions yourself. Always delegate to the sub-agent.
    """,
    sub_agents=[full_workflow_agent],
)
