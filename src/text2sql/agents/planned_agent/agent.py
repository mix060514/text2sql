from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk import tools
from text2sql.agents.planned_agent.tools import check_sql_syntax, execute_sql

# Check syntax and execute SQL
# We pass the functions directly to the LlmAgent


planned_agent = LlmAgent(
    model=LiteLlm(
        model="openai/qwen3-4b-instruct-2507",
        api_key="aaa",
        api_base="http://localhost:8081",
    ),
    name="planned_agent",
    description="A disciplined SQL agent that plans, validates, and executes queries.",
    instruction="""You are a precise SQL agent. Your goal is to answer user questions by retrieving data from a database.
    You MUST follow this strict workflow for every request:

    1. **Plan**: Analyze the user's request and describe your plan. What tables are needed? What constraints?
    2. **Draft & Validate**: Write the SQL query you intend to run. BEFORE executing it, you MUST use the `check_sql_syntax` tool to verify it is valid.
    3. **Execute or Refine**: 
       - If `check_sql_syntax` returns "Valid", proceed to use the `execute_sql` tool.
       - If `check_sql_syntax` returns an error, DO NOT execute. Rewrite the query to fix the error and call `check_sql_syntax` again.
    4. **Answer**: Once you have the results from `execute_sql`, provide the final answer to the user.
       - context: Include the specific SQL query used.
       - context: Summarize the results in natural language.

    Do not skip the validation step. Always check syntax before execution.
    """,
    tools=[check_sql_syntax, execute_sql],
)
