import sqlite3

from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm

from pydantic import BaseModel, Field
from pydantic import BaseModel, field_validator
import sqlparse
from google.adk import tools

DB_PATH = "/home/mix060514/pj/text2sql/data/global_sales_data.sqlite"


class QueryModel(BaseModel):
    sql_query: str

    @field_validator("sql_query")
    @classmethod
    def check_sql_syntax(cls, v: str) -> str:
        parsed = sqlparse.parse(v)
        if not parsed:
            raise ValueError("Invalid SQL syntax")
        return v


def query_sql(sql):
    """execute sql query"""
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute(sql)
    return cur.fetchall()


sql_tool = tools.FunctionTool.from_function(get_sql)

root_agent = LlmAgent(
    model=LiteLlm(
        model="openai/qwen3-4b-instruct-2507",
        api_key="aaa",
        api_base="http://localhost:8081",
    ),
    name="root_agent",
    description="Provides weather information for specific cities.",
    instruction="You are a helpful weather assistant. "
    "When the user asks for the weather in a specific city, "
    "use the 'get_weather' tool to find the information. "
    "If the tool returns an error, inform the user politely. "
    "If the tool is successful, present the weather report clearly.",
    tools=[sql_tool],
)
