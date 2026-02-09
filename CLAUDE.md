# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Text-to-SQL multi-agent system built on Google ADK (Agent Development Kit). Converts natural language queries to SQL, executes against SQLite, and optionally renders charts using Plotly. Uses local LLM (Qwen3-4B via llama.cpp).

## Common Commands

```bash
# Run Web UI (Streamlit)
uv run streamlit run web_ui/app.py

# Run evaluation (LLM-as-Judge scoring)
uv run python eval/eval.py

# Run end-to-end evaluation (SQL + Plot + Chart validation)
uv run python eval/eval_e2e.py
uv run python eval/eval_e2e.py --limit 5        # Test subset
uv run python eval/eval_e2e.py --retry-failed   # Retry only failed cases

# Run agent directly
uv run python scripts/run_planned_agent.py

# Docker development
docker compose -f docker-compose.dev.yml up --build
```

## Architecture

### Agent Hierarchy (Google ADK patterns)
```
root_agent (LlmAgent)
└── full_workflow (SequentialAgent)
    ├── query_and_answer (SequentialAgent)
    │   ├── get_data_agent (LoopAgent, max 3 retries)
    │   │   ├── sql_gen_sequential (SequentialAgent)
    │   │   │   ├── region_country_check (LlmAgent) - alias mapping
    │   │   │   └── sql_gen (LlmAgent) - SQL generation
    │   │   └── sql_critic_sequential (SequentialAgent)
    │   │       ├── check_sql (LlmAgent) - syntax check
    │   │       ├── execute_sql (LlmAgent) - run query
    │   │       └── critic (LlmAgent) - validate results, calls exit_loop to break
    │   └── answer (LlmAgent) - business language response
    ├── plot_decision (LlmAgent, structured output) - decide if chart needed
    ├── plot_render (LlmAgent) - render Plotly chart
    └── final_echo (LlmAgent) - return answer to user
```

### Key Design Patterns
- **LoopAgent + exit_loop tool**: Self-correction mechanism. Critic agent calls `exit_loop` to break retry loop when results are valid.
- **Structured Output**: `PlotDecision` and `JudgeResult` use Pydantic schemas for reliable LLM outputs.
- **State propagation**: Agents share data via `tool_context.state` (sql_query, raw_query_result, column_names, chart_json).

### Core Files
- `src/text2sql/agents/planned_agent/agent.py` - Main agent definitions and prompts
- `src/text2sql/agents/planned_agent/tools.py` - SQL execution, chart rendering
- `src/text2sql/agents/judge_agent/agent.py` - LLM-as-Judge evaluation agent
- `src/text2sql/config.py` - Environment config (LLAMA_CPP_* vars)

### Database
- SQLite database at `data/global_sales_data.sqlite`
- Single table: `sales_data` with columns: Order ID, Order Date, Region, Country, Customer Name, Product Category, Product Name, ASP, Quantity, Total Revenue

## Evaluation System

Two evaluation approaches:
1. `eval/eval.py` - SQL-only evaluation using Judge Agent (eval_set_v2.jsonl)
2. `eval/eval_e2e.py` - Full pipeline: SQL + plot decision + chart generation (eval_set_v3.jsonl)

Results saved to `eval/results/<timestamp>/` with JSON and CSV outputs.

## Environment Variables

```
LLAMA_CPP_MODEL=openai/qwen3-4b-instruct-2507
LLAMA_CPP_API_BASE=http://localhost:8081
LLAMA_CPP_VL_MODEL=openai/Qwen3VL-4B-Instruct-Q4_K_M.gguf  # Vision model
LLAMA_CPP_VL_API_BASE=http://localhost:8082
MAX_RETRIES=3
```

## Web UI Pages

- `1_Chat.py` - Main chat interface with agent execution tracing
- `2_Eval.py` - View evaluation results
- `3_System_Monitor.py` - GPU monitoring
- `4_Data_Dashboard.py` - Data visualization
- `5_Image_Read_Chat.py` - Vision language model demo
