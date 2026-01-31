# Implementation Plan - Text to SQL

## Planned Agent
- [x] Create Planned Agent (`src/text2sql/agents/planned_agent/`)
- [x] Create runner script (`scripts/run_planned_agent.py`)
- [x] Create evaluation script (`eval/eval2.py`)
- [x] Add schema and examples to Planned Agent prompt
- [x] Create Planned Agent V2 with column value examples
- [x] Create unified evaluation script (`eval/eval3.py`)
- [x] Implement Planned Agent V3 with Sub-Agent Architecture
- [x] Create Unified Eval Script (`eval/eval3.py`) and compare V1/V2/V3
- [x] Analyze Eval 3 results (`eval/results/eval3_comparison.json`)
- [x] Create Planned Agent V4 with ambiguity resolution and complex SQL examples
- [x] Create Eval 4 script (`eval/eval4.py`) with improved DataFrame/List comparison logic
- [ ] Run Eval 4 and verify improvements
