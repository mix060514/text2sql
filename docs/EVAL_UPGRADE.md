# Eval Upgrade Changelog

## Feature: Eval Mechanism Upgrade (2026-02-08)

This branch (`feature/eval-upgrade`) implements comprehensive improvements to the evaluation system for Plotly charts and ground truth validation.

### Changes Overview

#### Phase 1: Chart Data Storage (選項 C)
**File:** `src/text2sql/agents/planned_agent/agent.py`

- Added `chart_data` to session state when chart is rendered
- Contains: `chart_type`, `x_column`, `y_column`, `color_column`, `x_values`, `y_values`, `data_rows`
- Enables direct data comparison without parsing images

#### Phase 2: Extended Eval Set (選項 A)
**Files:**
- `data/eval_set_v3.jsonl` - New eval set with `expected_plot` field
- `scripts/add_expected_plot.py` - Script to generate v3 from v2

**New `expected_plot` schema:**
```json
{
  "should_plot": true,
  "chart_type": "line",
  "x_column": "Period",
  "y_column": "Revenue",
  "color_column": null,
  "expected_x_order": ["2023-01", "2023-02", ...]
}
```

**Coverage:** 21 cases with expected plot annotations
- Monthly trend data (17 cases)
- Year comparison data (3 cases)
- Regional preference data (1 case)

#### Phase 3: End-to-End Evaluation (選項 B)
**File:** `eval/eval_e2e.py`

Validates the complete workflow:
1. ✅ SQL generation and execution correctness (via Judge Agent)
2. ✅ Plot decision accuracy (should_plot, chart_type)
3. ✅ Chart data correctness (x_values order validation)
4. ✅ Chart image generation confirmation

**Output Metrics:**
- `sql_accuracy` - SQL query correctness rate
- `plot_decision_accuracy` - Plot decision match rate
- `chart_generation_rate` - Chart image generation rate
- `x_order_accuracy` - X-axis data order correctness
- `e2e_success_rate` - Overall end-to-end success rate

### Usage

```bash
# Run full end-to-end evaluation
uv run python eval/eval_e2e.py

# Run with limited cases (for testing)
uv run python eval/eval_e2e.py --limit 5

# Generate expected_plot annotations (if needed)
uv run python scripts/add_expected_plot.py
```

### Related Files
- `eval/eval.py` - Original SQL-only evaluation
- `eval/eval_plot.py` - Plot decision-only evaluation
- `data/eval_set_v2.jsonl` - Original eval set (without expected_plot)
