"""Reference answers: a question's SQL is run on the database; no person or model fills them in."""
import json
from pathlib import Path

from text2sql.agents.planned_agent import tools

from .complex import compare_rows
from .templates import compare_answers


def load_seeds(paths) -> list[dict]:
    seeds = []
    for path in paths:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                seeds.append(json.loads(line))
    ids = [seed["seed_id"] for seed in seeds]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate seed_id")
    return seeds


def compute_oracle(seed: dict, db_path=tools.DB_PATH) -> dict:
    """Golden answer from the seed SQL through the same read-only boundary the app uses."""
    result = tools.read_sales(seed["sql"], db_path)
    if not result["rows"]:
        raise ValueError(f"{seed['seed_id']}: seed SQL returned no rows")
    oracle = {"columns": result["columns"], "rows": result["rows"]}
    plot = dict(seed["expected_plot"]) if seed.get("expected_plot") is not None else None
    sql_x = None
    if plot and plot.get("should_plot"):
        index = result["columns"].index(plot.pop("x_from_column"))
        sql_x = plot["expected_x_order"] = [str(row[index]) for row in result["rows"]]
    oracle["expected_plot"] = plot
    if "reference_answer" in seed:  # second, independent oracle (legacy pandas on the source CSV)
        problems = compare_answers(seed["eval_type"], result["columns"], result["rows"], seed["reference_answer"])
        if problems:
            raise ValueError(f"{seed['seed_id']}: SQL and pandas disagree: {'; '.join(problems)}")
    if "reference_rows" in seed:  # complex seeds: independent pandas implementation on the CSV
        problems = compare_rows(result["rows"], seed["reference_rows"])
        if problems:
            raise ValueError(f"{seed['seed_id']}: SQL and pandas disagree: {'; '.join(problems[:3])}")
    oracle["oracle"] = seed.get("oracle", "sql")
    return oracle
