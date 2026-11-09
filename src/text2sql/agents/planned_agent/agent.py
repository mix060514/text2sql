"""Analysis backend entry point: the fixed planner → execute → review → report workflow.

TEXT2SQL_BACKEND accepts only "qwen27b"; config rejects anything else.
"""

from .qwen_workflow import root_agent, model  # noqa: F401  (model used by live scripts)

__all__ = ["root_agent"]
