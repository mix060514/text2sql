"""Select the analysis backend; both share tools, contracts, and publication rules.

TEXT2SQL_BACKEND=qwen27b (default): fixed planner → execute → review → report workflow.
TEXT2SQL_BACKEND=codex: one Codex CLI invocation with the read-only sales MCP tool.
"""

from text2sql.config import BACKEND

if BACKEND == "codex":
    from .codex_workflow import root_agent
else:
    from .qwen_workflow import root_agent, model  # noqa: F401  (model used by live scripts)

__all__ = ["root_agent"]
