"""The root export is lazy so business tools do not initialize an agent."""

def __getattr__(name):
    if name == "root_agent":
        from .agent import root_agent
        return root_agent
    raise AttributeError(name)
