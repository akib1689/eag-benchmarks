from abc import ABC, abstractmethod
from typing import Any, Dict


class ToolABC(ABC):
    """Base class for all agent tools.

    Tools provide agents with capabilities to interact with their environment.
    Each tool has a name, description (for LLM prompt inclusion), and a run method.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier used in action dispatch, e.g. 'Execute'."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description for LLM prompt inclusion."""
        ...

    @abstractmethod
    def run(self, params: Dict[str, Any]) -> str:
        """Execute the tool with given parameters and return an observation string.

        Args:
            params: Tool-specific parameters as a dictionary.

        Returns:
            Observation string that gets fed back to the agent.
        """
        ...

    def format_for_prompt(self) -> str:
        """Return a formatted description for inclusion in agent system prompts."""
        return f"- {self.name}: {self.description}"


TOOL_REGISTRY: Dict[str, type] = {}


def register_tool(cls: type) -> type:
    """Decorator to register a tool class in the global registry."""
    TOOL_REGISTRY[cls.__name__] = cls
    return cls


def get_tool(name: str) -> type:
    """Look up a tool class by name."""
    if name not in TOOL_REGISTRY:
        raise KeyError(f"Tool '{name}' not found. Available: {list(TOOL_REGISTRY.keys())}")
    return TOOL_REGISTRY[name]


def all_tools_prompt() -> str:
    """Generate a prompt section listing all registered tools and their descriptions."""
    lines = ["Available tools:"]
    for cls in TOOL_REGISTRY.values():
        instance = cls.__new__(cls)
        lines.append(instance.format_for_prompt())
    return "\n".join(lines)
