"""AgentServe: an LLM serving gateway that schedules agent sessions."""
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

try:
    __version__ = _version("agentserve")
except PackageNotFoundError:
    # Running from a source tree that was never installed.
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
