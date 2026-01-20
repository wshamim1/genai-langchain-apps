"""
Function tool implementation for wrapping Python functions as tools.
"""

from typing import Any, Callable
from .base import BaseTool


class FunctionTool(BaseTool):
    """Tool that wraps a Python function.
    
    This allows any Python function to be used as a tool that can be
    called by LLMs or agents.
    """
    
    func: Callable
    
    class Config:
        arbitrary_types_allowed = True
    
    def _run(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the wrapped function.
        
        Args:
            *args: Positional arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            The result of the function execution
        """
        return self.func(*args, **kwargs)

# Made with Bob
