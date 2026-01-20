"""
Base tool class for LangChain framework.

This module provides the base interface for all tools.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Dict, Callable
from pydantic import BaseModel, Field


class BaseTool(BaseModel, ABC):
    """Base class for all tools.
    
    Tools are utilities that can be called by LLMs to perform specific tasks.
    Each tool has a name, description, and a run method that executes the tool's logic.
    """
    
    name: str = Field(description="The name of the tool")
    description: str = Field(description="A description of what the tool does")
    return_direct: bool = Field(
        default=False,
        description="Whether to return the tool's output directly to the user"
    )
    
    class Config:
        arbitrary_types_allowed = True
    
    @abstractmethod
    def _run(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the tool's main logic.
        
        This method must be implemented by all tool subclasses.
        
        Args:
            *args: Positional arguments for the tool
            **kwargs: Keyword arguments for the tool
            
        Returns:
            The result of the tool execution
        """
        pass
    
    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Public method to run the tool.
        
        This method wraps _run and can add additional logic like
        error handling, logging, etc.
        
        Args:
            *args: Positional arguments for the tool
            **kwargs: Keyword arguments for the tool
            
        Returns:
            The result of the tool execution
        """
        try:
            return self._run(*args, **kwargs)
        except Exception as e:
            return f"Error executing tool {self.name}: {str(e)}"
    
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Allow the tool to be called like a function.
        
        Args:
            *args: Positional arguments for the tool
            **kwargs: Keyword arguments for the tool
            
        Returns:
            The result of the tool execution
        """
        return self.run(*args, **kwargs)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the tool to a dictionary representation.
        
        Returns:
            Dictionary with tool name and description
        """
        return {
            "name": self.name,
            "description": self.description,
            "return_direct": self.return_direct,
        }
    
    @classmethod
    def from_function(
        cls,
        func: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None,
        return_direct: bool = False,
    ) -> "BaseTool":
        """Create a tool from a function.
        
        Args:
            func: The function to wrap as a tool
            name: Optional name for the tool (defaults to function name)
            description: Optional description (defaults to function docstring)
            return_direct: Whether to return output directly
            
        Returns:
            A FunctionTool instance wrapping the function
        """
        from .function_tool import FunctionTool
        
        return FunctionTool(
            func=func,
            name=name or func.__name__,
            description=description or func.__doc__ or "No description provided",
            return_direct=return_direct,
        )

# Made with Bob
