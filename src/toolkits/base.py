"""
Base toolkit class for LangChain framework.

This module provides the base interface for all toolkits.
"""

from abc import ABC, abstractmethod
from typing import List
from pydantic import BaseModel


class BaseToolkit(BaseModel, ABC):
    """Base class for all toolkits.
    
    Toolkits are collections of related tools that work together
    to accomplish a specific set of tasks.
    """
    
    class Config:
        arbitrary_types_allowed = True
    
    @abstractmethod
    def get_tools(self) -> List:
        """Get all tools in the toolkit.
        
        Returns:
            List of tools in the toolkit
        """
        pass
    
    def get_tool_names(self) -> List[str]:
        """Get names of all tools in the toolkit.
        
        Returns:
            List of tool names
        """
        return [tool.name for tool in self.get_tools()]
    
    def get_tool_descriptions(self) -> List[str]:
        """Get descriptions of all tools in the toolkit.
        
        Returns:
            List of tool descriptions
        """
        return [f"{tool.name}: {tool.description}" for tool in self.get_tools()]

# Made with Bob
