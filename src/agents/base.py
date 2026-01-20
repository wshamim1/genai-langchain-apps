"""
Base agent class for LangChain framework.

This module provides the base interface for all agents.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class AgentAction(BaseModel):
    """Represents an action taken by an agent.
    
    An action consists of a tool to use and the input to that tool.
    """
    
    tool: str = Field(description="The name of the tool to use")
    tool_input: str = Field(description="The input to pass to the tool")
    log: str = Field(default="", description="Log of the agent's reasoning")
    
    class Config:
        arbitrary_types_allowed = True


class AgentFinish(BaseModel):
    """Represents the final output of an agent.
    
    This is returned when the agent has completed its task.
    """
    
    return_values: Dict[str, Any] = Field(description="The final output values")
    log: str = Field(default="", description="Log of the agent's final reasoning")
    
    class Config:
        arbitrary_types_allowed = True


class BaseAgent(BaseModel, ABC):
    """Base class for all agents.
    
    Agents are systems that use LLMs to decide which tools to use
    and in what order to accomplish a task.
    """
    
    llm: Any = Field(description="The language model to use for decision making")
    tools: List[Any] = Field(default_factory=list, description="List of tools available to the agent")
    max_iterations: int = Field(default=10, description="Maximum number of iterations")
    verbose: bool = Field(default=False, description="Whether to print verbose output")
    
    class Config:
        arbitrary_types_allowed = True
    
    @abstractmethod
    def plan(
        self,
        intermediate_steps: List[tuple],
        **kwargs: Any
    ) -> AgentAction | AgentFinish:
        """Decide what action to take next.
        
        Args:
            intermediate_steps: List of (AgentAction, observation) tuples
            **kwargs: Additional arguments
            
        Returns:
            Either an AgentAction to take or AgentFinish if done
        """
        pass
    
    def get_tool_names(self) -> List[str]:
        """Get names of all available tools.
        
        Returns:
            List of tool names
        """
        return [tool.name for tool in self.tools]
    
    def get_tool_by_name(self, name: str) -> Optional[Any]:
        """Get a tool by its name.
        
        Args:
            name: The name of the tool
            
        Returns:
            The tool if found, None otherwise
        """
        for tool in self.tools:
            if tool.name == name:
                return tool
        return None
    
    def format_tools(self) -> str:
        """Format the list of tools for the prompt.
        
        Returns:
            Formatted string describing all tools
        """
        tool_strings = []
        for tool in self.tools:
            tool_strings.append(f"{tool.name}: {tool.description}")
        return "\n".join(tool_strings)

# Made with Bob
