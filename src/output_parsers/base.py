"""
Base Output Parser
Provides a base class for output parser implementations.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BaseOutputParser(ABC):
    """
    Base class for output parser implementations.
    Wraps LangChain output parsers with additional functionality.
    """
    
    def __init__(self):
        """Initialize the output parser."""
        self.parser = None
    
    @abstractmethod
    def parse(self, text: str) -> Any:
        """
        Parse the output text into structured data.
        
        Args:
            text: The text to parse
            
        Returns:
            Parsed structured data
        """
        pass
    
    @abstractmethod
    def get_format_instructions(self) -> str:
        """
        Get instructions for formatting the output.
        
        Returns:
            Format instructions string
        """
        pass
    
    def parse_with_prompt(self, completion: str, prompt: Optional[str] = None) -> Any:
        """
        Parse the completion with optional prompt context.
        
        Args:
            completion: The completion text to parse
            prompt: Optional prompt for context
            
        Returns:
            Parsed structured data
        """
        return self.parse(completion)
    
    @property
    def _type(self) -> str:
        """Return the type of the output parser."""
        return self.__class__.__name__

# Made with Bob
