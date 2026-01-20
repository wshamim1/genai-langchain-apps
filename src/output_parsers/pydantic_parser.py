"""
Pydantic Output Parser
Parses LLM output into Pydantic models for type-safe structured data.
"""

from typing import Type, TypeVar
from pydantic import BaseModel
from langchain_core.output_parsers import PydanticOutputParser as LangChainPydanticParser

from .base import BaseOutputParser

T = TypeVar('T', bound=BaseModel)


class PydanticOutputParser(BaseOutputParser):
    """
    Pydantic output parser implementation.
    Parses LLM output into Pydantic models with automatic validation.
    """
    
    def __init__(self, pydantic_object: Type[T]):
        """
        Initialize Pydantic output parser.
        
        Args:
            pydantic_object: The Pydantic model class to parse into
        """
        super().__init__()
        self.pydantic_object = pydantic_object
        self.parser = LangChainPydanticParser(pydantic_object=pydantic_object)
    
    def parse(self, text: str) -> T:
        """
        Parse the output text into a Pydantic model.
        
        Args:
            text: The text to parse
            
        Returns:
            Pydantic model instance
        """
        return self.parser.parse(text)
    
    def get_format_instructions(self) -> str:
        """
        Get instructions for formatting the output as JSON matching the Pydantic schema.
        
        Returns:
            Format instructions string
        """
        return self.parser.get_format_instructions()
    
    def get_schema(self) -> dict:
        """
        Get the JSON schema of the Pydantic model.
        
        Returns:
            JSON schema dictionary
        """
        return self.pydantic_object.model_json_schema()
    
    def parse_with_prompt(self, completion: str, prompt: str = None) -> T:
        """
        Parse the completion with optional prompt context.
        
        Args:
            completion: The completion text to parse
            prompt: Optional prompt for context
            
        Returns:
            Pydantic model instance
        """
        if hasattr(self.parser, 'parse_with_prompt'):
            return self.parser.parse_with_prompt(completion, prompt)
        return self.parse(completion)

# Made with Bob
