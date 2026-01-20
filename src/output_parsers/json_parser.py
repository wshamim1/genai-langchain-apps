"""
JSON Output Parser
Parses LLM output into JSON objects.
"""

import json
from typing import Any, Dict
from langchain_core.output_parsers import JsonOutputParser as LangChainJsonParser

from .base import BaseOutputParser


class JSONOutputParser(BaseOutputParser):
    """
    JSON output parser implementation.
    Parses LLM output into JSON/dictionary objects.
    """
    
    def __init__(self):
        """Initialize JSON output parser."""
        super().__init__()
        self.parser = LangChainJsonParser()
    
    def parse(self, text: str) -> Dict[str, Any]:
        """
        Parse the output text into a JSON object.
        
        Args:
            text: The text to parse
            
        Returns:
            Dictionary representing the JSON object
        """
        return self.parser.parse(text)
    
    def get_format_instructions(self) -> str:
        """
        Get instructions for formatting the output as JSON.
        
        Returns:
            Format instructions string
        """
        return "Return a valid JSON object."
    
    def parse_json_string(self, text: str) -> Dict[str, Any]:
        """
        Parse a JSON string directly.
        
        Args:
            text: JSON string to parse
            
        Returns:
            Dictionary representing the JSON object
        """
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}")
    
    def validate_json(self, text: str) -> bool:
        """
        Validate if the text is valid JSON.
        
        Args:
            text: Text to validate
            
        Returns:
            True if valid JSON, False otherwise
        """
        try:
            json.loads(text)
            return True
        except json.JSONDecodeError:
            return False

# Made with Bob
