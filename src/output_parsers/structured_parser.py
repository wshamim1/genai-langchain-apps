"""
Structured Output Parser
Parses LLM output into structured data with defined schemas.
"""

from typing import Any, Dict, List
import json
import re
from pydantic import BaseModel, Field

from .base import BaseOutputParser


class ResponseSchema(BaseModel):
    """Schema for a single response field."""
    name: str = Field(description="Name of the field")
    description: str = Field(description="Description of the field")
    type: str = Field(default="string", description="Type of the field")


class StructuredOutputParser(BaseOutputParser):
    """
    Structured output parser implementation.
    Parses LLM output into structured data based on response schemas.
    """
    
    def __init__(self, response_schemas: List[ResponseSchema]):
        """
        Initialize structured output parser.
        
        Args:
            response_schemas: List of ResponseSchema objects defining the structure
        """
        super().__init__()
        self.response_schemas = response_schemas
    
    def parse(self, text: str) -> Dict[str, Any]:
        """
        Parse the output text into structured data.
        
        Args:
            text: The text to parse
            
        Returns:
            Dictionary with structured data
        """
        # Try to parse as JSON first
        try:
            # Look for JSON in the text
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                return json.loads(json_str)
        except json.JSONDecodeError:
            pass
        
        # Fallback: parse key-value pairs
        result = {}
        for schema in self.response_schemas:
            # Look for the field in the text
            pattern = rf"{schema.name}:\s*(.+?)(?:\n|$)"
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                result[schema.name] = match.group(1).strip()
            else:
                result[schema.name] = ""
        
        return result
    
    def get_format_instructions(self) -> str:
        """
        Get instructions for formatting the output according to the schema.
        
        Returns:
            Format instructions string
        """
        schema_str = "```json\n{\n"
        for schema in self.response_schemas:
            schema_str += f'  "{schema.name}": "{schema.description}",\n'
        schema_str = schema_str.rstrip(",\n") + "\n}\n```"
        
        instructions = f"""Please provide your response in the following JSON format:

{schema_str}

Make sure to include all fields and provide appropriate values based on the descriptions."""
        
        return instructions
    
    def get_schema_info(self) -> List[Dict[str, str]]:
        """
        Get information about the response schemas.
        
        Returns:
            List of schema information dictionaries
        """
        return [
            {
                "name": schema.name,
                "description": schema.description,
                "type": schema.type if hasattr(schema, 'type') else "string"
            }
            for schema in self.response_schemas
        ]
    
    @classmethod
    def from_schema_dict(cls, schema_dict: Dict[str, str]) -> "StructuredOutputParser":
        """
        Create a structured output parser from a dictionary of field names and descriptions.
        
        Args:
            schema_dict: Dictionary mapping field names to descriptions
            
        Returns:
            StructuredOutputParser instance
        """
        response_schemas = [
            ResponseSchema(name=name, description=description)
            for name, description in schema_dict.items()
        ]
        return cls(response_schemas)

# Made with Bob
