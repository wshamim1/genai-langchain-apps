"""
API request tool for making HTTP requests.
"""

import json
from typing import Any, Optional, Dict
from .base import BaseTool


class APIRequestTool(BaseTool):
    """Tool for making HTTP API requests.
    
    This tool can make GET and POST requests to APIs.
    """
    
    name: str = "api_request"
    description: str = (
        "Make HTTP API requests. "
        "Input should be in format: 'METHOD|URL|optional_json_body'. "
        "Example: 'GET|https://api.example.com/data' or "
        "'POST|https://api.example.com/data|{\"key\": \"value\"}'"
    )
    headers: Optional[Dict[str, str]] = None
    
    def _run(self, input_str: str) -> Any:
        """Make an HTTP API request.
        
        Args:
            input_str: String in format "METHOD|URL|optional_json_body"
            
        Returns:
            API response or error message
        """
        try:
            import requests
        except ImportError:
            return (
                "API requests require the 'requests' package. "
                "Install it with: pip install requests"
            )
        
        try:
            # Parse input
            parts = input_str.split('|')
            
            if len(parts) < 2:
                return "Error: Input must be in format 'METHOD|URL|optional_json_body'"
            
            method = parts[0].strip().upper()
            url = parts[1].strip()
            body = parts[2] if len(parts) > 2 else None
            
            # Validate method
            if method not in ['GET', 'POST', 'PUT', 'DELETE', 'PATCH']:
                return f"Error: Unsupported HTTP method: {method}"
            
            # Prepare headers
            headers = self.headers or {}
            if body and 'Content-Type' not in headers:
                headers['Content-Type'] = 'application/json'
            
            # Make request
            if method == 'GET':
                response = requests.get(url, headers=headers, timeout=10)
            elif method == 'POST':
                data = json.loads(body) if body else None
                response = requests.post(url, json=data, headers=headers, timeout=10)
            elif method == 'PUT':
                data = json.loads(body) if body else None
                response = requests.put(url, json=data, headers=headers, timeout=10)
            elif method == 'DELETE':
                response = requests.delete(url, headers=headers, timeout=10)
            elif method == 'PATCH':
                data = json.loads(body) if body else None
                response = requests.patch(url, json=data, headers=headers, timeout=10)
            
            # Format response
            result = f"Status Code: {response.status_code}\n\n"
            
            try:
                json_response = response.json()
                result += f"Response:\n{json.dumps(json_response, indent=2)}"
            except json.JSONDecodeError:
                result += f"Response:\n{response.text[:500]}"
            
            return result
            
        except requests.exceptions.Timeout:
            return "Error: Request timed out"
        except requests.exceptions.ConnectionError:
            return "Error: Could not connect to the server"
        except json.JSONDecodeError:
            return "Error: Invalid JSON in request body"
        except Exception as e:
            return f"Error making API request: {str(e)}"
    
    @staticmethod
    def create(headers: Optional[Dict[str, str]] = None) -> "APIRequestTool":
        """Factory method to create an API request tool.
        
        Args:
            headers: Optional default headers for requests
            
        Returns:
            A new APIRequestTool instance
        """
        return APIRequestTool(headers=headers)

# Made with Bob
