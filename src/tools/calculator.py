"""
Calculator tool for performing mathematical operations.
"""

import re
from typing import Any
from .base import BaseTool


class CalculatorTool(BaseTool):
    """Tool for performing mathematical calculations.
    
    This tool can evaluate mathematical expressions safely.
    """
    
    name: str = "calculator"
    description: str = (
        "Useful for performing mathematical calculations. "
        "Input should be a valid mathematical expression. "
        "Example: '2 + 2' or '10 * 5 + 3'"
    )
    
    def _run(self, expression: str) -> Any:
        """Evaluate a mathematical expression.
        
        Args:
            expression: A string containing a mathematical expression
            
        Returns:
            The result of the calculation
        """
        try:
            # Remove any potentially dangerous characters
            expression = expression.strip()
            
            # Only allow numbers, operators, parentheses, and whitespace
            if not re.match(r'^[\d\s\+\-\*\/\(\)\.\^]+$', expression):
                return "Error: Invalid characters in expression. Only numbers and basic operators (+, -, *, /, ^, parentheses) are allowed."
            
            # Replace ^ with ** for exponentiation
            expression = expression.replace('^', '**')
            
            # Evaluate the expression safely
            result = eval(expression, {"__builtins__": {}}, {})
            
            return f"Result: {result}"
            
        except ZeroDivisionError:
            return "Error: Division by zero"
        except SyntaxError:
            return "Error: Invalid mathematical expression"
        except Exception as e:
            return f"Error: {str(e)}"
    
    @staticmethod
    def create() -> "CalculatorTool":
        """Factory method to create a calculator tool.
        
        Returns:
            A new CalculatorTool instance
        """
        return CalculatorTool()

# Made with Bob
