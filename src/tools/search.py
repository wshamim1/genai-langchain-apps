"""
Search tool for performing web searches.
"""

from typing import Any, Optional
from .base import BaseTool


class SearchTool(BaseTool):
    """Tool for performing web searches.
    
    This is a mock implementation. In production, you would integrate
    with a real search API like Google, Bing, or DuckDuckGo.
    """
    
    name: str = "search"
    description: str = (
        "Useful for searching the internet for current information. "
        "Input should be a search query string. "
        "Example: 'What is the weather in New York?'"
    )
    api_key: Optional[str] = None
    
    def _run(self, query: str) -> Any:
        """Perform a web search.
        
        Args:
            query: The search query string
            
        Returns:
            Search results (mock implementation)
        """
        # This is a mock implementation
        # In production, you would call a real search API
        
        mock_results = {
            "weather": "The weather is sunny with a high of 75°F.",
            "python": "Python is a high-level programming language known for its simplicity and readability.",
            "langchain": "LangChain is a framework for developing applications powered by language models.",
            "ai": "Artificial Intelligence (AI) refers to the simulation of human intelligence in machines.",
        }
        
        query_lower = query.lower()
        
        # Find matching results
        for key, value in mock_results.items():
            if key in query_lower:
                return f"Search results for '{query}':\n{value}"
        
        return f"Search results for '{query}':\nNo specific results found. This is a mock search tool. In production, integrate with a real search API."
    
    @staticmethod
    def create(api_key: Optional[str] = None) -> "SearchTool":
        """Factory method to create a search tool.
        
        Args:
            api_key: Optional API key for the search service
            
        Returns:
            A new SearchTool instance
        """
        return SearchTool(api_key=api_key)


class DuckDuckGoSearchTool(BaseTool):
    """Tool for searching using DuckDuckGo.
    
    This requires the duckduckgo-search package to be installed.
    """
    
    name: str = "duckduckgo_search"
    description: str = (
        "Search the web using DuckDuckGo. "
        "Input should be a search query. "
        "Returns a list of search results with titles and snippets."
    )
    max_results: int = 5
    
    def _run(self, query: str) -> Any:
        """Perform a DuckDuckGo search.
        
        Args:
            query: The search query string
            
        Returns:
            Search results from DuckDuckGo
        """
        try:
            from duckduckgo_search import DDGS
            
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=self.max_results))
                
                if not results:
                    return f"No results found for query: {query}"
                
                formatted_results = []
                for i, result in enumerate(results, 1):
                    formatted_results.append(
                        f"{i}. {result.get('title', 'No title')}\n"
                        f"   {result.get('body', 'No description')}\n"
                        f"   URL: {result.get('href', 'No URL')}"
                    )
                
                return "\n\n".join(formatted_results)
                
        except ImportError:
            return (
                "DuckDuckGo search requires the 'duckduckgo-search' package. "
                "Install it with: pip install duckduckgo-search"
            )
        except Exception as e:
            return f"Error performing search: {str(e)}"
    
    @staticmethod
    def create(max_results: int = 5) -> "DuckDuckGoSearchTool":
        """Factory method to create a DuckDuckGo search tool.
        
        Args:
            max_results: Maximum number of results to return
            
        Returns:
            A new DuckDuckGoSearchTool instance
        """
        return DuckDuckGoSearchTool(max_results=max_results)

# Made with Bob
