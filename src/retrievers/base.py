"""
Base Retriever
Provides a base class for retriever implementations.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document


class BaseRetriever(ABC):
    """
    Base class for retriever implementations.
    Wraps LangChain retrievers with additional functionality.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the retriever.
        
        Args:
            **kwargs: Additional arguments for the retriever
        """
        self.retriever = None
        self.config = kwargs
    
    @abstractmethod
    def get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: The query string
            **kwargs: Additional arguments
            
        Returns:
            List of relevant documents
        """
        pass
    
    def invoke(self, query: str, **kwargs) -> List[Document]:
        """
        Invoke the retriever with a query.
        
        Args:
            query: The query string
            **kwargs: Additional arguments
            
        Returns:
            List of relevant documents
        """
        return self.get_relevant_documents(query, **kwargs)
    
    def batch(self, queries: List[str], **kwargs) -> List[List[Document]]:
        """
        Retrieve documents for multiple queries.
        
        Args:
            queries: List of query strings
            **kwargs: Additional arguments
            
        Returns:
            List of document lists, one for each query
        """
        return [self.get_relevant_documents(query, **kwargs) for query in queries]
    
    @property
    def _type(self) -> str:
        """Return the type of the retriever."""
        return self.__class__.__name__
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the retriever configuration.
        
        Returns:
            Configuration dictionary
        """
        return self.config.copy()

# Made with Bob
