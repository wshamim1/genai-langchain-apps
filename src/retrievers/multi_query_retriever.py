"""
Multi-Query Retriever
Generates multiple queries from a single input and retrieves documents for each.
"""

from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain.retrievers.multi_query import MultiQueryRetriever as LangChainMultiQueryRetriever

from .base import BaseRetriever


class MultiQueryRetriever(BaseRetriever):
    """
    Multi-query retriever implementation.
    Generates multiple perspectives of a query and retrieves documents for each.
    """
    
    def __init__(
        self,
        retriever: Any,
        llm: BaseLanguageModel,
        include_original: bool = True,
        **kwargs
    ):
        """
        Initialize multi-query retriever.
        
        Args:
            retriever: Base retriever to use
            llm: Language model for query generation
            include_original: Whether to include the original query
            **kwargs: Additional arguments
        """
        super().__init__(**kwargs)
        self.base_retriever = retriever
        self.llm = llm
        self.include_original = include_original
        
        # Create the multi-query retriever
        self.retriever = LangChainMultiQueryRetriever.from_llm(
            retriever=retriever,
            llm=llm,
            include_original=include_original
        )
    
    def get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        """
        Retrieve relevant documents using multiple query perspectives.
        
        Args:
            query: The original query string
            **kwargs: Additional arguments
            
        Returns:
            List of relevant documents (deduplicated)
        """
        return self.retriever.invoke(query)
    
    def get_generated_queries(self, query: str) -> List[str]:
        """
        Get the generated query variations.
        
        Args:
            query: The original query
            
        Returns:
            List of generated queries
        """
        # This would require accessing internal methods
        # For now, we'll return a placeholder
        return [query]  # In practice, this would show all generated queries
    
    @classmethod
    def from_retriever(
        cls,
        retriever: Any,
        llm: BaseLanguageModel,
        include_original: bool = True,
        **kwargs
    ) -> "MultiQueryRetriever":
        """
        Create a multi-query retriever from a base retriever.
        
        Args:
            retriever: Base retriever
            llm: Language model for query generation
            include_original: Whether to include original query
            **kwargs: Additional arguments
            
        Returns:
            MultiQueryRetriever instance
        """
        return cls(
            retriever=retriever,
            llm=llm,
            include_original=include_original,
            **kwargs
        )

# Made with Bob
