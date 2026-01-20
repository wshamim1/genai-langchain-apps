"""
Vector Store Retriever
Retrieves documents from a vector store using similarity search.
"""

from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from .base import BaseRetriever


class VectorStoreRetriever(BaseRetriever):
    """
    Vector store retriever implementation.
    Retrieves documents from a vector store using various search strategies.
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        search_type: str = "similarity",
        search_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Initialize vector store retriever.
        
        Args:
            vector_store: The vector store to retrieve from
            search_type: Type of search ("similarity", "mmr", "similarity_score_threshold")
            search_kwargs: Additional search parameters (k, score_threshold, etc.)
            **kwargs: Additional arguments
        """
        super().__init__(**kwargs)
        self.vector_store = vector_store
        self.search_type = search_type
        self.search_kwargs = search_kwargs or {"k": 4}
        
        # Create the retriever from vector store
        self.retriever = vector_store.as_retriever(
            search_type=search_type,
            search_kwargs=self.search_kwargs
        )
    
    def get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: The query string
            **kwargs: Additional arguments
            
        Returns:
            List of relevant documents
        """
        return self.retriever.invoke(query)
    
    def get_relevant_documents_with_scores(
        self,
        query: str,
        k: Optional[int] = None
    ) -> List[tuple[Document, float]]:
        """
        Retrieve relevant documents with similarity scores.
        
        Args:
            query: The query string
            k: Number of documents to retrieve
            
        Returns:
            List of (document, score) tuples
        """
        k = k or self.search_kwargs.get("k", 4)
        return self.vector_store.similarity_search_with_score(query, k=k)
    
    def update_search_kwargs(self, **kwargs):
        """
        Update search parameters.
        
        Args:
            **kwargs: New search parameters
        """
        self.search_kwargs.update(kwargs)
        self.retriever = self.vector_store.as_retriever(
            search_type=self.search_type,
            search_kwargs=self.search_kwargs
        )
    
    def set_k(self, k: int):
        """
        Set the number of documents to retrieve.
        
        Args:
            k: Number of documents
        """
        self.update_search_kwargs(k=k)
    
    def set_score_threshold(self, threshold: float):
        """
        Set the minimum similarity score threshold.
        
        Args:
            threshold: Minimum score (0-1)
        """
        self.search_type = "similarity_score_threshold"
        self.update_search_kwargs(score_threshold=threshold)
    
    def enable_mmr(self, fetch_k: int = 20, lambda_mult: float = 0.5):
        """
        Enable Maximal Marginal Relevance search for diversity.
        
        Args:
            fetch_k: Number of documents to fetch for MMR
            lambda_mult: Diversity parameter (0=max diversity, 1=min diversity)
        """
        self.search_type = "mmr"
        self.update_search_kwargs(fetch_k=fetch_k, lambda_mult=lambda_mult)
    
    @classmethod
    def from_vector_store(
        cls,
        vector_store: VectorStore,
        k: int = 4,
        search_type: str = "similarity",
        **kwargs
    ) -> "VectorStoreRetriever":
        """
        Create a retriever from a vector store.
        
        Args:
            vector_store: The vector store
            k: Number of documents to retrieve
            search_type: Type of search
            **kwargs: Additional arguments
            
        Returns:
            VectorStoreRetriever instance
        """
        return cls(
            vector_store=vector_store,
            search_type=search_type,
            search_kwargs={"k": k, **kwargs}
        )

# Made with Bob
