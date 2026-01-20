"""
Base Vector Store
Provides a base class for vector store implementations.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Tuple
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


class BaseVectorStore(ABC):
    """
    Base class for vector store implementations.
    Wraps LangChain vector stores with additional functionality.
    """
    
    def __init__(self, embedding_function: Embeddings, **kwargs):
        """
        Initialize the vector store.
        
        Args:
            embedding_function: Embeddings model to use
            **kwargs: Additional arguments for the vector store
        """
        self.embedding_function = embedding_function
        self.store = None
    
    @abstractmethod
    def add_documents(self, documents: List[Document], **kwargs) -> List[str]:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of documents to add
            **kwargs: Additional arguments
            
        Returns:
            List of document IDs
        """
        pass
    
    @abstractmethod
    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> List[str]:
        """
        Add texts to the vector store.
        
        Args:
            texts: List of texts to add
            metadatas: Optional list of metadata dicts
            **kwargs: Additional arguments
            
        Returns:
            List of document IDs
        """
        pass
    
    @abstractmethod
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs
    ) -> List[Document]:
        """
        Search for similar documents.
        
        Args:
            query: Query text
            k: Number of results to return
            **kwargs: Additional arguments
            
        Returns:
            List of similar documents
        """
        pass
    
    @abstractmethod
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        **kwargs
    ) -> List[Tuple[Document, float]]:
        """
        Search for similar documents with similarity scores.
        
        Args:
            query: Query text
            k: Number of results to return
            **kwargs: Additional arguments
            
        Returns:
            List of (document, score) tuples
        """
        pass
    
    @abstractmethod
    def delete(self, ids: Optional[List[str]] = None, **kwargs) -> Optional[bool]:
        """
        Delete documents from the vector store.
        
        Args:
            ids: List of document IDs to delete
            **kwargs: Additional arguments
            
        Returns:
            Success status
        """
        pass
    
    def as_retriever(self, **kwargs):
        """
        Return the vector store as a retriever.
        
        Args:
            **kwargs: Arguments for the retriever
            
        Returns:
            Retriever instance
        """
        if self.store is None:
            raise ValueError("Vector store not initialized")
        return self.store.as_retriever(**kwargs)
    
    @abstractmethod
    def persist(self) -> None:
        """Persist the vector store to disk."""
        pass
    
    @abstractmethod
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the collection.
        
        Returns:
            Dictionary with collection information
        """
        pass

# Made with Bob
