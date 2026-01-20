"""
Chroma Vector Store
Wrapper around LangChain's Chroma vector store.
"""

from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma

from .base import BaseVectorStore


class ChromaVectorStore(BaseVectorStore):
    """
    Chroma vector store implementation.
    Provides persistent vector storage with similarity search.
    """
    
    def __init__(
        self,
        embedding_function: Embeddings,
        collection_name: str = "langchain",
        persist_directory: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize Chroma vector store.
        
        Args:
            embedding_function: Embeddings model to use
            collection_name: Name of the collection
            persist_directory: Directory to persist the database
            **kwargs: Additional arguments for Chroma
        """
        super().__init__(embedding_function, **kwargs)
        
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        
        # Initialize Chroma
        if persist_directory:
            Path(persist_directory).mkdir(parents=True, exist_ok=True)
            
        self.store = Chroma(
            collection_name=collection_name,
            embedding_function=embedding_function,
            persist_directory=persist_directory,
            **kwargs
        )
    
    @classmethod
    def from_documents(
        cls,
        documents: List[Document],
        embedding_function: Embeddings,
        collection_name: str = "langchain",
        persist_directory: Optional[str] = None,
        **kwargs
    ) -> "ChromaVectorStore":
        """
        Create a Chroma vector store from documents.
        
        Args:
            documents: List of documents to add
            embedding_function: Embeddings model to use
            collection_name: Name of the collection
            persist_directory: Directory to persist the database
            **kwargs: Additional arguments
            
        Returns:
            ChromaVectorStore instance
        """
        if persist_directory:
            Path(persist_directory).mkdir(parents=True, exist_ok=True)
            
        store = Chroma.from_documents(
            documents=documents,
            embedding=embedding_function,
            collection_name=collection_name,
            persist_directory=persist_directory,
            **kwargs
        )
        
        instance = cls(
            embedding_function=embedding_function,
            collection_name=collection_name,
            persist_directory=persist_directory,
            **kwargs
        )
        instance.store = store
        return instance
    
    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding_function: Embeddings,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        collection_name: str = "langchain",
        persist_directory: Optional[str] = None,
        **kwargs
    ) -> "ChromaVectorStore":
        """
        Create a Chroma vector store from texts.
        
        Args:
            texts: List of texts to add
            embedding_function: Embeddings model to use
            metadatas: Optional list of metadata dicts
            collection_name: Name of the collection
            persist_directory: Directory to persist the database
            **kwargs: Additional arguments
            
        Returns:
            ChromaVectorStore instance
        """
        if persist_directory:
            Path(persist_directory).mkdir(parents=True, exist_ok=True)
            
        store = Chroma.from_texts(
            texts=texts,
            embedding=embedding_function,
            metadatas=metadatas,
            collection_name=collection_name,
            persist_directory=persist_directory,
            **kwargs
        )
        
        instance = cls(
            embedding_function=embedding_function,
            collection_name=collection_name,
            persist_directory=persist_directory,
            **kwargs
        )
        instance.store = store
        return instance
    
    def add_documents(self, documents: List[Document], **kwargs) -> List[str]:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of documents to add
            **kwargs: Additional arguments
            
        Returns:
            List of document IDs
        """
        return self.store.add_documents(documents, **kwargs)
    
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
        return self.store.add_texts(texts, metadatas=metadatas, **kwargs)
    
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Document]:
        """
        Search for similar documents.
        
        Args:
            query: Query text
            k: Number of results to return
            filter: Optional metadata filter
            **kwargs: Additional arguments
            
        Returns:
            List of similar documents
        """
        return self.store.similarity_search(query, k=k, filter=filter, **kwargs)
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Tuple[Document, float]]:
        """
        Search for similar documents with similarity scores.
        
        Args:
            query: Query text
            k: Number of results to return
            filter: Optional metadata filter
            **kwargs: Additional arguments
            
        Returns:
            List of (document, score) tuples
        """
        return self.store.similarity_search_with_score(query, k=k, filter=filter, **kwargs)
    
    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Document]:
        """
        Search for similar documents by embedding vector.
        
        Args:
            embedding: Query embedding vector
            k: Number of results to return
            filter: Optional metadata filter
            **kwargs: Additional arguments
            
        Returns:
            List of similar documents
        """
        return self.store.similarity_search_by_vector(embedding, k=k, filter=filter, **kwargs)
    
    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Document]:
        """
        Return docs selected using the maximal marginal relevance.
        Maximal marginal relevance optimizes for similarity to query AND diversity.
        
        Args:
            query: Query text
            k: Number of results to return
            fetch_k: Number of documents to fetch for MMR algorithm
            lambda_mult: Diversity parameter (0=max diversity, 1=min diversity)
            filter: Optional metadata filter
            **kwargs: Additional arguments
            
        Returns:
            List of documents selected by MMR
        """
        return self.store.max_marginal_relevance_search(
            query, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult, filter=filter, **kwargs
        )
    
    def delete(self, ids: Optional[List[str]] = None, **kwargs) -> Optional[bool]:
        """
        Delete documents from the vector store.
        
        Args:
            ids: List of document IDs to delete
            **kwargs: Additional arguments
            
        Returns:
            Success status
        """
        if ids:
            self.store.delete(ids=ids, **kwargs)
            return True
        return False
    
    def persist(self) -> None:
        """Persist the vector store to disk."""
        if self.persist_directory:
            # Chroma automatically persists when persist_directory is set
            pass
        else:
            raise ValueError("No persist_directory specified")
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the collection.
        
        Returns:
            Dictionary with collection information
        """
        collection = self.store._collection
        return {
            "name": self.collection_name,
            "count": collection.count(),
            "persist_directory": self.persist_directory,
        }
    
    def get(
        self,
        ids: Optional[List[str]] = None,
        where: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Get documents from the collection.
        
        Args:
            ids: Optional list of document IDs
            where: Optional metadata filter
            limit: Optional limit on number of results
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with documents and metadata
        """
        return self.store.get(ids=ids, where=where, limit=limit, **kwargs)
    
    def update_document(self, document_id: str, document: Document) -> None:
        """
        Update a document in the collection.
        
        Args:
            document_id: ID of the document to update
            document: New document content
        """
        self.store.update_document(document_id=document_id, document=document)
    
    def clear(self) -> None:
        """Clear all documents from the collection."""
        collection = self.store._collection
        collection.delete()

# Made with Bob
