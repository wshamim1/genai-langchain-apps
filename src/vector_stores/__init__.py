"""
Vector Stores Module
Provides vector store implementations for document embeddings and similarity search.
"""

# Import base class (no external dependencies)
from .base import BaseVectorStore

# Import Chroma store (requires langchain-chroma)
from .chroma_store import ChromaVectorStore

__all__ = ['BaseVectorStore', 'ChromaVectorStore']
