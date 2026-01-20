"""
Retrievers Module
Provides retriever implementations for document retrieval from vector stores.
"""

from .base import BaseRetriever
from .vector_store_retriever import VectorStoreRetriever

__all__ = ['BaseRetriever', 'VectorStoreRetriever']

# Try to import advanced retrievers (require additional dependencies)
try:
    from .multi_query_retriever import MultiQueryRetriever
    __all__.append('MultiQueryRetriever')
except ImportError:
    MultiQueryRetriever = None

try:
    from .contextual_compression_retriever import ContextualCompressionRetriever
    __all__.append('ContextualCompressionRetriever')
except ImportError:
    ContextualCompressionRetriever = None

# Made with Bob
