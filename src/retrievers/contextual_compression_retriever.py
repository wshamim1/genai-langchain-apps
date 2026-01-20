"""
Contextual Compression Retriever
Compresses retrieved documents to only include relevant information.
"""

from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain.retrievers import ContextualCompressionRetriever as LangChainCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

from .base import BaseRetriever


class ContextualCompressionRetriever(BaseRetriever):
    """
    Contextual compression retriever implementation.
    Retrieves documents and compresses them to only relevant content.
    """
    
    def __init__(
        self,
        base_retriever: Any,
        llm: BaseLanguageModel,
        **kwargs
    ):
        """
        Initialize contextual compression retriever.
        
        Args:
            base_retriever: Base retriever to use
            llm: Language model for compression
            **kwargs: Additional arguments
        """
        super().__init__(**kwargs)
        self.base_retriever = base_retriever
        self.llm = llm
        
        # Create compressor
        self.compressor = LLMChainExtractor.from_llm(llm)
        
        # Create the compression retriever
        self.retriever = LangChainCompressionRetriever(
            base_compressor=self.compressor,
            base_retriever=base_retriever
        )
    
    def get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        """
        Retrieve and compress relevant documents.
        
        Args:
            query: The query string
            **kwargs: Additional arguments
            
        Returns:
            List of compressed documents
        """
        return self.retriever.invoke(query)
    
    def get_uncompressed_documents(self, query: str, **kwargs) -> List[Document]:
        """
        Get documents without compression.
        
        Args:
            query: The query string
            **kwargs: Additional arguments
            
        Returns:
            List of uncompressed documents
        """
        return self.base_retriever.invoke(query)
    
    @classmethod
    def from_retriever(
        cls,
        base_retriever: Any,
        llm: BaseLanguageModel,
        **kwargs
    ) -> "ContextualCompressionRetriever":
        """
        Create a contextual compression retriever from a base retriever.
        
        Args:
            base_retriever: Base retriever
            llm: Language model for compression
            **kwargs: Additional arguments
            
        Returns:
            ContextualCompressionRetriever instance
        """
        return cls(
            base_retriever=base_retriever,
            llm=llm,
            **kwargs
        )

# Made with Bob
