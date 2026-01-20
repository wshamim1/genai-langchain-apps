"""
Base Document Loader Interface
Wrapper around LangChain's document loaders.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from pathlib import Path

from langchain_core.documents import Document as LangChainDocument


# Re-export LangChain's Document class
Document = LangChainDocument


class BaseDocumentLoader(ABC):
    """
    Base class for document loaders.
    Provides a simplified interface around LangChain's document loaders.
    """
    
    def __init__(self, file_path: str, **kwargs):
        """
        Initialize the document loader.
        
        Args:
            file_path: Path to the file to load
            **kwargs: Additional loader-specific parameters
        """
        self.file_path = Path(file_path)
        self.kwargs = kwargs
        
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
    
    @abstractmethod
    def load(self) -> List[Document]:
        """
        Load documents from the file.
        
        Returns:
            List of Document objects
        """
        pass
    
    def load_and_split(self, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
        """
        Load documents and split them into chunks.
        
        Args:
            chunk_size: Size of each chunk
            chunk_overlap: Overlap between chunks
            
        Returns:
            List of Document chunks
        """
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        
        documents = self.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        
        return text_splitter.split_documents(documents)
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get file metadata."""
        return {
            "file_path": str(self.file_path),
            "file_name": self.file_path.name,
            "file_size": self.file_path.stat().st_size,
            "file_extension": self.file_path.suffix,
        }
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(file_path='{self.file_path}')"

# Made with Bob
