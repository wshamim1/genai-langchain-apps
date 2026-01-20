"""
PDF Document Loader
Uses LangChain's PDF loaders to load PDF documents.
"""

from typing import List, Optional
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, PDFMinerLoader, PyMuPDFLoader

from .base import BaseDocumentLoader, Document


class PDFLoader(BaseDocumentLoader):
    """
    PDF Document Loader using LangChain.
    Supports multiple PDF parsing backends.
    """
    
    def __init__(
        self,
        file_path: str,
        backend: str = "pypdf",
        extract_images: bool = False,
        **kwargs
    ):
        """
        Initialize PDF loader.
        
        Args:
            file_path: Path to the PDF file
            backend: PDF parsing backend ('pypdf', 'pdfminer', 'pymupdf')
            extract_images: Whether to extract images (pymupdf only)
            **kwargs: Additional parameters for the loader
        """
        super().__init__(file_path, **kwargs)
        
        self.backend = backend.lower()
        self.extract_images = extract_images
        
        # Initialize the appropriate LangChain loader
        self.loader = self._initialize_loader()
    
    def _initialize_loader(self):
        """Initialize the appropriate LangChain PDF loader based on backend."""
        file_path_str = str(self.file_path)
        
        if self.backend == "pypdf":
            return PyPDFLoader(file_path_str)
        elif self.backend == "pdfminer":
            return PDFMinerLoader(file_path_str)
        elif self.backend == "pymupdf":
            return PyMuPDFLoader(
                file_path_str,
                extract_images=self.extract_images
            )
        else:
            raise ValueError(
                f"Unknown backend: {self.backend}. "
                "Choose from: 'pypdf', 'pdfminer', 'pymupdf'"
            )
    
    def load(self) -> List[Document]:
        """
        Load the PDF document.
        
        Returns:
            List of Document objects (one per page)
        """
        documents = self.loader.load()
        
        # Add custom metadata
        for i, doc in enumerate(documents):
            doc.metadata.update({
                "source": str(self.file_path),
                "file_name": self.file_path.name,
                "page": i + 1,
                "backend": self.backend,
            })
        
        return documents
    
    def load_page(self, page_number: int) -> Optional[Document]:
        """
        Load a specific page from the PDF.
        
        Args:
            page_number: Page number to load (1-indexed)
            
        Returns:
            Document object for the specified page, or None if page doesn't exist
        """
        documents = self.load()
        
        if 1 <= page_number <= len(documents):
            return documents[page_number - 1]
        return None
    
    def get_page_count(self) -> int:
        """Get the total number of pages in the PDF."""
        return len(self.load())
    
    def get_text(self) -> str:
        """Get all text from the PDF as a single string."""
        documents = self.load()
        return "\n\n".join([doc.page_content for doc in documents])
    
    @staticmethod
    def get_available_backends() -> List[str]:
        """Get list of available PDF parsing backends."""
        return ["pypdf", "pdfminer", "pymupdf"]

# Made with Bob
