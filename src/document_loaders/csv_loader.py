"""
CSV Document Loader
Uses LangChain's CSV loader to load CSV documents.
"""

from typing import List, Optional, Dict, Any
from pathlib import Path

from langchain_community.document_loaders import CSVLoader as LangChainCSVLoader

from .base import BaseDocumentLoader, Document


class CSVLoader(BaseDocumentLoader):
    """
    CSV Document Loader using LangChain.
    Loads CSV files and converts each row to a document.
    """
    
    def __init__(
        self,
        file_path: str,
        source_column: Optional[str] = None,
        csv_args: Optional[Dict[str, Any]] = None,
        encoding: str = "utf-8",
        **kwargs
    ):
        """
        Initialize CSV loader.
        
        Args:
            file_path: Path to the CSV file
            source_column: Column to use as the source (optional)
            csv_args: Additional arguments for csv.DictReader
            encoding: File encoding (default: utf-8)
            **kwargs: Additional parameters
        """
        super().__init__(file_path, **kwargs)
        
        self.source_column = source_column
        self.csv_args = csv_args or {}
        self.encoding = encoding
        
        # Initialize the LangChain CSV loader
        self.loader = self._initialize_loader()
    
    def _initialize_loader(self):
        """Initialize the LangChain CSV loader."""
        return LangChainCSVLoader(
            file_path=str(self.file_path),
            source_column=self.source_column,
            csv_args=self.csv_args,
            encoding=self.encoding,
        )
    
    def load(self) -> List[Document]:
        """
        Load the CSV document.
        
        Returns:
            List of Document objects (one per row)
        """
        documents = self.loader.load()
        
        # Add custom metadata
        for i, doc in enumerate(documents):
            doc.metadata.update({
                "source": str(self.file_path),
                "file_name": self.file_path.name,
                "row": i + 1,
            })
        
        return documents
    
    def load_as_dataframe(self):
        """
        Load CSV as a pandas DataFrame.
        
        Returns:
            pandas DataFrame
        """
        try:
            import pandas as pd
            return pd.read_csv(self.file_path, encoding=self.encoding, **self.csv_args)
        except ImportError:
            raise ImportError("pandas is required for load_as_dataframe(). Install with: pip install pandas")
    
    def get_columns(self) -> List[str]:
        """Get list of column names from the CSV."""
        try:
            import csv
            with open(self.file_path, 'r', encoding=self.encoding) as f:
                reader = csv.DictReader(f, **self.csv_args)
                return list(reader.fieldnames or [])
        except Exception as e:
            raise Exception(f"Error reading CSV columns: {e}")
    
    def get_row_count(self) -> int:
        """Get the total number of rows in the CSV (excluding header)."""
        return len(self.load())
    
    def filter_by_column(self, column: str, value: Any) -> List[Document]:
        """
        Filter documents by a specific column value.
        
        Args:
            column: Column name to filter by
            value: Value to match
            
        Returns:
            List of filtered Document objects
        """
        documents = self.load()
        return [
            doc for doc in documents
            if doc.metadata.get(column) == value
        ]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary information about the CSV file."""
        columns = self.get_columns()
        row_count = self.get_row_count()
        
        return {
            "file_path": str(self.file_path),
            "file_name": self.file_path.name,
            "columns": columns,
            "column_count": len(columns),
            "row_count": row_count,
            "encoding": self.encoding,
        }

# Made with Bob
