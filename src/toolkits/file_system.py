"""
File system toolkit for file operations.
"""

from typing import List
from .base import BaseToolkit
from ..tools.file_operations import FileReadTool, FileWriteTool, FileListTool


class FileSystemToolkit(BaseToolkit):
    """Toolkit for file system operations.
    
    This toolkit provides tools for reading, writing, and listing files.
    """
    
    base_directory: str = "."
    
    def get_tools(self) -> List:
        """Get all file system tools.
        
        Returns:
            List of file system tools
        """
        return [
            FileReadTool(base_directory=self.base_directory),
            FileWriteTool(base_directory=self.base_directory),
            FileListTool(base_directory=self.base_directory),
        ]
    
    @staticmethod
    def create(base_directory: str = ".") -> "FileSystemToolkit":
        """Factory method to create a file system toolkit.
        
        Args:
            base_directory: Base directory for file operations
            
        Returns:
            A new FileSystemToolkit instance
        """
        return FileSystemToolkit(base_directory=base_directory)

# Made with Bob
