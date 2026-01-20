"""
File operation tools for reading and writing files.
"""

import os
from typing import Any
from .base import BaseTool


class FileReadTool(BaseTool):
    """Tool for reading file contents.
    
    This tool can read text files and return their contents.
    """
    
    name: str = "file_read"
    description: str = (
        "Read the contents of a file. "
        "Input should be a file path. "
        "Example: 'data/sample.txt'"
    )
    base_directory: str = "."
    
    def _run(self, file_path: str) -> Any:
        """Read a file's contents.
        
        Args:
            file_path: Path to the file to read
            
        Returns:
            The contents of the file
        """
        try:
            # Construct full path
            full_path = os.path.join(self.base_directory, file_path)
            
            # Security check: ensure path is within base directory
            full_path = os.path.abspath(full_path)
            base_path = os.path.abspath(self.base_directory)
            
            if not full_path.startswith(base_path):
                return "Error: Access denied. File path is outside the allowed directory."
            
            # Check if file exists
            if not os.path.exists(full_path):
                return f"Error: File not found: {file_path}"
            
            # Check if it's a file (not a directory)
            if not os.path.isfile(full_path):
                return f"Error: Path is not a file: {file_path}"
            
            # Read the file
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            return f"File contents of {file_path}:\n\n{content}"
            
        except UnicodeDecodeError:
            return f"Error: File {file_path} is not a text file or uses an unsupported encoding."
        except PermissionError:
            return f"Error: Permission denied to read file: {file_path}"
        except Exception as e:
            return f"Error reading file: {str(e)}"
    
    @staticmethod
    def create(base_directory: str = ".") -> "FileReadTool":
        """Factory method to create a file read tool.
        
        Args:
            base_directory: Base directory for file operations
            
        Returns:
            A new FileReadTool instance
        """
        return FileReadTool(base_directory=base_directory)


class FileWriteTool(BaseTool):
    """Tool for writing content to files.
    
    This tool can create or overwrite files with new content.
    """
    
    name: str = "file_write"
    description: str = (
        "Write content to a file. Creates the file if it doesn't exist, overwrites if it does. "
        "Input should be in format: 'file_path|content'. "
        "Example: 'output.txt|Hello, World!'"
    )
    base_directory: str = "."
    
    def _run(self, input_str: str) -> Any:
        """Write content to a file.
        
        Args:
            input_str: String in format "file_path|content"
            
        Returns:
            Success or error message
        """
        try:
            # Parse input
            if '|' not in input_str:
                return "Error: Input must be in format 'file_path|content'"
            
            parts = input_str.split('|', 1)
            file_path = parts[0].strip()
            content = parts[1] if len(parts) > 1 else ""
            
            # Construct full path
            full_path = os.path.join(self.base_directory, file_path)
            
            # Security check: ensure path is within base directory
            full_path = os.path.abspath(full_path)
            base_path = os.path.abspath(self.base_directory)
            
            if not full_path.startswith(base_path):
                return "Error: Access denied. File path is outside the allowed directory."
            
            # Create directory if it doesn't exist
            directory = os.path.dirname(full_path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
            
            # Write the file
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return f"Successfully wrote {len(content)} characters to {file_path}"
            
        except PermissionError:
            return f"Error: Permission denied to write file: {file_path}"
        except Exception as e:
            return f"Error writing file: {str(e)}"
    
    @staticmethod
    def create(base_directory: str = ".") -> "FileWriteTool":
        """Factory method to create a file write tool.
        
        Args:
            base_directory: Base directory for file operations
            
        Returns:
            A new FileWriteTool instance
        """
        return FileWriteTool(base_directory=base_directory)


class FileListTool(BaseTool):
    """Tool for listing files in a directory.
    
    This tool can list all files in a specified directory.
    """
    
    name: str = "file_list"
    description: str = (
        "List files in a directory. "
        "Input should be a directory path (or empty for current directory). "
        "Example: 'data/' or leave empty for current directory"
    )
    base_directory: str = "."
    
    def _run(self, directory: str = "") -> Any:
        """List files in a directory.
        
        Args:
            directory: Directory path to list (relative to base_directory)
            
        Returns:
            List of files in the directory
        """
        try:
            # Construct full path
            if directory:
                full_path = os.path.join(self.base_directory, directory)
            else:
                full_path = self.base_directory
            
            # Security check: ensure path is within base directory
            full_path = os.path.abspath(full_path)
            base_path = os.path.abspath(self.base_directory)
            
            if not full_path.startswith(base_path):
                return "Error: Access denied. Directory path is outside the allowed directory."
            
            # Check if directory exists
            if not os.path.exists(full_path):
                return f"Error: Directory not found: {directory or 'current directory'}"
            
            # Check if it's a directory
            if not os.path.isdir(full_path):
                return f"Error: Path is not a directory: {directory}"
            
            # List files
            items = os.listdir(full_path)
            
            files = []
            directories = []
            
            for item in sorted(items):
                item_path = os.path.join(full_path, item)
                if os.path.isfile(item_path):
                    size = os.path.getsize(item_path)
                    files.append(f"  📄 {item} ({size} bytes)")
                elif os.path.isdir(item_path):
                    directories.append(f"  📁 {item}/")
            
            result = f"Contents of {directory or 'current directory'}:\n\n"
            
            if directories:
                result += "Directories:\n" + "\n".join(directories) + "\n\n"
            
            if files:
                result += "Files:\n" + "\n".join(files)
            
            if not directories and not files:
                result += "Directory is empty"
            
            return result
            
        except PermissionError:
            return f"Error: Permission denied to list directory: {directory}"
        except Exception as e:
            return f"Error listing directory: {str(e)}"
    
    @staticmethod
    def create(base_directory: str = ".") -> "FileListTool":
        """Factory method to create a file list tool.
        
        Args:
            base_directory: Base directory for file operations
            
        Returns:
            A new FileListTool instance
        """
        return FileListTool(base_directory=base_directory)

# Made with Bob
