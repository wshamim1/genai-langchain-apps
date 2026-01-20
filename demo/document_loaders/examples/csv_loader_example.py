"""
CSV Loader Examples
Demonstrates loading and processing CSV documents.
"""

import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).parent.parent.parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.document_loaders import CSVLoader


def example_basic_csv_loading():
    """Basic CSV loading example."""
    print("\n=== Basic CSV Loading ===")
    
    csv_path = "data/sample_data.csv"
    
    try:
        loader = CSVLoader(csv_path)
        documents = loader.load()
        
        print(f"Loaded {len(documents)} rows")
        print(f"\nFirst document:")
        print(f"  Content: {documents[0].page_content}")
        print(f"  Metadata: {documents[0].metadata}")
        
    except FileNotFoundError:
        print(f"CSV file not found at: {csv_path}")
    except Exception as e:
        print(f"Error: {e}")


def example_get_columns():
    """Example getting CSV columns."""
    print("\n=== Get CSV Columns ===")
    
    csv_path = "data/sample_data.csv"
    
    try:
        loader = CSVLoader(csv_path)
        columns = loader.get_columns()
        
        print(f"Columns: {columns}")
        print(f"Total columns: {len(columns)}")
        
    except FileNotFoundError:
        print(f"CSV file not found at: {csv_path}")


def example_get_summary():
    """Example getting CSV summary."""
    print("\n=== CSV Summary ===")
    
    csv_path = "data/sample_data.csv"
    
    try:
        loader = CSVLoader(csv_path)
        summary = loader.get_summary()
        
        print("CSV Summary:")
        for key, value in summary.items():
            print(f"  {key}: {value}")
        
    except FileNotFoundError:
        print(f"CSV file not found at: {csv_path}")


def example_filter_by_column():
    """Example filtering CSV by column value."""
    print("\n=== Filter by Column ===")
    
    csv_path = "data/sample_data.csv"
    
    try:
        loader = CSVLoader(csv_path)
        
        # Filter by category
        electronics = loader.filter_by_column("category", "Electronics")
        
        print(f"Found {len(electronics)} Electronics items:")
        for doc in electronics:
            print(f"  - {doc.page_content[:100]}...")
        
    except FileNotFoundError:
        print(f"CSV file not found at: {csv_path}")


def example_load_as_dataframe():
    """Example loading CSV as pandas DataFrame."""
    print("\n=== Load as DataFrame ===")
    
    csv_path = "data/sample_data.csv"
    
    try:
        loader = CSVLoader(csv_path)
        df = loader.load_as_dataframe()
        
        print("DataFrame Info:")
        print(df.info())
        print("\nFirst few rows:")
        print(df.head())
        
    except ImportError:
        print("pandas not installed. Install with: pip install pandas")
    except FileNotFoundError:
        print(f"CSV file not found at: {csv_path}")


def example_split_into_chunks():
    """Example splitting CSV into chunks."""
    print("\n=== Split CSV into Chunks ===")
    
    csv_path = "data/sample_data.csv"
    
    try:
        loader = CSVLoader(csv_path)
        
        # Split into chunks
        chunks = loader.load_and_split(chunk_size=200, chunk_overlap=20)
        
        print(f"Split into {len(chunks)} chunks")
        print(f"\nFirst chunk:")
        print(f"  {chunks[0].page_content}")
        
    except FileNotFoundError:
        print(f"CSV file not found at: {csv_path}")


def example_custom_source_column():
    """Example using a custom source column."""
    print("\n=== Custom Source Column ===")
    
    csv_path = "data/sample_data.csv"
    
    try:
        # Use 'name' column as the source
        loader = CSVLoader(csv_path, source_column="name")
        documents = loader.load()
        
        print(f"Loaded {len(documents)} documents")
        print(f"\nFirst document with custom source:")
        print(f"  Content: {documents[0].page_content[:100]}...")
        print(f"  Source: {documents[0].metadata.get('source')}")
        
    except FileNotFoundError:
        print(f"CSV file not found at: {csv_path}")


def example_process_all_rows():
    """Example processing all rows."""
    print("\n=== Process All Rows ===")
    
    csv_path = "data/sample_data.csv"
    
    try:
        loader = CSVLoader(csv_path)
        documents = loader.load()
        
        print(f"Processing {len(documents)} rows:\n")
        
        for i, doc in enumerate(documents, 1):
            # Extract information from content
            content = doc.page_content
            print(f"{i}. {content[:80]}...")
        
    except FileNotFoundError:
        print(f"CSV file not found at: {csv_path}")


def example_get_metadata():
    """Example getting file metadata."""
    print("\n=== File Metadata ===")
    
    csv_path = "data/sample_data.csv"
    
    try:
        loader = CSVLoader(csv_path)
        metadata = loader.get_metadata()
        
        print("File Metadata:")
        for key, value in metadata.items():
            print(f"  {key}: {value}")
        
        print(f"\nRow count: {loader.get_row_count()}")
        
    except FileNotFoundError:
        print(f"CSV file not found at: {csv_path}")


if __name__ == "__main__":
    print("=" * 60)
    print("CSV Loader Examples")
    print("=" * 60)
    
    try:
        example_basic_csv_loading()
        example_get_columns()
        example_get_summary()
        example_filter_by_column()
        example_load_as_dataframe()
        example_split_into_chunks()
        example_custom_source_column()
        example_process_all_rows()
        example_get_metadata()
        
        print("\n" + "=" * 60)
        print("✅ All examples completed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

# Made with Bob
