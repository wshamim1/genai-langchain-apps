"""
PDF Loader Examples
Demonstrates loading and processing PDF documents.
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

from src.document_loaders import PDFLoader


def example_basic_pdf_loading():
    """Basic PDF loading example."""
    print("\n=== Basic PDF Loading ===")
    
    # Note: You'll need to provide your own PDF file
    # For this example, we'll show the code structure
    pdf_path = "data/sample.pdf"
    
    try:
        loader = PDFLoader(pdf_path)
        documents = loader.load()
        
        print(f"Loaded {len(documents)} pages")
        print(f"First page preview: {documents[0].page_content[:200]}...")
        
    except FileNotFoundError:
        print(f"PDF file not found at: {pdf_path}")
        print("Please add a PDF file to test with.")


def example_different_backends():
    """Example using different PDF parsing backends."""
    print("\n=== Different PDF Backends ===")
    
    pdf_path = "data/sample.pdf"
    
    backends = PDFLoader.get_available_backends()
    print(f"Available backends: {backends}")
    
    for backend in backends:
        try:
            print(f"\nTrying backend: {backend}")
            loader = PDFLoader(pdf_path, backend=backend)
            documents = loader.load()
            print(f"  ✓ Successfully loaded {len(documents)} pages")
        except FileNotFoundError:
            print(f"  ⚠ PDF file not found")
            break
        except Exception as e:
            print(f"  ✗ Error with {backend}: {e}")


def example_load_specific_page():
    """Example loading a specific page."""
    print("\n=== Load Specific Page ===")
    
    pdf_path = "data/sample.pdf"
    
    try:
        loader = PDFLoader(pdf_path)
        
        # Load page 1
        page = loader.load_page(1)
        if page:
            print(f"Page 1 content: {page.page_content[:200]}...")
            print(f"Page metadata: {page.metadata}")
        
    except FileNotFoundError:
        print(f"PDF file not found at: {pdf_path}")


def example_split_into_chunks():
    """Example splitting PDF into chunks."""
    print("\n=== Split PDF into Chunks ===")
    
    pdf_path = "data/sample.pdf"
    
    try:
        loader = PDFLoader(pdf_path)
        
        # Split into chunks
        chunks = loader.load_and_split(chunk_size=500, chunk_overlap=50)
        
        print(f"Split into {len(chunks)} chunks")
        print(f"First chunk: {chunks[0].page_content[:200]}...")
        
    except FileNotFoundError:
        print(f"PDF file not found at: {pdf_path}")


def example_get_metadata():
    """Example getting PDF metadata."""
    print("\n=== PDF Metadata ===")
    
    pdf_path = "data/sample.pdf"
    
    try:
        loader = PDFLoader(pdf_path)
        metadata = loader.get_metadata()
        
        print("File metadata:")
        for key, value in metadata.items():
            print(f"  {key}: {value}")
        
        print(f"\nTotal pages: {loader.get_page_count()}")
        
    except FileNotFoundError:
        print(f"PDF file not found at: {pdf_path}")


def example_extract_all_text():
    """Example extracting all text from PDF."""
    print("\n=== Extract All Text ===")
    
    pdf_path = "data/sample.pdf"
    
    try:
        loader = PDFLoader(pdf_path)
        text = loader.get_text()
        
        print(f"Total text length: {len(text)} characters")
        print(f"Preview: {text[:300]}...")
        
    except FileNotFoundError:
        print(f"PDF file not found at: {pdf_path}")


def create_sample_pdf():
    """Helper function to create a sample PDF for testing."""
    print("\n=== Creating Sample PDF ===")
    
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
        
        pdf_path = "data/sample.pdf"
        os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
        
        c = canvas.Canvas(pdf_path, pagesize=letter)
        
        # Page 1
        c.drawString(100, 750, "Sample PDF Document")
        c.drawString(100, 730, "Page 1")
        c.drawString(100, 700, "This is a sample PDF created for testing the PDF loader.")
        c.drawString(100, 680, "It contains multiple pages with different content.")
        c.showPage()
        
        # Page 2
        c.drawString(100, 750, "Page 2")
        c.drawString(100, 730, "This is the second page of the sample PDF.")
        c.drawString(100, 700, "PDF loaders can extract text from each page separately.")
        c.showPage()
        
        c.save()
        print(f"✓ Sample PDF created at: {pdf_path}")
        return True
        
    except ImportError:
        print("reportlab not installed. Install with: pip install reportlab")
        print("Or provide your own PDF file for testing.")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("PDF Loader Examples")
    print("=" * 60)
    
    # Try to create a sample PDF first
    pdf_created = create_sample_pdf()
    
    if pdf_created:
        try:
            example_basic_pdf_loading()
            example_different_backends()
            example_load_specific_page()
            example_split_into_chunks()
            example_get_metadata()
            example_extract_all_text()
            
            print("\n" + "=" * 60)
            print("✅ All examples completed!")
            print("=" * 60)
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
    else:
        print("\nPlease install reportlab or provide a PDF file to test with.")
        print("Install: pip install reportlab")

# Made with Bob
