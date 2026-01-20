"""
Chroma Vector Store Examples
Demonstrates using Chroma for document embeddings and similarity search.
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

from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from src.vector_stores import ChromaVectorStore
from src.document_loaders import PDFLoader, CSVLoader


def example_basic_vector_store():
    """Basic vector store creation and search."""
    print("\n=== Basic Vector Store ===")
    
    # Initialize embeddings
    embeddings = OpenAIEmbeddings()
    
    # Create sample documents
    documents = [
        Document(
            page_content="Python is a high-level programming language.",
            metadata={"source": "python_doc", "topic": "programming"}
        ),
        Document(
            page_content="JavaScript is used for web development.",
            metadata={"source": "js_doc", "topic": "programming"}
        ),
        Document(
            page_content="Machine learning is a subset of artificial intelligence.",
            metadata={"source": "ml_doc", "topic": "ai"}
        ),
        Document(
            page_content="Neural networks are inspired by biological neurons.",
            metadata={"source": "nn_doc", "topic": "ai"}
        ),
    ]
    
    # Create vector store from documents
    vector_store = ChromaVectorStore.from_documents(
        documents=documents,
        embedding_function=embeddings,
        collection_name="demo_collection",
        persist_directory="./databases/chroma"
    )
    
    print(f"✓ Created vector store with {len(documents)} documents")
    
    # Perform similarity search
    query = "What is Python?"
    results = vector_store.similarity_search(query, k=2)
    
    print(f"\nQuery: '{query}'")
    print(f"Found {len(results)} similar documents:")
    for i, doc in enumerate(results, 1):
        print(f"\n{i}. {doc.page_content}")
        print(f"   Metadata: {doc.metadata}")
    
    return vector_store


def example_similarity_search_with_scores(vector_store):
    """Search with similarity scores."""
    print("\n=== Similarity Search with Scores ===")
    
    query = "artificial intelligence and neural networks"
    results = vector_store.similarity_search_with_score(query, k=3)
    
    print(f"Query: '{query}'")
    print(f"Found {len(results)} results with scores:")
    for i, (doc, score) in enumerate(results, 1):
        print(f"\n{i}. Score: {score:.4f}")
        print(f"   Content: {doc.page_content}")
        print(f"   Metadata: {doc.metadata}")


def example_add_documents(vector_store):
    """Add more documents to existing vector store."""
    print("\n=== Adding Documents ===")
    
    new_documents = [
        Document(
            page_content="Docker is a containerization platform.",
            metadata={"source": "docker_doc", "topic": "devops"}
        ),
        Document(
            page_content="Kubernetes orchestrates containerized applications.",
            metadata={"source": "k8s_doc", "topic": "devops"}
        ),
    ]
    
    ids = vector_store.add_documents(new_documents)
    print(f"✓ Added {len(new_documents)} documents")
    print(f"  Document IDs: {ids}")
    
    # Search for the new documents
    query = "container orchestration"
    results = vector_store.similarity_search(query, k=2)
    
    print(f"\nQuery: '{query}'")
    print("Results:")
    for i, doc in enumerate(results, 1):
        print(f"{i}. {doc.page_content}")


def example_metadata_filtering(vector_store):
    """Filter search results by metadata."""
    print("\n=== Metadata Filtering ===")
    
    query = "programming"
    
    # Search without filter
    print("Without filter:")
    results = vector_store.similarity_search(query, k=3)
    for i, doc in enumerate(results, 1):
        print(f"{i}. {doc.page_content} (topic: {doc.metadata.get('topic')})")
    
    # Search with filter
    print("\nWith filter (topic='ai'):")
    results = vector_store.similarity_search(
        query,
        k=3,
        filter={"topic": "ai"}
    )
    for i, doc in enumerate(results, 1):
        print(f"{i}. {doc.page_content} (topic: {doc.metadata.get('topic')})")


def example_mmr_search(vector_store):
    """Maximal Marginal Relevance search for diversity."""
    print("\n=== MMR Search (Diverse Results) ===")
    
    query = "programming languages and AI"
    
    # Regular similarity search
    print("Regular similarity search:")
    results = vector_store.similarity_search(query, k=3)
    for i, doc in enumerate(results, 1):
        print(f"{i}. {doc.page_content[:60]}...")
    
    # MMR search for diversity
    print("\nMMR search (more diverse):")
    results = vector_store.max_marginal_relevance_search(
        query,
        k=3,
        fetch_k=10,
        lambda_mult=0.5  # 0=max diversity, 1=min diversity
    )
    for i, doc in enumerate(results, 1):
        print(f"{i}. {doc.page_content[:60]}...")


def example_with_pdf_documents():
    """Load PDF documents and add to vector store."""
    print("\n=== Vector Store with PDF Documents ===")
    
    pdf_path = "data/sample.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"PDF file not found at: {pdf_path}")
        print("Run the PDF loader example first to create the sample PDF.")
        return
    
    # Load PDF
    pdf_loader = PDFLoader(pdf_path)
    pdf_documents = pdf_loader.load()
    
    print(f"Loaded {len(pdf_documents)} pages from PDF")
    
    # Create vector store
    embeddings = OpenAIEmbeddings()
    vector_store = ChromaVectorStore.from_documents(
        documents=pdf_documents,
        embedding_function=embeddings,
        collection_name="pdf_collection",
        persist_directory="./databases/chroma_pdf"
    )
    
    print(f"✓ Created vector store with PDF content")
    
    # Search the PDF content
    query = "What is this PDF about?"
    results = vector_store.similarity_search(query, k=2)
    
    print(f"\nQuery: '{query}'")
    print("Results:")
    for i, doc in enumerate(results, 1):
        print(f"\n{i}. {doc.page_content[:150]}...")
        print(f"   Page: {doc.metadata.get('page', 'N/A')}")


def example_with_csv_documents():
    """Load CSV documents and add to vector store."""
    print("\n=== Vector Store with CSV Documents ===")
    
    csv_path = "data/sample_data.csv"
    
    if not os.path.exists(csv_path):
        print(f"CSV file not found at: {csv_path}")
        return
    
    # Load CSV
    csv_loader = CSVLoader(csv_path)
    csv_documents = csv_loader.load()
    
    print(f"Loaded {len(csv_documents)} rows from CSV")
    
    # Create vector store
    embeddings = OpenAIEmbeddings()
    vector_store = ChromaVectorStore.from_documents(
        documents=csv_documents,
        embedding_function=embeddings,
        collection_name="csv_collection",
        persist_directory="./databases/chroma_csv"
    )
    
    print(f"✓ Created vector store with CSV content")
    
    # Search the CSV content
    query = "laptop computer"
    results = vector_store.similarity_search(query, k=3)
    
    print(f"\nQuery: '{query}'")
    print("Results:")
    for i, doc in enumerate(results, 1):
        print(f"\n{i}. {doc.page_content[:100]}...")


def example_collection_info(vector_store):
    """Get information about the collection."""
    print("\n=== Collection Information ===")
    
    info = vector_store.get_collection_info()
    print("Collection Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")


def example_as_retriever(vector_store):
    """Use vector store as a retriever."""
    print("\n=== Using as Retriever ===")
    
    # Create retriever
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 2}
    )
    
    query = "What programming languages are mentioned?"
    results = retriever.invoke(query)
    
    print(f"Query: '{query}'")
    print(f"Retrieved {len(results)} documents:")
    for i, doc in enumerate(results, 1):
        print(f"\n{i}. {doc.page_content}")


def example_delete_documents(vector_store):
    """Delete documents from the collection."""
    print("\n=== Deleting Documents ===")
    
    # Get current count
    info = vector_store.get_collection_info()
    print(f"Current document count: {info['count']}")
    
    # Get some document IDs
    results = vector_store.get(limit=2)
    if results and 'ids' in results and results['ids']:
        ids_to_delete = results['ids'][:1]  # Delete first document
        print(f"Deleting document ID: {ids_to_delete[0]}")
        
        vector_store.delete(ids=ids_to_delete)
        
        # Check new count
        info = vector_store.get_collection_info()
        print(f"New document count: {info['count']}")
    else:
        print("No documents to delete")


if __name__ == "__main__":
    print("=" * 60)
    print("Chroma Vector Store Examples")
    print("=" * 60)
    
    try:
        # Basic examples
        vector_store = example_basic_vector_store()
        example_similarity_search_with_scores(vector_store)
        example_add_documents(vector_store)
        example_metadata_filtering(vector_store)
        example_mmr_search(vector_store)
        example_collection_info(vector_store)
        example_as_retriever(vector_store)
        example_delete_documents(vector_store)
        
        # Examples with document loaders
        example_with_pdf_documents()
        example_with_csv_documents()
        
        print("\n" + "=" * 60)
        print("✅ All examples completed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

# Made with Bob
