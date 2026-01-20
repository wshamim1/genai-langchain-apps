"""
Retriever Examples
Demonstrates using different retrievers for document retrieval.
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

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from src.vector_stores import ChromaVectorStore
from src.retrievers import VectorStoreRetriever

# Try to import advanced retrievers
try:
    from src.retrievers import MultiQueryRetriever, ContextualCompressionRetriever
    ADVANCED_RETRIEVERS_AVAILABLE = (MultiQueryRetriever is not None and
                                     ContextualCompressionRetriever is not None)
except (ImportError, AttributeError):
    ADVANCED_RETRIEVERS_AVAILABLE = False
    MultiQueryRetriever = None
    ContextualCompressionRetriever = None

if not ADVANCED_RETRIEVERS_AVAILABLE:
    print("Note: Advanced retrievers not available. Install langchain for full functionality.")


def setup_vector_store():
    """Create a vector store with sample documents."""
    print("\n=== Setting Up Vector Store ===")
    
    documents = [
        Document(
            page_content="Python is a high-level programming language known for its simplicity and readability. It's widely used in web development, data science, and automation.",
            metadata={"source": "python_doc", "topic": "programming"}
        ),
        Document(
            page_content="Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.",
            metadata={"source": "ml_doc", "topic": "ai"}
        ),
        Document(
            page_content="Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes that process information.",
            metadata={"source": "nn_doc", "topic": "ai"}
        ),
        Document(
            page_content="Docker is a platform for developing, shipping, and running applications in containers. It provides isolation and portability for applications.",
            metadata={"source": "docker_doc", "topic": "devops"}
        ),
        Document(
            page_content="Kubernetes is an open-source container orchestration platform that automates deployment, scaling, and management of containerized applications.",
            metadata={"source": "k8s_doc", "topic": "devops"}
        ),
        Document(
            page_content="React is a JavaScript library for building user interfaces. It uses a component-based architecture and virtual DOM for efficient rendering.",
            metadata={"source": "react_doc", "topic": "frontend"}
        ),
        Document(
            page_content="PostgreSQL is a powerful open-source relational database system. It supports advanced data types and performance optimization features.",
            metadata={"source": "postgres_doc", "topic": "database"}
        ),
        Document(
            page_content="Git is a distributed version control system for tracking changes in source code. It enables collaboration among developers.",
            metadata={"source": "git_doc", "topic": "tools"}
        ),
    ]
    
    embeddings = OpenAIEmbeddings()
    vector_store = ChromaVectorStore.from_documents(
        documents=documents,
        embedding_function=embeddings,
        collection_name="retriever_demo",
        persist_directory="./databases/chroma_retrievers"
    )
    
    print(f"✓ Created vector store with {len(documents)} documents")
    return vector_store


def example_basic_vector_store_retriever(vector_store):
    """Example using basic vector store retriever."""
    print("\n=== Basic Vector Store Retriever ===")
    
    # Create retriever
    retriever = VectorStoreRetriever.from_vector_store(
        vector_store=vector_store.store,
        k=3
    )
    
    # Retrieve documents
    query = "What is machine learning?"
    docs = retriever.get_relevant_documents(query)
    
    print(f"Query: '{query}'")
    print(f"Retrieved {len(docs)} documents:")
    for i, doc in enumerate(docs, 1):
        print(f"\n{i}. {doc.page_content[:100]}...")
        print(f"   Topic: {doc.metadata.get('topic')}")
    
    return retriever


def example_retriever_with_scores(vector_store):
    """Example retrieving documents with similarity scores."""
    print("\n=== Retriever with Similarity Scores ===")
    
    retriever = VectorStoreRetriever.from_vector_store(
        vector_store=vector_store.store,
        k=3
    )
    
    query = "container orchestration"
    docs_with_scores = retriever.get_relevant_documents_with_scores(query)
    
    print(f"Query: '{query}'")
    print(f"Retrieved {len(docs_with_scores)} documents with scores:")
    for i, (doc, score) in enumerate(docs_with_scores, 1):
        print(f"\n{i}. Score: {score:.4f}")
        print(f"   {doc.page_content[:80]}...")
        print(f"   Topic: {doc.metadata.get('topic')}")


def example_mmr_retriever(vector_store):
    """Example using MMR for diverse results."""
    print("\n=== MMR Retriever (Diverse Results) ===")
    
    retriever = VectorStoreRetriever.from_vector_store(
        vector_store=vector_store.store,
        k=4
    )
    
    # Enable MMR
    retriever.enable_mmr(fetch_k=8, lambda_mult=0.5)
    
    query = "programming and development"
    docs = retriever.get_relevant_documents(query)
    
    print(f"Query: '{query}'")
    print(f"Retrieved {len(docs)} diverse documents:")
    for i, doc in enumerate(docs, 1):
        print(f"\n{i}. {doc.page_content[:80]}...")
        print(f"   Topic: {doc.metadata.get('topic')}")


def example_score_threshold_retriever(vector_store):
    """Example using score threshold filtering."""
    print("\n=== Score Threshold Retriever ===")
    
    retriever = VectorStoreRetriever.from_vector_store(
        vector_store=vector_store.store,
        k=5
    )
    
    # Set score threshold
    retriever.set_score_threshold(0.3)
    
    query = "artificial intelligence"
    docs = retriever.get_relevant_documents(query)
    
    print(f"Query: '{query}'")
    print(f"Retrieved {len(docs)} documents above threshold:")
    for i, doc in enumerate(docs, 1):
        print(f"\n{i}. {doc.page_content[:80]}...")


def example_multi_query_retriever(vector_store):
    """Example using multi-query retriever."""
    print("\n=== Multi-Query Retriever ===")
    
    # Create base retriever
    base_retriever = vector_store.store.as_retriever(search_kwargs={"k": 3})
    
    # Create LLM for query generation
    llm = ChatOpenAI(temperature=0)
    
    # Create multi-query retriever
    retriever = MultiQueryRetriever.from_retriever(
        retriever=base_retriever,
        llm=llm,
        include_original=True
    )
    
    query = "How do containers work?"
    docs = retriever.get_relevant_documents(query)
    
    print(f"Original Query: '{query}'")
    print(f"Retrieved {len(docs)} documents (from multiple query perspectives):")
    for i, doc in enumerate(docs, 1):
        print(f"\n{i}. {doc.page_content[:80]}...")
        print(f"   Topic: {doc.metadata.get('topic')}")


def example_contextual_compression_retriever(vector_store):
    """Example using contextual compression retriever."""
    print("\n=== Contextual Compression Retriever ===")
    
    # Create base retriever
    base_retriever = vector_store.store.as_retriever(search_kwargs={"k": 3})
    
    # Create LLM for compression
    llm = ChatOpenAI(temperature=0)
    
    # Create compression retriever
    retriever = ContextualCompressionRetriever.from_retriever(
        base_retriever=base_retriever,
        llm=llm
    )
    
    query = "What is Python used for?"
    
    # Get uncompressed documents
    print(f"Query: '{query}'")
    print("\nUncompressed documents:")
    uncompressed = retriever.get_uncompressed_documents(query)
    for i, doc in enumerate(uncompressed, 1):
        print(f"\n{i}. Length: {len(doc.page_content)} chars")
        print(f"   {doc.page_content[:100]}...")
    
    # Get compressed documents
    print("\nCompressed documents:")
    compressed = retriever.get_relevant_documents(query)
    for i, doc in enumerate(compressed, 1):
        print(f"\n{i}. Length: {len(doc.page_content)} chars")
        print(f"   {doc.page_content}")


def example_dynamic_k_adjustment(vector_store):
    """Example dynamically adjusting k parameter."""
    print("\n=== Dynamic K Adjustment ===")
    
    retriever = VectorStoreRetriever.from_vector_store(
        vector_store=vector_store.store,
        k=2
    )
    
    query = "programming languages"
    
    # Retrieve with k=2
    print(f"Query: '{query}'")
    print("\nWith k=2:")
    docs = retriever.get_relevant_documents(query)
    print(f"Retrieved {len(docs)} documents")
    
    # Adjust to k=4
    retriever.set_k(4)
    print("\nWith k=4:")
    docs = retriever.get_relevant_documents(query)
    print(f"Retrieved {len(docs)} documents")
    for i, doc in enumerate(docs, 1):
        print(f"{i}. {doc.metadata.get('source')}: {doc.page_content[:60]}...")


def example_batch_retrieval(vector_store):
    """Example retrieving for multiple queries at once."""
    print("\n=== Batch Retrieval ===")
    
    retriever = VectorStoreRetriever.from_vector_store(
        vector_store=vector_store.store,
        k=2
    )
    
    queries = [
        "What is machine learning?",
        "How does Docker work?",
        "What is React?"
    ]
    
    results = retriever.batch(queries)
    
    print(f"Batch retrieved for {len(queries)} queries:")
    for query, docs in zip(queries, results):
        print(f"\nQuery: '{query}'")
        print(f"  Retrieved {len(docs)} documents")
        for doc in docs:
            print(f"  - {doc.metadata.get('source')}")


if __name__ == "__main__":
    print("=" * 60)
    print("Retriever Examples")
    print("=" * 60)
    
    try:
        # Setup
        vector_store = setup_vector_store()
        
        # Basic retriever examples
        example_basic_vector_store_retriever(vector_store)
        example_retriever_with_scores(vector_store)
        example_mmr_retriever(vector_store)
        example_score_threshold_retriever(vector_store)
        example_dynamic_k_adjustment(vector_store)
        example_batch_retrieval(vector_store)
        
        # Advanced retriever examples (if available)
        if ADVANCED_RETRIEVERS_AVAILABLE:
            example_multi_query_retriever(vector_store)
            example_contextual_compression_retriever(vector_store)
        else:
            print("\n⚠️  Skipping advanced retriever examples (dependencies not installed)")
        
        print("\n" + "=" * 60)
        print("✅ All examples completed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

# Made with Bob
