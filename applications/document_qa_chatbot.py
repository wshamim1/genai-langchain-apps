"""
Document Q&A Chatbot Application

A real-world RAG (Retrieval-Augmented Generation) chatbot that:
- Loads documents (PDF, CSV, TXT)
- Creates vector embeddings
- Retrieves relevant context
- Answers questions using LLM with context
- Provides structured responses

This application demonstrates the integration of:
- Chat Models (OpenAI)
- Document Loaders (PDF, CSV)
- Vector Stores (Chroma)
- Retrievers (Vector Store Retriever)
- Output Parsers (Structured Parser)
"""

import os
import sys
from typing import List, Optional
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.chat_models import OpenAIChatModel
from src.document_loaders import PDFLoader, CSVLoader
from src.vector_stores import ChromaVectorStore
from src.retrievers import VectorStoreRetriever
from src.output_parsers import StructuredOutputParser
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.schema import ResponseSchema

# Load environment variables
load_dotenv()


class DocumentQAChatbot:
    """RAG-based Document Q&A Chatbot.
    
    This chatbot can answer questions about documents by:
    1. Loading and indexing documents
    2. Retrieving relevant context
    3. Generating answers using LLM
    """
    
    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.0,
        collection_name: str = "document_qa",
        persist_directory: str = "databases/document_qa"
    ):
        """Initialize the chatbot.
        
        Args:
            model_name: LLM model to use
            temperature: Temperature for generation
            collection_name: Chroma collection name
            persist_directory: Directory to persist vector store
        """
        # Check for API key
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        # Initialize components
        self.llm = OpenAIChatModel(
            model_name=model_name,
            temperature=temperature
        )
        
        self.vector_store = ChromaVectorStore(
            collection_name=collection_name,
            persist_directory=persist_directory
        )
        
        self.retriever: Optional[VectorStoreRetriever] = None
        self.documents_loaded = False
        
        # Setup output parser for structured responses
        response_schemas = [
            ResponseSchema(
                name="answer",
                description="The answer to the user's question based on the context"
            ),
            ResponseSchema(
                name="sources",
                description="The sources or document sections used to answer the question"
            ),
            ResponseSchema(
                name="confidence",
                description="Confidence level: high, medium, or low"
            )
        ]
        self.output_parser = StructuredOutputParser(response_schemas=response_schemas)
        
        print("✅ Document Q&A Chatbot initialized!")
    
    def load_documents(self, document_paths: List[str]):
        """Load documents into the vector store.
        
        Args:
            document_paths: List of paths to documents (PDF, CSV, TXT)
        """
        print(f"\n📚 Loading {len(document_paths)} document(s)...")
        
        all_documents = []
        
        for path in document_paths:
            print(f"  Loading: {path}")
            
            if path.endswith('.pdf'):
                loader = PDFLoader(file_path=path)
                documents = loader.load()
            elif path.endswith('.csv'):
                loader = CSVLoader(file_path=path)
                documents = loader.load()
            elif path.endswith('.txt'):
                # Simple text file loading
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                from langchain.schema import Document
                documents = [Document(page_content=content, metadata={"source": path})]
            else:
                print(f"  ⚠️  Unsupported file type: {path}")
                continue
            
            all_documents.extend(documents)
            print(f"    ✓ Loaded {len(documents)} chunks")
        
        # Add documents to vector store
        print(f"\n🔄 Creating embeddings and indexing...")
        self.vector_store.add_documents(all_documents)
        
        # Create retriever
        self.retriever = VectorStoreRetriever.from_vector_store(
            self.vector_store,
            k=4  # Retrieve top 4 most relevant chunks
        )
        
        self.documents_loaded = True
        print(f"✅ Indexed {len(all_documents)} document chunks!\n")
    
    def ask(self, question: str, verbose: bool = False) -> dict:
        """Ask a question about the documents.
        
        Args:
            question: The question to ask
            verbose: Whether to print detailed information
            
        Returns:
            Dictionary with answer, sources, and confidence
        """
        if not self.documents_loaded:
            return {
                "answer": "No documents loaded. Please load documents first.",
                "sources": "N/A",
                "confidence": "N/A"
            }
        
        if verbose:
            print(f"\n❓ Question: {question}")
            print(f"🔍 Retrieving relevant context...")
        
        # Retrieve relevant documents
        relevant_docs = self.retriever.get_relevant_documents(question)
        
        if verbose:
            print(f"   Found {len(relevant_docs)} relevant chunks")
        
        # Build context from retrieved documents
        context = "\n\n".join([
            f"[Source {i+1}]: {doc.page_content}"
            for i, doc in enumerate(relevant_docs)
        ])
        
        # Create prompt with context
        system_prompt = """You are a helpful assistant that answers questions based on the provided context.

Instructions:
1. Answer the question using ONLY the information from the context
2. If the context doesn't contain enough information, say so
3. Cite which sources you used
4. Provide a confidence level (high/medium/low)

Context:
{context}

{format_instructions}
"""
        
        format_instructions = self.output_parser.get_format_instructions()
        
        messages = [
            SystemMessage(content=system_prompt.format(
                context=context,
                format_instructions=format_instructions
            )),
            HumanMessage(content=question)
        ]
        
        if verbose:
            print(f"💭 Generating answer...")
        
        # Generate response
        response = self.llm.invoke(messages)
        
        # Parse structured output
        try:
            parsed_response = self.output_parser.parse(response.content)
        except Exception as e:
            # Fallback if parsing fails
            parsed_response = {
                "answer": response.content,
                "sources": "Unable to parse sources",
                "confidence": "unknown"
            }
        
        if verbose:
            print(f"✅ Answer generated!\n")
        
        return parsed_response
    
    def chat(self):
        """Start an interactive chat session."""
        print("\n" + "="*80)
        print("  📚 Document Q&A Chatbot")
        print("="*80)
        print("\nCommands:")
        print("  - Type your question to get an answer")
        print("  - Type 'quit' or 'exit' to end the session")
        print("  - Type 'help' for more information")
        print("\n" + "="*80 + "\n")
        
        if not self.documents_loaded:
            print("⚠️  No documents loaded. Please load documents first using load_documents()")
            return
        
        while True:
            try:
                # Get user input
                question = input("You: ").strip()
                
                if not question:
                    continue
                
                # Handle commands
                if question.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Goodbye!\n")
                    break
                
                if question.lower() == 'help':
                    print("\n📖 Help:")
                    print("  - Ask any question about the loaded documents")
                    print("  - The chatbot will retrieve relevant context and answer")
                    print("  - Answers include sources and confidence levels\n")
                    continue
                
                # Get answer
                result = self.ask(question, verbose=False)
                
                # Display response
                print(f"\n🤖 Assistant:")
                print(f"   {result['answer']}")
                print(f"\n   📄 Sources: {result['sources']}")
                print(f"   📊 Confidence: {result['confidence']}\n")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!\n")
                break
            except Exception as e:
                print(f"\n❌ Error: {str(e)}\n")


def main():
    """Run the Document Q&A Chatbot application."""
    print("\n" + "="*80)
    print("  🚀 Document Q&A Chatbot Application")
    print("="*80)
    
    try:
        # Initialize chatbot
        chatbot = DocumentQAChatbot(
            model_name="gpt-3.5-turbo",
            temperature=0.0,
            collection_name="document_qa_demo",
            persist_directory="databases/document_qa_demo"
        )
        
        # Load sample documents
        document_paths = [
            "data/sample_document.txt",
            "data/sample.pdf",
            "data/sample_data.csv"
        ]
        
        # Filter to only existing files
        existing_docs = [path for path in document_paths if os.path.exists(path)]
        
        if not existing_docs:
            print("\n⚠️  No sample documents found in data/ directory")
            print("Please add some documents to test the chatbot.\n")
            return
        
        chatbot.load_documents(existing_docs)
        
        # Example questions
        print("="*80)
        print("  📝 Example Questions")
        print("="*80 + "\n")
        
        example_questions = [
            "What is this document about?",
            "Can you summarize the main points?",
            "What are the key topics covered?"
        ]
        
        for i, question in enumerate(example_questions, 1):
            print(f"{i}. {question}")
            result = chatbot.ask(question, verbose=True)
            
            print(f"📋 Answer: {result['answer']}")
            print(f"📄 Sources: {result['sources']}")
            print(f"📊 Confidence: {result['confidence']}")
            print("\n" + "-"*80 + "\n")
        
        # Start interactive chat
        print("\n" + "="*80)
        print("  💬 Starting Interactive Chat")
        print("="*80)
        chatbot.chat()
        
    except ValueError as e:
        print(f"\n❌ Configuration Error: {str(e)}")
        print("Please set OPENAI_API_KEY in your .env file\n")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

# Made with Bob
