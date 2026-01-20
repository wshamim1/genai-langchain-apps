"""
Document Q&A Chatbot - Streamlit Web Interface

A beautiful web interface for the RAG-based Document Q&A Chatbot.

Features:
- Upload documents (PDF, CSV, TXT)
- Interactive chat interface
- View sources and confidence
- Conversation history
- Document management

Run with: streamlit run applications/document_qa_streamlit.py
"""

import os
import sys
import streamlit as st
from typing import List
from datetime import datetime
import tempfile

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.chat_models import OpenAIChatModel
from src.document_loaders import PDFLoader, CSVLoader
from src.vector_stores import ChromaVectorStore
from src.retrievers import VectorStoreRetriever
from src.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings


# Page configuration
st.set_page_config(
    page_title="Document Q&A Chatbot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        background-color: #f5f5f5;
        border-left: 4px solid #4caf50;
    }
    .source-box {
        background-color: #fff3e0;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin-top: 0.5rem;
        font-size: 0.9rem;
    }
    .confidence-high {
        color: #4caf50;
        font-weight: bold;
    }
    .confidence-medium {
        color: #ff9800;
        font-weight: bold;
    }
    .confidence-low {
        color: #f44336;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = None
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'documents_loaded' not in st.session_state:
        st.session_state.documents_loaded = False
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []


def create_chatbot():
    """Create and initialize the chatbot."""
    try:
        # Check for API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            st.error("⚠️ OPENAI_API_KEY not found in environment variables!")
            st.info("Please set your OpenAI API key in the .env file or as an environment variable.")
            return None
        
        # Initialize components
        llm = OpenAIChatModel(
            model_name="gpt-3.5-turbo",
            temperature=0.0
        )
        
        # Initialize embeddings
        embeddings = OpenAIEmbeddings()
        
        vector_store = ChromaVectorStore(
            collection_name="streamlit_qa",
            persist_directory="databases/streamlit_qa",
            embedding_function=embeddings
        )
        
        # Setup output parser
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
        output_parser = StructuredOutputParser(response_schemas=response_schemas)
        
        return {
            'llm': llm,
            'vector_store': vector_store,
            'output_parser': output_parser,
            'retriever': None
        }
    
    except Exception as e:
        st.error(f"Error initializing chatbot: {str(e)}")
        return None


def load_document(file, file_type):
    """Load a single document."""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_type}") as tmp_file:
            tmp_file.write(file.getvalue())
            tmp_path = tmp_file.name
        
        # Load based on file type
        if file_type == 'pdf':
            loader = PDFLoader(file_path=tmp_path)
            documents = loader.load()
        elif file_type == 'csv':
            loader = CSVLoader(file_path=tmp_path)
            documents = loader.load()
        elif file_type == 'txt':
            with open(tmp_path, 'r', encoding='utf-8') as f:
                content = f.read()
            documents = [Document(page_content=content, metadata={"source": file.name})]
        else:
            documents = []
        
        # Clean up temp file
        os.unlink(tmp_path)
        
        return documents
    
    except Exception as e:
        st.error(f"Error loading {file.name}: {str(e)}")
        return []


def process_uploaded_files(uploaded_files):
    """Process uploaded files and add to vector store."""
    if not uploaded_files:
        return False
    
    with st.spinner("📚 Loading and indexing documents..."):
        all_documents = []
        
        for file in uploaded_files:
            file_type = file.name.split('.')[-1].lower()
            
            if file_type in ['pdf', 'csv', 'txt']:
                documents = load_document(file, file_type)
                all_documents.extend(documents)
                st.success(f"✓ Loaded {file.name} ({len(documents)} chunks)")
            else:
                st.warning(f"⚠️ Unsupported file type: {file.name}")
        
        if all_documents:
            # Add to vector store
            st.session_state.chatbot['vector_store'].add_documents(all_documents)
            
            # Create retriever
            st.session_state.chatbot['retriever'] = VectorStoreRetriever.from_vector_store(
                st.session_state.chatbot['vector_store'],
                k=4
            )
            
            st.session_state.documents_loaded = True
            st.session_state.uploaded_files = [f.name for f in uploaded_files]
            
            st.success(f"✅ Successfully indexed {len(all_documents)} document chunks!")
            return True
    
    return False


def ask_question(question: str):
    """Ask a question and get an answer."""
    if not st.session_state.documents_loaded:
        return {
            "answer": "Please upload documents first.",
            "sources": "N/A",
            "confidence": "N/A"
        }
    
    try:
        # Retrieve relevant documents
        retriever = st.session_state.chatbot['retriever']
        relevant_docs = retriever.get_relevant_documents(question)
        
        # Build context
        context = "\n\n".join([
            f"[Source {i+1}]: {doc.page_content}"
            for i, doc in enumerate(relevant_docs)
        ])
        
        # Create prompt
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
        
        format_instructions = st.session_state.chatbot['output_parser'].get_format_instructions()
        
        messages = [
            SystemMessage(content=system_prompt.format(
                context=context,
                format_instructions=format_instructions
            )),
            HumanMessage(content=question)
        ]
        
        # Generate response
        response = st.session_state.chatbot['llm'].invoke(messages)
        
        # Parse output
        try:
            parsed_response = st.session_state.chatbot['output_parser'].parse(response.content)
        except:
            parsed_response = {
                "answer": response.content,
                "sources": "Unable to parse sources",
                "confidence": "unknown"
            }
        
        return parsed_response
    
    except Exception as e:
        return {
            "answer": f"Error: {str(e)}",
            "sources": "N/A",
            "confidence": "N/A"
        }


def main():
    """Main Streamlit application."""
    
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.markdown('<div class="main-header">📚 Document Q&A Chatbot</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("📁 Document Management")
        
        # Initialize chatbot if not already done
        if st.session_state.chatbot is None:
            with st.spinner("Initializing chatbot..."):
                st.session_state.chatbot = create_chatbot()
                if st.session_state.chatbot:
                    st.success("✅ Chatbot initialized!")
        
        # File uploader
        st.subheader("Upload Documents")
        uploaded_files = st.file_uploader(
            "Choose files (PDF, CSV, TXT)",
            type=['pdf', 'csv', 'txt'],
            accept_multiple_files=True,
            help="Upload documents to ask questions about"
        )
        
        if uploaded_files and st.button("📤 Load Documents", type="primary"):
            if st.session_state.chatbot:
                process_uploaded_files(uploaded_files)
        
        # Show loaded documents
        if st.session_state.documents_loaded:
            st.success("✅ Documents loaded!")
            with st.expander("📄 Loaded Files"):
                for filename in st.session_state.uploaded_files:
                    st.text(f"• {filename}")
        
        # Clear conversation
        st.markdown("---")
        if st.button("🗑️ Clear Conversation"):
            st.session_state.messages = []
            st.rerun()
        
        # Settings
        st.markdown("---")
        st.subheader("⚙️ Settings")
        st.info("Model: GPT-3.5-turbo\nTemperature: 0.0\nRetrieval: Top 4 chunks")
    
    # Main chat interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Chat")
        
        # Display chat messages
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.messages:
                if message["role"] == "user":
                    st.markdown(f"""
                    <div class="chat-message user-message">
                        <strong>You:</strong> {message["content"]}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    confidence_class = f"confidence-{message.get('confidence', 'unknown')}"
                    st.markdown(f"""
                    <div class="chat-message assistant-message">
                        <strong>🤖 Assistant:</strong><br>
                        {message["content"]}
                        <div class="source-box">
                            📄 <strong>Sources:</strong> {message.get("sources", "N/A")}<br>
                            📊 <strong>Confidence:</strong> <span class="{confidence_class}">{message.get("confidence", "N/A")}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Chat input
        if st.session_state.chatbot and st.session_state.documents_loaded:
            question = st.chat_input("Ask a question about your documents...")
            
            if question:
                # Add user message
                st.session_state.messages.append({
                    "role": "user",
                    "content": question
                })
                
                # Get answer
                with st.spinner("🤔 Thinking..."):
                    result = ask_question(question)
                
                # Add assistant message
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["sources"],
                    "confidence": result["confidence"]
                })
                
                st.rerun()
        else:
            st.info("👈 Please upload documents in the sidebar to start chatting!")
    
    with col2:
        st.header("ℹ️ Information")
        
        st.markdown("""
        ### How to use:
        
        1. **Upload Documents**
           - Click "Browse files" in sidebar
           - Select PDF, CSV, or TXT files
           - Click "Load Documents"
        
        2. **Ask Questions**
           - Type your question in the chat
           - Get answers with sources
           - View confidence levels
        
        3. **Tips**
           - Ask specific questions
           - Reference document content
           - Check sources for accuracy
        
        ### Features:
        - 📚 Multi-document support
        - 🔍 Semantic search
        - 📊 Confidence scoring
        - 📄 Source citations
        - 💬 Conversation history
        """)
        
        # Example questions
        if st.session_state.documents_loaded:
            st.markdown("---")
            st.subheader("💡 Example Questions")
            example_questions = [
                "What is this document about?",
                "Can you summarize the main points?",
                "What are the key topics covered?",
                "Tell me about [specific topic]"
            ]
            for q in example_questions:
                if st.button(q, key=f"example_{q}"):
                    st.session_state.messages.append({"role": "user", "content": q})
                    with st.spinner("🤔 Thinking..."):
                        result = ask_question(q)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": result["sources"],
                        "confidence": result["confidence"]
                    })
                    st.rerun()


if __name__ == "__main__":
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    main()

# Made with Bob
