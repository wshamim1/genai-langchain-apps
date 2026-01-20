# LangChain Framework - Complete Implementation

A comprehensive, production-ready LangChain framework implementation with all core components, real-world applications, and beautiful Streamlit web interfaces.

## 🌟 Features

### Core Components
- 🤖 **Chat Models** - OpenAI, Anthropic, Google with unified interface
- 📄 **Document Loaders** - PDF, CSV with text splitting
- 🗄️ **Vector Stores** - Chroma with persistent storage
- 🔍 **Retrievers** - Vector Store, Multi-Query, Contextual Compression
- 📊 **Output Parsers** - Pydantic, JSON, Structured (custom implementation)
- 🔧 **Tools** - Calculator, Search, File Operations, API requests
- 📦 **Toolkits** - File System toolkit
- 🤖 **Agents** - ReAct and Tool-Calling agents with executor

### Applications
- 📚 **Document Q&A Chatbot** - RAG-based Q&A with CLI and Web UI
- 🔬 **Research Assistant** - Agent-powered research with CLI and Web UI

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd genai-langchain

# Install dependencies
pip install -r requirements.txt
```

### 2. Set Up API Key

```bash
# Option A: Environment variable
export OPENAI_API_KEY='sk-your-api-key-here'

# Option B: Create .env file
echo "OPENAI_API_KEY=sk-your-api-key-here" > .env
```

### 3. Run Applications

#### Document Q&A Chatbot (Web)
```bash
streamlit run applications/document_qa_streamlit.py
# Opens at http://localhost:8502
```

#### Research Assistant (Web)
```bash
streamlit run applications/research_assistant_streamlit.py
# Opens at http://localhost:8501
```

#### CLI Versions
```bash
# Document Q&A
python applications/document_qa_chatbot.py

# Research Assistant
python applications/research_assistant.py --mode interactive
```

## 📁 Project Structure

```
genai-langchain/
├── src/                          # Core framework components
│   ├── chat_models/              # Chat model implementations
│   │   ├── base.py              # Base chat model interface
│   │   ├── openai_chat.py       # OpenAI implementation
│   │   ├── anthropic_chat.py    # Anthropic implementation
│   │   └── google_chat.py       # Google implementation
│   ├── document_loaders/         # Document loading
│   │   ├── base.py              # Base loader interface
│   │   ├── pdf_loader.py        # PDF loader
│   │   └── csv_loader.py        # CSV loader
│   ├── vector_stores/            # Vector storage
│   │   ├── base.py              # Base vector store
│   │   └── chroma_store.py      # Chroma implementation
│   ├── retrievers/               # Document retrieval
│   │   ├── base.py              # Base retriever
│   │   ├── vector_store_retriever.py
│   │   ├── multi_query_retriever.py
│   │   └── contextual_compression_retriever.py
│   ├── output_parsers/           # Output parsing
│   │   ├── base.py              # Base parser
│   │   ├── pydantic_parser.py   # Pydantic parser
│   │   ├── json_parser.py       # JSON parser
│   │   └── structured_parser.py # Structured parser (custom)
│   ├── tools/                    # Agent tools
│   │   ├── base.py              # Base tool interface
│   │   ├── calculator.py        # Calculator tool
│   │   ├── search.py            # Search tool
│   │   ├── file_operations.py   # File tools
│   │   └── api_request.py       # API tool
│   ├── toolkits/                 # Tool collections
│   │   ├── base.py              # Base toolkit
│   │   └── file_system.py       # File system toolkit
│   └── agents/                   # Agent implementations
│       ├── base.py              # Base agent
│       ├── react_agent.py       # ReAct agent
│       ├── tool_calling_agent.py # Tool-calling agent
│       └── agent_executor.py    # Agent executor
├── applications/                 # Real-world applications
│   ├── document_qa_chatbot.py   # CLI Document Q&A
│   ├── document_qa_streamlit.py # Web Document Q&A
│   ├── research_assistant.py    # CLI Research Assistant
│   ├── research_assistant_streamlit.py # Web Research Assistant
│   └── README.md                # Applications documentation
├── demo/                         # Example code
│   ├── chat_models/             # Chat model examples
│   ├── document_loaders/        # Loader examples
│   ├── vector_stores/           # Vector store examples
│   ├── output_parsers/          # Parser examples
│   ├── retrievers/              # Retriever examples
│   ├── tools/                   # Tool examples
│   └── agents/                  # Agent examples
├── data/                         # Data files
├── databases/                    # Vector databases
├── requirements.txt              # Python dependencies
├── .env.example                 # Example environment file
└── README.md                    # This file
```

## 💻 Usage Examples

### Chat Models

```python
from src.chat_models import OpenAIChatModel

# Initialize model
model = OpenAIChatModel(
    model_name="gpt-3.5-turbo",
    temperature=0.7
)

# Simple chat
response = model.chat("What is the capital of France?")
print(response.content)
```

### Document Q&A

```python
from src.chat_models import OpenAIChatModel
from src.document_loaders import PDFLoader
from src.vector_stores import ChromaVectorStore
from src.retrievers import VectorStoreRetriever
from langchain_openai import OpenAIEmbeddings

# Load documents
loader = PDFLoader("document.pdf")
documents = loader.load()

# Create vector store
embeddings = OpenAIEmbeddings()
vector_store = ChromaVectorStore(
    collection_name="my_docs",
    embedding_function=embeddings
)
vector_store.add_documents(documents)

# Create retriever
retriever = VectorStoreRetriever.from_vector_store(vector_store, k=4)

# Ask questions
relevant_docs = retriever.get_relevant_documents("What is this about?")
```

### Agents

```python
from src.chat_models import OpenAIChatModel
from src.tools import CalculatorTool, SearchTool
from src.agents import ReActAgent, AgentExecutor

# Initialize components
llm = OpenAIChatModel(model_name="gpt-3.5-turbo")
tools = [CalculatorTool.create(), SearchTool.create()]

# Create agent
agent = ReActAgent.create(llm=llm, tools=tools)
executor = AgentExecutor.create(agent=agent, tools=tools)

# Execute task
result = executor.run("Calculate 15% of 2,450")
print(result['output'])
```

## 🎨 Streamlit Applications

### Document Q&A Chatbot

**Features:**
- 📤 Upload PDF, CSV, TXT files
- 💬 Interactive chat interface
- 📊 Confidence scoring
- 📄 Source citations
- 🗂️ Document management

**Usage:**
1. Run: `streamlit run applications/document_qa_streamlit.py`
2. Upload documents
3. Ask questions
4. Get answers with sources!

### Research Assistant

**Features:**
- 🔬 Execute complex research tasks
- 🧠 Real-time agent reasoning visualization
- 💭 See thought process
- 🔧 Watch tool execution
- 📂 Workspace file browser
- 📜 Complete task history
- 📝 Generate summary reports

**Usage:**
1. Run: `streamlit run applications/research_assistant_streamlit.py`
2. Enter a research task
3. Watch the agent work
4. Get results with full reasoning!

**Example Tasks:**
- "Calculate 15% of 2,450 and save to file"
- "Search for quantum computing and create summary"
- "List all files and create an index"

## 🔧 Configuration

### Environment Variables

```bash
# Required
OPENAI_API_KEY=sk-...

# Optional (for other providers)
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
```

### Model Parameters

All chat models support:
- `model_name` - Specific model to use
- `temperature` - Sampling temperature (0.0-1.0)
- `max_tokens` - Maximum tokens to generate
- `api_key` - API key (overrides environment)

## 📚 Documentation

- **applications/README.md** - Detailed application documentation
- **demo/** - Comprehensive examples for each component
- Each component has inline documentation and type hints

## 🛠️ Development

### Adding New Components

1. **New Chat Model:**
   - Inherit from `BaseChatModel`
   - Implement `generate()` and `agenerate()`
   - Add to `src/chat_models/__init__.py`

2. **New Tool:**
   - Inherit from `BaseTool`
   - Implement `_run()` method
   - Add to `src/tools/__init__.py`

3. **New Retriever:**
   - Inherit from `BaseRetriever`
   - Implement `get_relevant_documents()`
   - Add to `src/retrievers/__init__.py`

### Running Examples

```bash
# Chat models
python demo/chat_models/providers/openai/examples.py

# Document loaders
python demo/document_loaders/examples/pdf_loader_example.py

# Vector stores
python demo/vector_stores/examples/chroma_example.py

# Agents
python demo/agents/examples/agent_examples.py
```

## 🎯 Key Features

### Custom Implementations

- **StructuredOutputParser**: Custom implementation without LangChain dependencies
- **ResponseSchema**: Pydantic-based schema definition
- **Agent Visualization**: Real-time reasoning display in Streamlit

### Production Ready

- ✅ Error handling
- ✅ Type hints throughout
- ✅ Async support
- ✅ Streaming responses
- ✅ Persistent storage
- ✅ Comprehensive examples

## 🐛 Troubleshooting

### "OPENAI_API_KEY not found"
Set your API key:
```bash
export OPENAI_API_KEY='sk-your-key-here'
```

### "Module not found"
Install dependencies:
```bash
pip install -r requirements.txt
```

### Torch Warning in Streamlit
The `torch._classes` warning is harmless and can be ignored. It's just Streamlit's file watcher having issues with PyTorch.

### Port Already in Use
```bash
# Kill existing Streamlit
pkill -f streamlit

# Or use different port
streamlit run app.py --server.port 8503
```

## 📦 Dependencies

Main dependencies:
- `langchain-core` - Core LangChain functionality
- `langchain-openai` - OpenAI integration
- `langchain-anthropic` - Anthropic integration
- `langchain-google-genai` - Google integration
- `langchain-community` - Community components
- `chromadb` - Vector database
- `streamlit` - Web interface
- `pypdf` - PDF processing
- `python-dotenv` - Environment management

See `requirements.txt` for complete list.

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests/examples
5. Submit a pull request

## 📄 License

MIT License - feel free to use in your projects!

## 🙏 Acknowledgments

Inspired by [LangChain](https://github.com/langchain-ai/langchain) and designed to provide a complete, production-ready implementation with beautiful web interfaces.

## 📞 Support

For issues or questions:
- Check the documentation in `applications/README.md`
- Review examples in `demo/`
- Check component-specific README files

---

**Built with ❤️ for the AI community**

🚀 **Ready to use!** Start with the Streamlit apps for the best experience.