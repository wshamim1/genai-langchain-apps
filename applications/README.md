# Real-World Applications

This folder contains complete, production-ready applications built using the LangChain framework components.

## Applications

### 1. Document Q&A Chatbot - CLI (`document_qa_chatbot.py`)

A RAG (Retrieval-Augmented Generation) based chatbot that answers questions about your documents.

**Features:**
- Load multiple document types (PDF, CSV, TXT)
- Create vector embeddings for semantic search
- Retrieve relevant context for questions
- Generate accurate answers with source citations
- Structured output with confidence levels
- Interactive chat interface

**Components Used:**
- Chat Models (OpenAI)
- Document Loaders (PDF, CSV)
- Vector Stores (Chroma)
- Retrievers (Vector Store Retriever)
- Output Parsers (Structured Parser)

**Usage:**
```bash
# Make sure OPENAI_API_KEY is set in .env
python applications/document_qa_chatbot.py
```

---

### 1b. Document Q&A Chatbot - Web UI (`document_qa_streamlit.py`)

A beautiful Streamlit web interface for the Document Q&A Chatbot.

**Features:**
- 🎨 Modern, intuitive web interface
- 📤 Drag-and-drop file upload
- 💬 Interactive chat interface
- 📊 Visual confidence indicators
- 📄 Source citations with highlighting
- 🗂️ Document management
- 💡 Example questions
- 🎯 Real-time responses

**Components Used:**
- All components from CLI version
- Streamlit for web interface
- Session state management
- File upload handling

**Usage:**
```bash
# Install Streamlit (included in requirements.txt)
pip install streamlit

# Run the web app
streamlit run applications/document_qa_streamlit.py

# Opens in browser at http://localhost:8501
```

**Features:**
- **Upload Documents**: Drag and drop PDF, CSV, or TXT files
- **Interactive Chat**: Ask questions in natural language
- **Source Citations**: See which documents were used
- **Confidence Scores**: High/Medium/Low confidence indicators
- **Conversation History**: Keep track of your questions
- **Example Questions**: Quick-start with suggested questions

**Example Questions:**
- "What is this document about?"
- "Can you summarize the main points?"
- "What are the key topics covered?"

---

### 2. Research Assistant - CLI (`research_assistant.py`)

An intelligent agent-based research assistant that can perform complex multi-step tasks.

**Features:**
- Search for information
- Perform calculations
- Read and write files
- Analyze data
- Generate research reports
- Multi-step task execution
- Interactive and demo modes

**Components Used:**
- Chat Models (OpenAI)
- Agents (ReAct Agent)
- Tools (Calculator, Search, File Operations)
- Agent Executor

**Usage:**
```bash
# Demo mode (runs example tasks)
python applications/research_assistant.py --mode demo

# Interactive mode (chat interface)
python applications/research_assistant.py --mode interactive
```

**Example Tasks:**
- "Calculate compound interest and save to file"
- "Search for Python info and create summary"
- "List all files and create inventory"
- "Read data.txt and analyze content"

---

### 2b. Research Assistant - Web UI (`research_assistant_streamlit.py`)

A beautiful Streamlit web interface for the Research Assistant with full agent visualization.

**Features:**
- 🔬 Interactive research task interface
- 🧠 Real-time agent reasoning visualization
- 🔧 See which tools the agent uses
- 💭 Watch the agent think step-by-step
- 📂 Workspace file browser
- 📜 Complete task history
- 📊 Statistics dashboard
- 📝 Automatic report generation

**Components Used:**
- All components from CLI version
- Streamlit for web interface
- Agent step visualization
- File management UI

**Usage:**
```bash
# Run the web app
streamlit run applications/research_assistant_streamlit.py

# Opens in browser at http://localhost:8501
```

**Features:**
- **Research Tasks**: Execute complex multi-step tasks
- **Agent Reasoning**: See the agent's thought process
- **Tool Execution**: Watch tools being used in real-time
- **Workspace Browser**: View and manage generated files
- **Task History**: Review all completed tasks
- **Summary Reports**: Generate comprehensive reports

**Example Tasks:**
- "Calculate 15% of 2,450 and save to calculation.txt"
- "Search for quantum computing and create a summary file"
- "List all workspace files and create an index"
- "Research Python history, calculate its age, and save a report"

**Agent Visualization:**
The UI shows:
- 💭 **Thought**: What the agent is thinking
- 🔧 **Action**: Which tool it's using
- 👁️ **Observation**: What it learned
- 📋 **Final Answer**: The complete result

---

## Setup

### Prerequisites

1. **Install Dependencies:**
```bash
pip install -r requirements.txt
```

2. **Set API Keys:**
Create a `.env` file in the project root:
```
OPENAI_API_KEY=your_openai_api_key_here
```

3. **Prepare Data:**
- For Document Q&A: Add documents to `data/` folder
- For Research Assistant: Creates `data/research/` automatically

### Running Applications

Both applications can run without API keys in limited mode (showing structure and examples), but require OpenAI API key for full functionality.

---

## Architecture

### Document Q&A Chatbot Architecture

```
User Question
     ↓
[Document Loader] → Load PDFs, CSVs, TXT
     ↓
[Text Splitter] → Split into chunks
     ↓
[Embeddings] → Create vector embeddings
     ↓
[Vector Store] → Store in Chroma
     ↓
[Retriever] → Find relevant chunks
     ↓
[LLM + Context] → Generate answer
     ↓
[Output Parser] → Structure response
     ↓
Structured Answer (answer, sources, confidence)
```

### Research Assistant Architecture

```
User Task
     ↓
[ReAct Agent] → Reason about task
     ↓
[Agent Executor] → Execute agent loop
     ↓
┌─────────────────────────────┐
│  Agent Decision Loop:       │
│  1. Think about what to do  │
│  2. Choose a tool           │
│  3. Execute tool            │
│  4. Observe result          │
│  5. Repeat or finish        │
└─────────────────────────────┘
     ↓
[Tools Available]
  - Calculator (math operations)
  - Search (find information)
  - File Read (read files)
  - File Write (save results)
  - File List (directory listing)
     ↓
Final Result + Report
```

---

## Customization

### Document Q&A Chatbot

**Change LLM Model:**
```python
chatbot = DocumentQAChatbot(
    model_name="gpt-4",  # Use GPT-4 for better quality
    temperature=0.0
)
```

**Adjust Retrieval:**
```python
# In load_documents method, modify:
self.retriever = VectorStoreRetriever.from_vector_store(
    self.vector_store,
    k=6  # Retrieve more chunks
)
```

**Custom Output Format:**
```python
# Modify response_schemas in __init__
response_schemas = [
    ResponseSchema(name="answer", description="..."),
    ResponseSchema(name="sources", description="..."),
    ResponseSchema(name="confidence", description="..."),
    ResponseSchema(name="follow_up_questions", description="...")  # Add new field
]
```

### Research Assistant

**Add Custom Tools:**
```python
from src.tools import BaseTool

class CustomTool(BaseTool):
    name = "custom_tool"
    description = "Description of what it does"
    
    def _run(self, input: str):
        # Your custom logic
        return result

# Add to assistant
assistant.tools.append(CustomTool())
```

**Change Agent Type:**
```python
from src.agents import ToolCallingAgent

# Use ToolCallingAgent instead of ReActAgent
self.agent = ToolCallingAgent.create(
    llm=self.llm,
    tools=self.tools
)
```

---

## Use Cases

### Document Q&A Chatbot

1. **Customer Support Knowledge Base**
   - Load company documentation
   - Answer customer questions
   - Provide source citations

2. **Legal Document Analysis**
   - Load contracts and legal docs
   - Answer specific questions
   - Find relevant clauses

3. **Research Paper Assistant**
   - Load academic papers
   - Answer research questions
   - Find specific information

4. **Technical Documentation Helper**
   - Load API docs, manuals
   - Answer how-to questions
   - Find code examples

### Research Assistant

1. **Data Analysis**
   - Load CSV files
   - Perform calculations
   - Generate reports

2. **Content Research**
   - Search for information
   - Summarize findings
   - Create documentation

3. **File Management**
   - Organize files
   - Create inventories
   - Analyze content

4. **Automated Reporting**
   - Gather data
   - Perform analysis
   - Generate reports

---

## Troubleshooting

### Common Issues

**1. "OPENAI_API_KEY not found"**
- Solution: Create `.env` file with your API key

**2. "No documents found"**
- Solution: Add documents to `data/` folder

**3. "Import errors"**
- Solution: Run `pip install -r requirements.txt`

**4. "Chroma database errors"**
- Solution: Delete `databases/` folder and restart

**5. "Agent not responding"**
- Solution: Check API key, increase max_iterations

---

## Performance Tips

1. **Use GPT-3.5-turbo for speed** (default)
2. **Use GPT-4 for quality** (slower, more expensive)
3. **Adjust chunk size** for better retrieval
4. **Increase k value** for more context
5. **Use temperature=0** for consistent results
6. **Cache embeddings** to avoid recomputation

---

## Future Enhancements

### Document Q&A Chatbot
- [ ] Add conversation memory
- [ ] Support more document types
- [ ] Add document upload interface
- [ ] Implement streaming responses
- [ ] Add multi-language support

### Research Assistant
- [ ] Add web scraping tool
- [ ] Implement database queries
- [ ] Add visualization tools
- [ ] Create scheduled tasks
- [ ] Add collaboration features

---

## Contributing

To add new applications:

1. Create new Python file in `applications/`
2. Import required components from `src/`
3. Implement application logic
4. Add documentation to this README
5. Include usage examples

---

## License

These applications are part of the LangChain framework implementation and follow the same license.

---

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review component documentation in `src/`
3. Check example code in `demo/`
4. Review LangChain documentation

---

**Built with ❤️ using LangChain Framework**