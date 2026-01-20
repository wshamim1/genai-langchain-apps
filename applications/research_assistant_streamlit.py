"""
Research Assistant - Streamlit Web Interface

An intelligent research assistant with agent capabilities that can:
- Search for information
- Perform calculations
- Read and write files
- Analyze data
- Generate reports

Features:
- Interactive chat interface with agent reasoning
- Real-time tool execution visualization
- Task history and report generation
- File management
- Workspace browser

Run with: streamlit run applications/research_assistant_streamlit.py
"""

import os
import sys
import streamlit as st
from typing import List, Dict
from datetime import datetime
import json

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.chat_models import OpenAIChatModel
from src.tools import CalculatorTool, SearchTool, FileReadTool, FileWriteTool, FileListTool
from src.agents import ReActAgent, AgentExecutor


# Page configuration
st.set_page_config(
    page_title="Research Assistant",
    page_icon="🔬",
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
        padding: 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .task-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    
    .tool-execution {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 8px;
        border-left: 3px solid #2196F3;
        margin: 0.5rem 0;
        font-family: 'Courier New', monospace;
        font-size: 0.9rem;
    }
    
    .thought-process {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 3px solid #ffc107;
        margin: 0.5rem 0;
        font-style: italic;
    }
    
    .final-answer {
        background-color: #d4edda;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
        font-weight: 500;
    }
    
    .step-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: bold;
        margin-right: 0.5rem;
    }
    
    .step-thought {
        background-color: #ffc107;
        color: #000;
    }
    
    .step-action {
        background-color: #2196F3;
        color: #fff;
    }
    
    .step-observation {
        background-color: #4caf50;
        color: #fff;
    }
    
    .workspace-file {
        background-color: #f8f9fa;
        padding: 0.75rem;
        border-radius: 6px;
        border: 1px solid #dee2e6;
        margin: 0.5rem 0;
    }
    
    .stat-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# Session state initialization
if 'assistant' not in st.session_state:
    st.session_state.assistant = None
if 'tasks' not in st.session_state:
    st.session_state.tasks = []
if 'current_task' not in st.session_state:
    st.session_state.current_task = None
if 'workspace_dir' not in st.session_state:
    st.session_state.workspace_dir = "data/research"


def create_research_assistant():
    """Initialize the research assistant."""
    try:
        # Check for API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            st.error("⚠️ OPENAI_API_KEY not found in environment variables")
            return None
        
        # Create workspace directory
        os.makedirs(st.session_state.workspace_dir, exist_ok=True)
        
        # Initialize LLM
        llm = OpenAIChatModel(
            model_name="gpt-3.5-turbo",
            temperature=0.0
        )
        
        # Initialize tools
        tools = [
            CalculatorTool.create(),
            SearchTool.create(),
            FileReadTool.create(base_directory=st.session_state.workspace_dir),
            FileWriteTool.create(base_directory=st.session_state.workspace_dir),
            FileListTool.create(base_directory=st.session_state.workspace_dir),
        ]
        
        # Initialize agent
        agent = ReActAgent.create(
            llm=llm,
            tools=tools,
            verbose=True,
            max_iterations=10
        )
        
        # Initialize executor
        executor = AgentExecutor.create(
            agent=agent,
            tools=tools,
            max_iterations=10,
            verbose=True,
            return_intermediate_steps=True
        )
        
        return {
            'llm': llm,
            'tools': tools,
            'agent': agent,
            'executor': executor
        }
    
    except Exception as e:
        st.error(f"Error initializing assistant: {str(e)}")
        return None


def execute_research_task(task: str):
    """Execute a research task and return results."""
    if not st.session_state.assistant:
        return None
    
    try:
        # Execute task
        result = st.session_state.assistant['executor'].run(task)
        
        # Store task in history
        task_record = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'task': task,
            'result': result,
            'output': result.get('output', 'No output'),
            'steps': result.get('intermediate_steps', [])
        }
        
        st.session_state.tasks.append(task_record)
        st.session_state.current_task = task_record
        
        return task_record
    
    except Exception as e:
        st.error(f"Error executing task: {str(e)}")
        return None


def display_agent_steps(steps: List):
    """Display agent reasoning steps."""
    if not steps:
        return
    
    st.markdown("### 🧠 Agent Reasoning Process")
    
    for i, step in enumerate(steps, 1):
        action = step[0]
        observation = step[1]
        
        # Display thought process
        if hasattr(action, 'log') and action.log:
            with st.expander(f"Step {i}: Thought Process", expanded=True):
                st.markdown(f"""
                <div class="thought-process">
                    <span class="step-badge step-thought">💭 THOUGHT</span>
                    {action.log}
                </div>
                """, unsafe_allow_html=True)
        
        # Display action
        with st.expander(f"Step {i}: Action - {action.tool}", expanded=True):
            st.markdown(f"""
            <div class="tool-execution">
                <span class="step-badge step-action">🔧 ACTION</span>
                <strong>Tool:</strong> {action.tool}<br>
                <strong>Input:</strong> {action.tool_input}
            </div>
            """, unsafe_allow_html=True)
            
            # Display observation
            st.markdown(f"""
            <div class="tool-execution">
                <span class="step-badge step-observation">👁️ OBSERVATION</span>
                {observation}
            </div>
            """, unsafe_allow_html=True)


def list_workspace_files():
    """List files in the workspace directory."""
    try:
        files = []
        for filename in os.listdir(st.session_state.workspace_dir):
            filepath = os.path.join(st.session_state.workspace_dir, filename)
            if os.path.isfile(filepath):
                size = os.path.getsize(filepath)
                modified = datetime.fromtimestamp(os.path.getmtime(filepath))
                files.append({
                    'name': filename,
                    'size': size,
                    'modified': modified.strftime("%Y-%m-%d %H:%M:%S")
                })
        return files
    except Exception as e:
        st.error(f"Error listing files: {str(e)}")
        return []


def read_workspace_file(filename: str):
    """Read a file from the workspace."""
    try:
        filepath = os.path.join(st.session_state.workspace_dir, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        return None


def generate_report():
    """Generate a summary report of all tasks."""
    if not st.session_state.tasks:
        st.warning("No tasks to report on yet!")
        return
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"research_summary_{timestamp}.txt"
    
    report_content = f"""# Research Assistant Summary Report
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Tasks Completed: {len(st.session_state.tasks)}

"""
    
    for i, task in enumerate(st.session_state.tasks, 1):
        report_content += f"""
{'='*80}
Task {i}: {task['timestamp']}
{'='*80}

Query: {task['task']}

Result: {task['output']}

Steps Taken: {len(task['steps'])}

"""
    
    report_content += f"""
{'='*80}
End of Report
{'='*80}

This report was automatically generated by the Research Assistant.
"""
    
    # Save report
    filepath = os.path.join(st.session_state.workspace_dir, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    st.success(f"📄 Report saved: {filename}")
    return filename


# Main UI
st.markdown('<div class="main-header">🔬 Research Assistant</div>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Initialize assistant if not already done
    if st.session_state.assistant is None:
        with st.spinner("Initializing Research Assistant..."):
            st.session_state.assistant = create_research_assistant()
            if st.session_state.assistant:
                st.success("✅ Assistant initialized!")
    
    # Display available tools
    if st.session_state.assistant:
        st.subheader("🔧 Available Tools")
        tools = st.session_state.assistant['tools']
        for tool in tools:
            st.text(f"• {tool.name}")
    
    st.markdown("---")
    
    # Workspace info
    st.subheader("📁 Workspace")
    st.info(f"Directory: {st.session_state.workspace_dir}")
    
    # Statistics
    st.subheader("📊 Statistics")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="stat-box">
            <h3>{len(st.session_state.tasks)}</h3>
            <p>Tasks</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        files = list_workspace_files()
        st.markdown(f"""
        <div class="stat-box">
            <h3>{len(files)}</h3>
            <p>Files</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Actions
    st.subheader("🎯 Actions")
    if st.button("📝 Generate Summary Report", use_container_width=True):
        generate_report()
    
    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.tasks = []
        st.session_state.current_task = None
        st.rerun()

# Main content area
tab1, tab2, tab3 = st.tabs(["💬 Research Tasks", "📂 Workspace", "📜 History"])

with tab1:
    st.header("Research Task Interface")
    
    # Task examples
    with st.expander("💡 Example Tasks"):
        st.markdown("""
        **Calculations:**
        - Calculate the compound interest on $10,000 at 5% for 3 years
        - What is 15% of 2,450?
        
        **Information Search:**
        - Search for information about machine learning
        - Find the latest news about artificial intelligence
        
        **File Operations:**
        - List all files in the workspace
        - Read the content of research_notes.txt
        - Create a file called summary.txt with key findings
        
        **Complex Tasks:**
        - Research Python programming, calculate how many years it's been around, and save a summary
        - Find information about climate change and create a report with statistics
        """)
    
    # Task input
    task_input = st.text_area(
        "Describe your research task:",
        height=100,
        placeholder="E.g., Search for information about quantum computing and save a summary to quantum_notes.txt"
    )
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        execute_button = st.button("🚀 Execute Task", type="primary", use_container_width=True)
    with col2:
        if st.button("🔄 Clear", use_container_width=True):
            st.rerun()
    
    # Execute task
    if execute_button and task_input:
        if st.session_state.assistant:
            with st.spinner("🤔 Agent is working..."):
                result = execute_research_task(task_input)
            
            if result:
                st.success("✅ Task completed!")
                
                # Display final answer
                st.markdown(f"""
                <div class="final-answer">
                    <h3>📋 Final Answer</h3>
                    {result['output']}
                </div>
                """, unsafe_allow_html=True)
                
                # Display agent steps
                if result['steps']:
                    display_agent_steps(result['steps'])
        else:
            st.error("Please initialize the assistant first!")
    
    # Display current task if exists
    if st.session_state.current_task and not execute_button:
        st.markdown("### 📌 Latest Task Result")
        task = st.session_state.current_task
        
        st.markdown(f"""
        <div class="task-card">
            <strong>🕐 Time:</strong> {task['timestamp']}<br>
            <strong>📝 Task:</strong> {task['task']}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="final-answer">
            <h3>📋 Result</h3>
            {task['output']}
        </div>
        """, unsafe_allow_html=True)
        
        if task['steps']:
            display_agent_steps(task['steps'])

with tab2:
    st.header("📂 Workspace Browser")
    
    files = list_workspace_files()
    
    if files:
        st.markdown(f"**Total Files:** {len(files)}")
        
        for file in files:
            with st.expander(f"📄 {file['name']}"):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.text(f"Size: {file['size']} bytes")
                    st.text(f"Modified: {file['modified']}")
                with col2:
                    if st.button("👁️ View", key=f"view_{file['name']}"):
                        content = read_workspace_file(file['name'])
                        if content:
                            st.code(content, language="text")
    else:
        st.info("No files in workspace yet. Complete some tasks to generate files!")

with tab3:
    st.header("📜 Task History")
    
    if st.session_state.tasks:
        st.markdown(f"**Total Tasks:** {len(st.session_state.tasks)}")
        
        for i, task in enumerate(reversed(st.session_state.tasks), 1):
            with st.expander(f"Task {len(st.session_state.tasks) - i + 1}: {task['timestamp']}"):
                st.markdown(f"""
                <div class="task-card">
                    <strong>📝 Task:</strong> {task['task']}<br><br>
                    <strong>📋 Result:</strong> {task['output']}<br><br>
                    <strong>🔧 Steps:</strong> {len(task['steps'])}
                </div>
                """, unsafe_allow_html=True)
                
                if st.checkbox("Show detailed steps", key=f"steps_{i}"):
                    display_agent_steps(task['steps'])
    else:
        st.info("No tasks completed yet. Start by executing a research task!")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>🔬 Research Assistant powered by LangChain Framework</p>
    <p>Built with ReAct Agent • OpenAI GPT-3.5 • Multiple Tools</p>
</div>
""", unsafe_allow_html=True)

# Made with Bob
