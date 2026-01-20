"""
Examples demonstrating agents in the LangChain framework.

This script shows how to use agents with tools to accomplish tasks.
"""

import os
import sys
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.chat_models import OpenAIChatModel
from src.tools import CalculatorTool, SearchTool, FileReadTool, FileWriteTool
from src.agents import ReActAgent, ToolCallingAgent, AgentExecutor

# Load environment variables
load_dotenv()


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def example_react_agent_with_calculator():
    """Example: ReAct Agent with Calculator Tool"""
    print_section("Example 1: ReAct Agent with Calculator")
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY not found. Skipping this example.")
        print("Set your API key in .env file to run this example.\n")
        return
    
    # Create LLM
    llm = OpenAIChatModel(model_name="gpt-3.5-turbo", temperature=0)
    
    # Create tools
    tools = [
        CalculatorTool.create(),
    ]
    
    # Create agent
    agent = ReActAgent.create(
        llm=llm,
        tools=tools,
        verbose=True,
        max_iterations=5
    )
    
    # Create executor
    executor = AgentExecutor.create(
        agent=agent,
        tools=tools,
        max_iterations=5,
        verbose=True
    )
    
    # Run the agent
    question = "What is 25 * 4 + 10?"
    result = executor.run(question)
    
    print(f"\n✅ Final Answer: {result['output']}\n")


def example_agent_with_multiple_tools():
    """Example: Agent with Multiple Tools"""
    print_section("Example 2: Agent with Multiple Tools")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY not found. Skipping this example.")
        return
    
    # Create LLM
    llm = OpenAIChatModel(model_name="gpt-3.5-turbo", temperature=0)
    
    # Create multiple tools
    tools = [
        CalculatorTool.create(),
        SearchTool.create(),
        FileReadTool.create(base_directory="."),
    ]
    
    # Create agent
    agent = ToolCallingAgent.create(
        llm=llm,
        tools=tools,
        verbose=True
    )
    
    # Create executor
    executor = AgentExecutor.create(
        agent=agent,
        tools=tools,
        verbose=True
    )
    
    # Run with a calculation task
    question = "Calculate 15 * 8 and then tell me what that number is"
    result = executor.run(question)
    
    print(f"\n✅ Final Answer: {result['output']}\n")


def example_agent_with_file_operations():
    """Example: Agent with File Operations"""
    print_section("Example 3: Agent with File Operations")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY not found. Skipping this example.")
        return
    
    # Create LLM
    llm = OpenAIChatModel(model_name="gpt-3.5-turbo", temperature=0)
    
    # Create file operation tools
    tools = [
        FileReadTool.create(base_directory="data"),
        FileWriteTool.create(base_directory="data"),
        CalculatorTool.create(),
    ]
    
    # Create agent
    agent = ToolCallingAgent.create(
        llm=llm,
        tools=tools,
        verbose=True
    )
    
    # Create executor
    executor = AgentExecutor.create(
        agent=agent,
        tools=tools,
        verbose=True
    )
    
    # Task: Calculate something and save to file
    question = "Calculate 100 * 1.15 and save the result to a file called agent_calculation.txt"
    result = executor.run(question)
    
    print(f"\n✅ Final Answer: {result['output']}\n")


def example_agent_without_api_key():
    """Example: Demonstrating Agent Structure (No API Key Required)"""
    print_section("Example 4: Agent Structure Demo (No API Required)")
    
    print("""
This example demonstrates the agent structure without requiring an API key.

Agent Components:
1. Agent - Decides what to do (ReActAgent, ToolCallingAgent)
2. Tools - Actions the agent can take (Calculator, Search, File Ops)
3. Executor - Runs the agent loop

Agent Loop:
1. Agent receives input
2. Agent decides which tool to use
3. Tool is executed
4. Agent observes the result
5. Repeat until task is complete or max iterations reached

Example Agent Flow:

User: "What is 25 * 4 + 10?"

Iteration 1:
  Agent Thought: "I need to calculate 25 * 4 + 10"
  Agent Action: Use calculator tool
  Tool Input: "25 * 4 + 10"
  Observation: "Result: 110"

Iteration 2:
  Agent Thought: "I now have the answer"
  Final Answer: "The result is 110"

Benefits of Agents:
- Autonomous decision making
- Can use multiple tools
- Iterative problem solving
- Handles complex multi-step tasks
- Adapts based on observations
""")


def example_agent_best_practices():
    """Example: Agent Best Practices"""
    print_section("Example 5: Agent Best Practices")
    
    print("""
Best Practices for Using Agents:

1. Tool Selection:
   - Provide only necessary tools
   - Clear tool descriptions
   - Well-defined tool inputs/outputs
   - Test tools independently first

2. Prompt Engineering:
   - Clear instructions in system prompt
   - Examples of desired behavior
   - Specify output format
   - Handle edge cases

3. Error Handling:
   - Set reasonable max_iterations
   - Handle tool failures gracefully
   - Provide informative error messages
   - Log agent decisions for debugging

4. Performance:
   - Use faster models for simple tasks
   - Cache tool results when appropriate
   - Limit tool execution time
   - Monitor token usage

5. Safety:
   - Validate tool inputs
   - Restrict file system access
   - Rate limit API calls
   - Review agent decisions in production

Example Configuration:
""")
    
    # Show example configuration
    print("""
# Good agent setup
agent = ReActAgent.create(
    llm=llm,
    tools=[calculator, search, file_read],  # Only necessary tools
    verbose=True,  # For debugging
    max_iterations=5  # Prevent infinite loops
)

executor = AgentExecutor.create(
    agent=agent,
    tools=tools,
    max_iterations=5,
    verbose=True,
    return_intermediate_steps=True  # For analysis
)
""")


def example_agent_use_cases():
    """Example: Agent Use Cases"""
    print_section("Example 6: Agent Use Cases")
    
    print("""
Common Agent Use Cases:

1. Research Assistant:
   Tools: Search, File Read, File Write
   Task: Research a topic and create a summary document
   
2. Data Analyst:
   Tools: CSV Loader, Calculator, File Write
   Task: Analyze data and generate reports
   
3. Code Helper:
   Tools: File Read, File Write, Search
   Task: Read code, suggest improvements, update files
   
4. Customer Support:
   Tools: Search (knowledge base), API Request
   Task: Answer questions using company knowledge
   
5. Task Automation:
   Tools: File Ops, API Request, Calculator
   Task: Automate repetitive business processes
   
6. Content Creator:
   Tools: Search, File Write, Image Generation
   Task: Research and create content
   
7. DevOps Assistant:
   Tools: Command Execution, File Ops, API Request
   Task: Monitor systems and perform maintenance
   
8. Personal Assistant:
   Tools: Calendar, Email, Search, Calculator
   Task: Manage schedule and handle tasks

Example Implementation:
""")
    
    print("""
# Research Assistant Agent
tools = [
    SearchTool.create(),
    FileReadTool.create(),
    FileWriteTool.create(),
]

agent = ReActAgent.create(llm=llm, tools=tools)
executor = AgentExecutor.create(agent=agent, tools=tools)

result = executor.run(
    "Research the latest developments in AI and create a summary document"
)
""")


def main():
    """Run all agent examples."""
    print("\n" + "=" * 80)
    print("  LangChain Agents Examples")
    print("=" * 80)
    
    try:
        # Run examples
        example_react_agent_with_calculator()
        example_agent_with_multiple_tools()
        example_agent_with_file_operations()
        example_agent_without_api_key()
        example_agent_best_practices()
        example_agent_use_cases()
        
        print("\n" + "=" * 80)
        print("  All examples completed!")
        print("=" * 80 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

# Made with Bob
