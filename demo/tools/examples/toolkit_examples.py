"""
Examples demonstrating toolkits in the LangChain framework.

This script shows how to use toolkits that group related tools together.
"""

import os
import sys
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.toolkits import FileSystemToolkit

# Load environment variables
load_dotenv()


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def example_file_system_toolkit():
    """Example: Using the File System Toolkit"""
    print_section("Example 1: File System Toolkit")
    
    # Create the toolkit
    toolkit = FileSystemToolkit.create(base_directory=".")
    
    print("File System Toolkit created!\n")
    
    # Get all tools
    tools = toolkit.get_tools()
    print(f"Number of tools: {len(tools)}\n")
    
    # Display tool information
    print("Available Tools:")
    for name in toolkit.get_tool_names():
        print(f"  - {name}")
    
    print("\nTool Descriptions:")
    for desc in toolkit.get_tool_descriptions():
        print(f"  • {desc}")
    
    print("\n" + "-" * 80 + "\n")
    
    # Use the tools
    print("Using the toolkit tools:\n")
    
    # Get individual tools
    file_read = tools[0]
    file_write = tools[1]
    file_list = tools[2]
    
    # 1. List files
    print("1. Listing files in data/ directory:")
    result = file_list.run("data")
    print(result)
    
    # 2. Write a file
    print("\n2. Creating a new file:")
    content = """# File System Toolkit Demo

This file was created using the FileSystemToolkit.

The toolkit provides:
- File reading capabilities
- File writing capabilities
- Directory listing capabilities

All tools work together to provide comprehensive file system access!
"""
    result = file_write.run("data/toolkit_demo.txt|" + content)
    print(result)
    
    # 3. Read the file back
    print("\n3. Reading the created file:")
    result = file_read.run("data/toolkit_demo.txt")
    print(result)


def example_toolkit_with_llm_simulation():
    """Example: Simulating how an LLM would use a toolkit"""
    print_section("Example 2: Toolkit with LLM Simulation")
    
    toolkit = FileSystemToolkit.create(base_directory=".")
    tools = toolkit.get_tools()
    
    print("Simulating an LLM agent using the File System Toolkit:\n")
    
    # Simulate a series of actions an LLM might take
    actions = [
        {
            "thought": "I need to check what files are available in the data directory",
            "tool": "file_list",
            "input": "data"
        },
        {
            "thought": "I should create a summary file of the available tools",
            "tool": "file_write",
            "input": "data/tools_summary.txt|Available Tools:\n1. Calculator\n2. Search\n3. File Operations\n4. API Requests"
        },
        {
            "thought": "Let me verify the summary was created correctly",
            "tool": "file_read",
            "input": "data/tools_summary.txt"
        },
    ]
    
    # Execute the simulated actions
    tool_map = {tool.name: tool for tool in tools}
    
    for i, action in enumerate(actions, 1):
        print(f"Step {i}:")
        print(f"Thought: {action['thought']}")
        print(f"Action: Use {action['tool']} with input: {action['input'][:50]}...")
        
        tool = tool_map.get(action['tool'])
        if tool:
            result = tool.run(action['input'])
            print(f"Observation: {result[:200]}...")
        else:
            print(f"Observation: Tool {action['tool']} not found")
        
        print()


def example_custom_toolkit():
    """Example: Creating a Custom Toolkit"""
    print_section("Example 3: Custom Toolkit Concept")
    
    print("Creating a custom toolkit for data analysis:\n")
    
    # This demonstrates the concept of creating custom toolkits
    print("""
Custom Toolkit Structure:

class DataAnalysisToolkit(BaseToolkit):
    '''Toolkit for data analysis tasks'''
    
    def get_tools(self):
        return [
            CSVLoaderTool(),
            DataStatsTool(),
            DataVisualizationTool(),
            DataExportTool(),
        ]

Benefits of Toolkits:
1. Organized tool collections
2. Related tools grouped together
3. Easy to add/remove tools
4. Consistent interface
5. Reusable across projects

Use Cases:
- File system operations (read, write, list)
- API interactions (GET, POST, PUT, DELETE)
- Data processing (load, transform, analyze)
- Web scraping (fetch, parse, extract)
- Database operations (query, insert, update)
""")


def example_toolkit_best_practices():
    """Example: Best Practices for Using Toolkits"""
    print_section("Example 4: Toolkit Best Practices")
    
    print("""
Best Practices for Using Toolkits:

1. Security:
   - Always validate inputs
   - Use base_directory restrictions for file operations
   - Sanitize user inputs before passing to tools
   - Implement proper error handling

2. Organization:
   - Group related tools together
   - Use descriptive tool names
   - Provide clear tool descriptions
   - Document expected inputs/outputs

3. Error Handling:
   - Tools should return informative error messages
   - Don't expose sensitive information in errors
   - Log errors for debugging
   - Provide fallback options

4. Performance:
   - Cache results when appropriate
   - Use async operations for I/O-bound tasks
   - Implement timeouts for external calls
   - Batch operations when possible

5. Testing:
   - Test each tool individually
   - Test toolkit as a whole
   - Test error conditions
   - Test with various inputs

Example Implementation:
""")
    
    # Demonstrate best practices
    toolkit = FileSystemToolkit.create(base_directory="data")
    
    print("\n✓ Toolkit created with restricted base_directory")
    print(f"✓ Base directory: data/")
    print(f"✓ Number of tools: {len(toolkit.get_tools())}")
    print(f"✓ All tools have clear names and descriptions")
    
    # Show tool information
    print("\nTool Information:")
    for tool in toolkit.get_tools():
        print(f"\n  Tool: {tool.name}")
        print(f"  Description: {tool.description}")
        print(f"  Base Directory: {tool.base_directory}")


def main():
    """Run all toolkit examples."""
    print("\n" + "=" * 80)
    print("  LangChain Toolkits Examples")
    print("=" * 80)
    
    try:
        # Run examples
        example_file_system_toolkit()
        example_toolkit_with_llm_simulation()
        example_custom_toolkit()
        example_toolkit_best_practices()
        
        print("\n" + "=" * 80)
        print("  All toolkit examples completed successfully!")
        print("=" * 80 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

# Made with Bob
