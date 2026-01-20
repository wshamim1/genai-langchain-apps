"""
Agent executor for running agents with tools.

This module provides the execution loop for agents.
"""

from typing import Any, Dict, List, Optional
from .base import BaseAgent, AgentAction, AgentFinish


class AgentExecutor:
    """Executor that runs an agent with tools.
    
    The executor handles the agent loop:
    1. Agent decides what to do
    2. Execute the action (call a tool)
    3. Observe the result
    4. Repeat until agent finishes or max iterations reached
    """
    
    def __init__(
        self,
        agent: BaseAgent,
        tools: List[Any],
        max_iterations: int = 10,
        verbose: bool = False,
        return_intermediate_steps: bool = False,
    ):
        """Initialize the agent executor.
        
        Args:
            agent: The agent to execute
            tools: List of tools available to the agent
            max_iterations: Maximum number of iterations
            verbose: Whether to print verbose output
            return_intermediate_steps: Whether to return intermediate steps
        """
        self.agent = agent
        self.tools = {tool.name: tool for tool in tools}
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.return_intermediate_steps = return_intermediate_steps
    
    def run(self, input: str) -> Dict[str, Any]:
        """Run the agent on an input.
        
        Args:
            input: The input question or task
            
        Returns:
            Dictionary with 'output' and optionally 'intermediate_steps'
        """
        intermediate_steps: List[tuple] = []
        iterations = 0
        
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"🚀 Starting Agent Execution")
            print(f"{'='*80}")
            print(f"📝 Input: {input}\n")
        
        while iterations < self.max_iterations:
            iterations += 1
            
            if self.verbose:
                print(f"\n{'─'*80}")
                print(f"🔄 Iteration {iterations}/{self.max_iterations}")
                print(f"{'─'*80}")
            
            # Agent decides what to do
            try:
                output = self.agent.plan(
                    intermediate_steps=intermediate_steps,
                    input=input
                )
            except Exception as e:
                if self.verbose:
                    print(f"\n❌ Error in agent planning: {str(e)}")
                return {
                    "output": f"Error: {str(e)}",
                    "intermediate_steps": intermediate_steps if self.return_intermediate_steps else None
                }
            
            # Check if agent is finished
            if isinstance(output, AgentFinish):
                if self.verbose:
                    print(f"\n{'='*80}")
                    print(f"✅ Agent Finished")
                    print(f"{'='*80}")
                    print(f"📤 Output: {output.return_values.get('output', '')}\n")
                
                result = {
                    "output": output.return_values.get("output", ""),
                }
                
                if self.return_intermediate_steps:
                    result["intermediate_steps"] = intermediate_steps
                
                return result
            
            # Execute the action
            if isinstance(output, AgentAction):
                if self.verbose:
                    print(f"\n🔧 Action: {output.tool}")
                    print(f"📥 Input: {output.tool_input}")
                
                # Get the tool
                tool = self.tools.get(output.tool)
                
                if tool is None:
                    observation = f"Error: Tool '{output.tool}' not found. Available tools: {list(self.tools.keys())}"
                    if self.verbose:
                        print(f"❌ {observation}")
                else:
                    # Execute the tool
                    try:
                        observation = tool.run(output.tool_input)
                        if self.verbose:
                            print(f"📊 Observation: {observation[:200]}{'...' if len(str(observation)) > 200 else ''}")
                    except Exception as e:
                        observation = f"Error executing tool: {str(e)}"
                        if self.verbose:
                            print(f"❌ {observation}")
                
                # Add to intermediate steps
                intermediate_steps.append((output, observation))
        
        # Max iterations reached
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"⚠️  Max iterations ({self.max_iterations}) reached")
            print(f"{'='*80}\n")
        
        result = {
            "output": f"Agent stopped after {self.max_iterations} iterations without finishing.",
        }
        
        if self.return_intermediate_steps:
            result["intermediate_steps"] = intermediate_steps
        
        return result
    
    def __call__(self, input: str) -> Dict[str, Any]:
        """Allow the executor to be called like a function.
        
        Args:
            input: The input question or task
            
        Returns:
            Dictionary with 'output' and optionally 'intermediate_steps'
        """
        return self.run(input)
    
    @staticmethod
    def create(
        agent: BaseAgent,
        tools: List[Any],
        max_iterations: int = 10,
        verbose: bool = False,
        return_intermediate_steps: bool = False,
    ) -> "AgentExecutor":
        """Factory method to create an agent executor.
        
        Args:
            agent: The agent to execute
            tools: List of tools available to the agent
            max_iterations: Maximum number of iterations
            verbose: Whether to print verbose output
            return_intermediate_steps: Whether to return intermediate steps
            
        Returns:
            A new AgentExecutor instance
        """
        return AgentExecutor(
            agent=agent,
            tools=tools,
            max_iterations=max_iterations,
            verbose=verbose,
            return_intermediate_steps=return_intermediate_steps,
        )

# Made with Bob
