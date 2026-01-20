"""
Tool-calling agent implementation.

This agent uses structured tool calling to decide which tools to use.
"""

from typing import Any, List
from .base import BaseAgent, AgentAction, AgentFinish


class ToolCallingAgent(BaseAgent):
    """Agent that uses structured tool calling.
    
    This agent formats tool information and lets the LLM decide
    which tool to call with what arguments.
    """
    
    def plan(
        self,
        intermediate_steps: List[tuple],
        **kwargs: Any
    ) -> AgentAction | AgentFinish:
        """Decide what action to take next.
        
        Args:
            intermediate_steps: List of (AgentAction, observation) tuples
            **kwargs: Must include 'input' - the user's question
            
        Returns:
            Either an AgentAction to take or AgentFinish if done
        """
        # Build conversation history
        conversation = self._build_conversation(intermediate_steps, kwargs.get("input", ""))
        
        # Get LLM response
        response = self.llm.invoke(conversation)
        
        # Extract content
        if hasattr(response, 'content'):
            text = response.content
        else:
            text = str(response)
        
        if self.verbose:
            print(f"\n🤖 Agent Response:\n{text}\n")
        
        # Simple parsing - look for tool usage patterns
        if "TOOL:" in text and "INPUT:" in text:
            lines = text.split("\n")
            tool_name = None
            tool_input = None
            
            for line in lines:
                if line.startswith("TOOL:"):
                    tool_name = line.replace("TOOL:", "").strip()
                elif line.startswith("INPUT:"):
                    tool_input = line.replace("INPUT:", "").strip()
            
            if tool_name and tool_input:
                return AgentAction(
                    tool=tool_name,
                    tool_input=tool_input,
                    log=text
                )
        
        # If no tool usage detected, return final answer
        return AgentFinish(
            return_values={"output": text},
            log=text
        )
    
    def _build_conversation(self, intermediate_steps: List[tuple], user_input: str) -> str:
        """Build the conversation prompt.
        
        Args:
            intermediate_steps: List of (AgentAction, observation) tuples
            user_input: The user's input
            
        Returns:
            Formatted conversation string
        """
        prompt = f"""You are a helpful AI assistant with access to the following tools:

{self.format_tools()}

To use a tool, respond with:
TOOL: <tool_name>
INPUT: <tool_input>

If you have enough information to answer, provide the final answer directly.

User Question: {user_input}
"""
        
        # Add intermediate steps
        if intermediate_steps:
            prompt += "\n\nPrevious Actions:\n"
            for action, observation in intermediate_steps:
                prompt += f"\nTool Used: {action.tool}"
                prompt += f"\nInput: {action.tool_input}"
                prompt += f"\nResult: {observation}\n"
        
        return prompt
    
    @staticmethod
    def create(llm: Any, tools: List[Any], verbose: bool = False, max_iterations: int = 10) -> "ToolCallingAgent":
        """Factory method to create a tool-calling agent.
        
        Args:
            llm: The language model to use
            tools: List of tools available to the agent
            verbose: Whether to print verbose output
            max_iterations: Maximum number of iterations
            
        Returns:
            A new ToolCallingAgent instance
        """
        return ToolCallingAgent(
            llm=llm,
            tools=tools,
            verbose=verbose,
            max_iterations=max_iterations
        )

# Made with Bob
