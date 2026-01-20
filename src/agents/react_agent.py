"""
ReAct (Reasoning and Acting) agent implementation.

This agent uses the ReAct prompting strategy to reason about actions
and observations in an interleaved manner.
"""

import re
from typing import Any, List
from langchain_core.messages import HumanMessage, SystemMessage

from .base import BaseAgent, AgentAction, AgentFinish


class ReActAgent(BaseAgent):
    """ReAct agent that reasons and acts in an interleaved manner.
    
    The ReAct framework combines reasoning traces and task-specific actions,
    allowing the agent to reason about what to do and then take action.
    """
    
    system_prompt: str = """You are a helpful AI assistant that can use tools to accomplish tasks.

You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
{agent_scratchpad}"""
    
    def plan(
        self,
        intermediate_steps: List[tuple],
        **kwargs: Any
    ) -> AgentAction | AgentFinish:
        """Decide what action to take next using ReAct reasoning.
        
        Args:
            intermediate_steps: List of (AgentAction, observation) tuples
            **kwargs: Must include 'input' - the user's question
            
        Returns:
            Either an AgentAction to take or AgentFinish if done
        """
        # Build the agent scratchpad from intermediate steps
        agent_scratchpad = self._construct_scratchpad(intermediate_steps)
        
        # Format the prompt
        prompt = self.system_prompt.format(
            tools=self.format_tools(),
            tool_names=", ".join(self.get_tool_names()),
            input=kwargs.get("input", ""),
            agent_scratchpad=agent_scratchpad
        )
        
        # Get LLM response
        messages = [SystemMessage(content=prompt)]
        response = self.llm.invoke(messages)
        
        # Extract content from response
        if hasattr(response, 'content'):
            text = response.content
        else:
            text = str(response)
        
        if self.verbose:
            print(f"\n🤖 Agent Response:\n{text}\n")
        
        # Parse the response
        return self._parse_output(text)
    
    def _construct_scratchpad(self, intermediate_steps: List[tuple]) -> str:
        """Construct the agent scratchpad from intermediate steps.
        
        Args:
            intermediate_steps: List of (AgentAction, observation) tuples
            
        Returns:
            Formatted scratchpad string
        """
        if not intermediate_steps:
            return ""
        
        thoughts = []
        for action, observation in intermediate_steps:
            thoughts.append(f"Thought: {action.log}")
            thoughts.append(f"Action: {action.tool}")
            thoughts.append(f"Action Input: {action.tool_input}")
            thoughts.append(f"Observation: {observation}")
        
        return "\n".join(thoughts)
    
    def _parse_output(self, text: str) -> AgentAction | AgentFinish:
        """Parse the LLM output into an AgentAction or AgentFinish.
        
        Args:
            text: The LLM's output text
            
        Returns:
            Either an AgentAction or AgentFinish
        """
        # Check for Final Answer
        if "Final Answer:" in text:
            final_answer = text.split("Final Answer:")[-1].strip()
            return AgentFinish(
                return_values={"output": final_answer},
                log=text
            )
        
        # Extract Action and Action Input
        action_match = re.search(r"Action:\s*(.+?)(?:\n|$)", text)
        action_input_match = re.search(r"Action Input:\s*(.+?)(?:\n|$)", text, re.DOTALL)
        
        if action_match and action_input_match:
            action = action_match.group(1).strip()
            action_input = action_input_match.group(1).strip()
            
            # Extract thought if present
            thought_match = re.search(r"Thought:\s*(.+?)(?:\n|$)", text)
            thought = thought_match.group(1).strip() if thought_match else ""
            
            return AgentAction(
                tool=action,
                tool_input=action_input,
                log=thought
            )
        
        # If we can't parse, return a finish with the raw text
        return AgentFinish(
            return_values={"output": text},
            log="Could not parse LLM output"
        )
    
    @staticmethod
    def create(llm: Any, tools: List[Any], verbose: bool = False, max_iterations: int = 10) -> "ReActAgent":
        """Factory method to create a ReAct agent.
        
        Args:
            llm: The language model to use
            tools: List of tools available to the agent
            verbose: Whether to print verbose output
            max_iterations: Maximum number of iterations
            
        Returns:
            A new ReActAgent instance
        """
        return ReActAgent(
            llm=llm,
            tools=tools,
            verbose=verbose,
            max_iterations=max_iterations
        )

# Made with Bob
