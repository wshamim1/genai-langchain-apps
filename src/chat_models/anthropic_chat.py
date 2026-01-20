"""
Anthropic Chat Model Implementation
Uses langchain-anthropic for Claude models integration.
"""

import os
from typing import List, Optional, Dict, Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage
from langchain_core.language_models.chat_models import BaseChatModel as LangChainBaseChatModel

from .base import BaseChatModel, ChatResponse


class AnthropicChatModel(BaseChatModel):
    """
    Anthropic Chat Model implementation using LangChain.
    Supports Claude models like claude-3-opus, claude-3-sonnet, claude-2.1, etc.
    """
    
    def __init__(
        self,
        model_name: str = "claude-3-sonnet-20240229",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        api_key: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize Anthropic chat model.
        
        Args:
            model_name: Anthropic model name (default: claude-3-sonnet-20240229)
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate (required for Anthropic)
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            **kwargs: Additional Anthropic API parameters
        """
        super().__init__(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            **kwargs
        )
        
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key must be provided or set in ANTHROPIC_API_KEY environment variable")
    
    def _initialize_model(self) -> LangChainBaseChatModel:
        """Initialize and return the LangChain ChatAnthropic model."""
        model_kwargs = {
            "model": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens or 1024,
            "anthropic_api_key": self.api_key,
        }
        
        # Add any additional kwargs
        model_kwargs.update(self.kwargs)
        
        return ChatAnthropic(**model_kwargs)
    
    def generate(
        self,
        messages: List[BaseMessage],
        **kwargs
    ) -> ChatResponse:
        """
        Generate a response using Anthropic's messages API via LangChain.
        
        Args:
            messages: List of chat messages
            **kwargs: Additional parameters (top_p, top_k, etc.)
            
        Returns:
            ChatResponse object
        """
        if self.langchain_model is None:
            self.langchain_model = self._initialize_model()
        
        # Invoke the model
        result = self.langchain_model.invoke(messages, **kwargs)
        
        # Extract usage information if available
        usage = None
        if hasattr(result, 'response_metadata') and result.response_metadata:
            usage_data = result.response_metadata.get('usage', {})
            if usage_data:
                usage = {
                    "prompt_tokens": usage_data.get('input_tokens', 0),
                    "completion_tokens": usage_data.get('output_tokens', 0),
                    "total_tokens": usage_data.get('input_tokens', 0) + usage_data.get('output_tokens', 0),
                }
        
        return ChatResponse(
            content=result.content,
            model=self.model_name,
            finish_reason=getattr(result, 'stop_reason', None),
            usage=usage,
            raw_response=result
        )
    
    async def agenerate(
        self,
        messages: List[BaseMessage],
        **kwargs
    ) -> ChatResponse:
        """
        Asynchronously generate a response using Anthropic's messages API.
        
        Args:
            messages: List of chat messages
            **kwargs: Additional parameters
            
        Returns:
            ChatResponse object
        """
        if self.langchain_model is None:
            self.langchain_model = self._initialize_model()
        
        # Async invoke the model
        result = await self.langchain_model.ainvoke(messages, **kwargs)
        
        # Extract usage information if available
        usage = None
        if hasattr(result, 'response_metadata') and result.response_metadata:
            usage_data = result.response_metadata.get('usage', {})
            if usage_data:
                usage = {
                    "prompt_tokens": usage_data.get('input_tokens', 0),
                    "completion_tokens": usage_data.get('output_tokens', 0),
                    "total_tokens": usage_data.get('input_tokens', 0) + usage_data.get('output_tokens', 0),
                }
        
        return ChatResponse(
            content=result.content,
            model=self.model_name,
            finish_reason=getattr(result, 'stop_reason', None),
            usage=usage,
            raw_response=result
        )

# Made with Bob
