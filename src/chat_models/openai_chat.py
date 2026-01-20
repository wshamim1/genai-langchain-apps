"""
OpenAI Chat Model Implementation
Uses langchain-openai for OpenAI GPT models integration.
"""

import os
from typing import List, Optional, Dict, Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage
from langchain_core.language_models.chat_models import BaseChatModel as LangChainBaseChatModel

from .base import BaseChatModel, ChatResponse


class OpenAIChatModel(BaseChatModel):
    """
    OpenAI Chat Model implementation using LangChain.
    Supports models like gpt-3.5-turbo, gpt-4, gpt-4-turbo, etc.
    """
    
    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        api_key: Optional[str] = None,
        organization: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize OpenAI chat model.
        
        Args:
            model_name: OpenAI model name (default: gpt-3.5-turbo)
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens to generate
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            organization: OpenAI organization ID
            base_url: Custom API base URL
            **kwargs: Additional OpenAI API parameters
        """
        super().__init__(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            **kwargs
        )
        
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key must be provided or set in OPENAI_API_KEY environment variable")
        
        self.organization = organization
        self.base_url = base_url
    
    def _initialize_model(self) -> LangChainBaseChatModel:
        """Initialize and return the LangChain ChatOpenAI model."""
        model_kwargs = {
            "model": self.model_name,
            "temperature": self.temperature,
            "openai_api_key": self.api_key,
        }
        
        if self.max_tokens:
            model_kwargs["max_tokens"] = self.max_tokens
        
        if self.organization:
            model_kwargs["openai_organization"] = self.organization
        
        if self.base_url:
            model_kwargs["openai_api_base"] = self.base_url
        
        # Add any additional kwargs
        model_kwargs.update(self.kwargs)
        
        return ChatOpenAI(**model_kwargs)
    
    def generate(
        self,
        messages: List[BaseMessage],
        **kwargs
    ) -> ChatResponse:
        """
        Generate a response using OpenAI's chat completion API via LangChain.
        
        Args:
            messages: List of chat messages
            **kwargs: Additional parameters (top_p, frequency_penalty, etc.)
            
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
            token_usage = result.response_metadata.get('token_usage', {})
            if token_usage:
                usage = {
                    "prompt_tokens": token_usage.get('prompt_tokens', 0),
                    "completion_tokens": token_usage.get('completion_tokens', 0),
                    "total_tokens": token_usage.get('total_tokens', 0),
                }
        
        return ChatResponse(
            content=result.content,
            model=self.model_name,
            finish_reason=getattr(result, 'finish_reason', None),
            usage=usage,
            raw_response=result
        )
    
    async def agenerate(
        self,
        messages: List[BaseMessage],
        **kwargs
    ) -> ChatResponse:
        """
        Asynchronously generate a response using OpenAI's chat completion API.
        
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
            token_usage = result.response_metadata.get('token_usage', {})
            if token_usage:
                usage = {
                    "prompt_tokens": token_usage.get('prompt_tokens', 0),
                    "completion_tokens": token_usage.get('completion_tokens', 0),
                    "total_tokens": token_usage.get('total_tokens', 0),
                }
        
        return ChatResponse(
            content=result.content,
            model=self.model_name,
            finish_reason=getattr(result, 'finish_reason', None),
            usage=usage,
            raw_response=result
        )

# Made with Bob
