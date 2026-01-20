"""
Google Chat Model Implementation
Uses langchain-google-genai for Gemini models integration.
"""

import os
from typing import List, Optional, Dict, Any

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage
from langchain_core.language_models.chat_models import BaseChatModel as LangChainBaseChatModel

from .base import BaseChatModel, ChatResponse


class GoogleChatModel(BaseChatModel):
    """
    Google Chat Model implementation using LangChain.
    Supports Gemini models like gemini-pro, gemini-pro-vision, etc.
    """
    
    def __init__(
        self,
        model_name: str = "gemini-pro",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        api_key: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize Google chat model.
        
        Args:
            model_name: Google model name (default: gemini-pro)
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            api_key: Google API key (defaults to GOOGLE_API_KEY env var)
            **kwargs: Additional Google API parameters
        """
        super().__init__(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            **kwargs
        )
        
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Google API key must be provided or set in GOOGLE_API_KEY environment variable")
    
    def _initialize_model(self) -> LangChainBaseChatModel:
        """Initialize and return the LangChain ChatGoogleGenerativeAI model."""
        model_kwargs = {
            "model": self.model_name,
            "temperature": self.temperature,
            "google_api_key": self.api_key,
        }
        
        if self.max_tokens:
            model_kwargs["max_output_tokens"] = self.max_tokens
        
        # Add any additional kwargs
        model_kwargs.update(self.kwargs)
        
        return ChatGoogleGenerativeAI(**model_kwargs)
    
    def generate(
        self,
        messages: List[BaseMessage],
        **kwargs
    ) -> ChatResponse:
        """
        Generate a response using Google's Generative AI API via LangChain.
        
        Args:
            messages: List of chat messages
            **kwargs: Additional parameters
            
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
            usage_data = result.response_metadata.get('usage_metadata', {})
            if usage_data:
                usage = {
                    "prompt_tokens": usage_data.get('prompt_token_count', 0),
                    "completion_tokens": usage_data.get('candidates_token_count', 0),
                    "total_tokens": usage_data.get('total_token_count', 0),
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
        Asynchronously generate a response using Google's Generative AI API.
        
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
            usage_data = result.response_metadata.get('usage_metadata', {})
            if usage_data:
                usage = {
                    "prompt_tokens": usage_data.get('prompt_token_count', 0),
                    "completion_tokens": usage_data.get('candidates_token_count', 0),
                    "total_tokens": usage_data.get('total_token_count', 0),
                }
        
        return ChatResponse(
            content=result.content,
            model=self.model_name,
            finish_reason=getattr(result, 'finish_reason', None),
            usage=usage,
            raw_response=result
        )

# Made with Bob
