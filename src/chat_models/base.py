"""
Base Chat Model Interface
Wrapper around LangChain's chat models for unified interface.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)
from langchain_core.language_models.chat_models import BaseChatModel as LangChainBaseChatModel


# Re-export LangChain message types for convenience
ChatMessage = BaseMessage


@dataclass
class ChatResponse:
    """Represents a response from a chat model."""
    content: str
    model: Optional[str] = None
    finish_reason: Optional[str] = None
    usage: Optional[Dict[str, int]] = None
    raw_response: Optional[Any] = None
    
    def to_message(self) -> AIMessage:
        """Convert response to an AIMessage."""
        return AIMessage(content=self.content)


class BaseChatModel(ABC):
    """
    Wrapper base class for chat models.
    Provides a simplified interface around LangChain's chat models.
    """
    
    def __init__(
        self,
        model_name: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        api_key: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the chat model.
        
        Args:
            model_name: Name of the model to use
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate
            api_key: API key for authentication
            **kwargs: Additional provider-specific parameters
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key
        self.kwargs = kwargs
        self.langchain_model: Optional[LangChainBaseChatModel] = None
    
    @abstractmethod
    def _initialize_model(self) -> LangChainBaseChatModel:
        """Initialize and return the LangChain model instance."""
        pass
    
    def generate(
        self,
        messages: List[BaseMessage],
        **kwargs
    ) -> ChatResponse:
        """
        Generate a response from the chat model.
        
        Args:
            messages: List of chat messages
            **kwargs: Additional generation parameters
            
        Returns:
            ChatResponse object containing the model's response
        """
        if self.langchain_model is None:
            self.langchain_model = self._initialize_model()
        
        result = self.langchain_model.invoke(messages, **kwargs)
        
        return ChatResponse(
            content=result.content,
            model=self.model_name,
            raw_response=result
        )
    
    def invoke(self, messages: Union[List[BaseMessage], str], **kwargs) -> Any:
        """
        Invoke the model directly (for agent compatibility).
        
        Args:
            messages: List of messages or a single string
            **kwargs: Additional parameters
            
        Returns:
            The raw model response
        """
        if self.langchain_model is None:
            self.langchain_model = self._initialize_model()
        
        # Convert string to HumanMessage if needed
        if isinstance(messages, str):
            messages = [HumanMessage(content=messages)]
        
        return self.langchain_model.invoke(messages, **kwargs)
    
    async def agenerate(
        self,
        messages: List[BaseMessage],
        **kwargs
    ) -> ChatResponse:
        """
        Asynchronously generate a response from the chat model.
        
        Args:
            messages: List of chat messages
            **kwargs: Additional generation parameters
            
        Returns:
            ChatResponse object containing the model's response
        """
        if self.langchain_model is None:
            self.langchain_model = self._initialize_model()
        
        result = await self.langchain_model.ainvoke(messages, **kwargs)
        
        return ChatResponse(
            content=result.content,
            model=self.model_name,
            raw_response=result
        )
    
    def chat(
        self,
        message: str,
        system_message: Optional[str] = None,
        history: Optional[List[BaseMessage]] = None,
        **kwargs
    ) -> ChatResponse:
        """
        Convenience method for single-turn chat.
        
        Args:
            message: User message
            system_message: Optional system message
            history: Optional conversation history
            **kwargs: Additional generation parameters
            
        Returns:
            ChatResponse object
        """
        messages = []
        
        if system_message:
            messages.append(SystemMessage(content=system_message))
        
        if history:
            messages.extend(history)
        
        messages.append(HumanMessage(content=message))
        
        return self.generate(messages, **kwargs)
    
    async def achat(
        self,
        message: str,
        system_message: Optional[str] = None,
        history: Optional[List[BaseMessage]] = None,
        **kwargs
    ) -> ChatResponse:
        """
        Asynchronous convenience method for single-turn chat.
        
        Args:
            message: User message
            system_message: Optional system message
            history: Optional conversation history
            **kwargs: Additional generation parameters
            
        Returns:
            ChatResponse object
        """
        messages = []
        
        if system_message:
            messages.append(SystemMessage(content=system_message))
        
        if history:
            messages.extend(history)
        
        messages.append(HumanMessage(content=message))
        
        return await self.agenerate(messages, **kwargs)
    
    def stream_generate(self, messages: List[BaseMessage], **kwargs):
        """
        Stream responses from the chat model.
        
        Args:
            messages: List of chat messages
            **kwargs: Additional generation parameters
            
        Yields:
            Chunks of the response as they arrive
        """
        if self.langchain_model is None:
            self.langchain_model = self._initialize_model()
        
        for chunk in self.langchain_model.stream(messages, **kwargs):
            if hasattr(chunk, 'content') and chunk.content:
                yield chunk.content
    
    async def astream_generate(self, messages: List[BaseMessage], **kwargs):
        """
        Asynchronously stream responses from the chat model.
        
        Args:
            messages: List of chat messages
            **kwargs: Additional generation parameters
            
        Yields:
            Chunks of the response as they arrive
        """
        if self.langchain_model is None:
            self.langchain_model = self._initialize_model()
        
        async for chunk in self.langchain_model.astream(messages, **kwargs):
            if hasattr(chunk, 'content') and chunk.content:
                yield chunk.content
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model_name='{self.model_name}')"

# Made with Bob
