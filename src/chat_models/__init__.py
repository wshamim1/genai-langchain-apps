"""
Chat Models Module
Provides implementations for various chat model providers using LangChain.
"""

from .base import BaseChatModel, ChatMessage, ChatResponse

# Import providers with optional dependencies
try:
    from .openai_chat import OpenAIChatModel
    __all_providers__ = ["OpenAIChatModel"]
except ImportError:
    OpenAIChatModel = None
    __all_providers__ = []

try:
    from .anthropic_chat import AnthropicChatModel
    __all_providers__.append("AnthropicChatModel")
except ImportError:
    AnthropicChatModel = None

try:
    from .google_chat import GoogleChatModel
    __all_providers__.append("GoogleChatModel")
except ImportError:
    GoogleChatModel = None

# Re-export LangChain message types for convenience
try:
    from langchain_core.messages import (
        BaseMessage,
        HumanMessage,
        AIMessage,
        SystemMessage,
        FunctionMessage,
    )
    __all_messages__ = [
        "BaseMessage",
        "HumanMessage",
        "AIMessage",
        "SystemMessage",
        "FunctionMessage",
    ]
except ImportError:
    # Fallback if langchain_core is not installed
    BaseMessage = None
    HumanMessage = None
    AIMessage = None
    SystemMessage = None
    FunctionMessage = None
    __all_messages__ = []

__all__ = [
    "BaseChatModel",
    "ChatMessage",
    "ChatResponse",
] + __all_providers__ + __all_messages__

# Made with Bob
