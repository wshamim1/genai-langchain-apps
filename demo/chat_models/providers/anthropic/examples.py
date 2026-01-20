"""
Anthropic (Claude) Examples
Examples using Anthropic's Claude models.
"""

import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent.parent.parent.parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

# Add parent directory to path to import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

from src.chat_models import AnthropicChatModel, HumanMessage, AIMessage, SystemMessage


def example_simple_chat():
    """Simplest possible example."""
    print("\n=== Simple Chat ===")
    
    model = AnthropicChatModel(model_name="claude-3-sonnet-20240229")
    response = model.chat("What is Python?")
    
    print(f"Response: {response.content}")


def example_with_system_message():
    """Chat with system instructions."""
    print("\n=== Chat with System Message ===")
    
    model = AnthropicChatModel(model_name="claude-3-sonnet-20240229")
    response = model.chat(
        message="Explain recursion",
        system_message="You are a patient teacher. Use simple language."
    )
    
    print(f"Response: {response.content}")


def example_conversation():
    """Multi-turn conversation."""
    print("\n=== Conversation with History ===")
    
    model = AnthropicChatModel(model_name="claude-3-sonnet-20240229")
    
    # First turn
    history = []
    response1 = model.chat("My name is John")
    history.append(HumanMessage(content="My name is John"))
    history.append(AIMessage(content=response1.content))
    print(f"User: My name is John")
    print(f"Claude: {response1.content}\n")
    
    # Second turn - Claude should remember the name
    response2 = model.chat("What's my name?", history=history)
    print(f"User: What's my name?")
    print(f"Claude: {response2.content}")


def example_streaming():
    """Streaming response."""
    print("\n=== Streaming Response ===")
    
    model = AnthropicChatModel(model_name="claude-3-sonnet-20240229")
    messages = [HumanMessage(content="Count from 1 to 5")]
    
    print("Response: ", end="", flush=True)
    for chunk in model.stream_generate(messages):
        print(chunk, end="", flush=True)
    print("\n")


def example_different_models():
    """Using different Claude models."""
    print("\n=== Different Claude Models ===")
    
    question = "What is AI in one sentence?"
    
    # Haiku - Fast and compact
    model_haiku = AnthropicChatModel(model_name="claude-3-haiku-20240307")
    response1 = model_haiku.chat(question)
    print(f"Haiku: {response1.content}")
    
    # Sonnet - Balanced
    model_sonnet = AnthropicChatModel(model_name="claude-3-sonnet-20240229")
    response2 = model_sonnet.chat(question)
    print(f"Sonnet: {response2.content}")


def example_token_usage():
    """Monitor token usage."""
    print("\n=== Token Usage Tracking ===")
    
    model = AnthropicChatModel(model_name="claude-3-sonnet-20240229")
    response = model.chat("Explain machine learning in one sentence")
    
    print(f"Response: {response.content}")
    if response.usage:
        print(f"\nToken Usage:")
        print(f"  Prompt tokens: {response.usage['prompt_tokens']}")
        print(f"  Completion tokens: {response.usage['completion_tokens']}")
        print(f"  Total tokens: {response.usage['total_tokens']}")


if __name__ == "__main__":
    print("=" * 60)
    print("Anthropic (Claude) Chat Examples")
    print("=" * 60)
    
    try:
        example_simple_chat()
        example_with_system_message()
        example_conversation()
        example_streaming()
        example_different_models()
        example_token_usage()
        
        print("\n" + "=" * 60)
        print("✅ All examples completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure:")
        print("1. You have installed: pip install langchain-anthropic")
        print("2. Your .env file has ANTHROPIC_API_KEY set")

# Made with Bob
