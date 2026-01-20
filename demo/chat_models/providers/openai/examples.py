"""
OpenAI Only Example
Simple examples using only OpenAI (no other providers needed).
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

from src.chat_models import OpenAIChatModel, HumanMessage, AIMessage, SystemMessage


def example_simple_chat():
    """Simplest possible example."""
    print("\n=== Simple Chat ===")
    
    model = OpenAIChatModel(model_name="gpt-3.5-turbo")
    response = model.chat("What is Python?")
    
    print(f"Response: {response.content}")


def example_with_system_message():
    """Chat with system instructions."""
    print("\n=== Chat with System Message ===")
    
    model = OpenAIChatModel(model_name="gpt-3.5-turbo")
    response = model.chat(
        message="Explain recursion",
        system_message="You are a patient teacher. Use simple language."
    )
    
    print(f"Response: {response.content}")


def example_conversation():
    """Multi-turn conversation."""
    print("\n=== Conversation with History ===")
    
    model = OpenAIChatModel(model_name="gpt-3.5-turbo")
    
    # First turn
    history = []
    response1 = model.chat("My name is John")
    history.append(HumanMessage(content="My name is John"))
    history.append(AIMessage(content=response1.content))
    print(f"User: My name is John")
    print(f"AI: {response1.content}\n")
    
    # Second turn - AI should remember the name
    response2 = model.chat("What's my name?", history=history)
    print(f"User: What's my name?")
    print(f"AI: {response2.content}")


def example_streaming():
    """Streaming response."""
    print("\n=== Streaming Response ===")
    
    model = OpenAIChatModel(model_name="gpt-3.5-turbo")
    messages = [HumanMessage(content="Count from 1 to 5")]
    
    print("Response: ", end="", flush=True)
    for chunk in model.stream_generate(messages):
        print(chunk, end="", flush=True)
    print("\n")


def example_with_parameters():
    """Using different model parameters."""
    print("\n=== Different Temperatures ===")
    
    question = "Write a creative tagline for a coffee shop"
    
    # Low temperature (more focused)
    model_focused = OpenAIChatModel(model_name="gpt-3.5-turbo", temperature=0.2)
    response1 = model_focused.chat(question)
    print(f"Low temp (0.2): {response1.content}")
    
    # High temperature (more creative)
    model_creative = OpenAIChatModel(model_name="gpt-3.5-turbo", temperature=1.0)
    response2 = model_creative.chat(question)
    print(f"High temp (1.0): {response2.content}")


def example_token_usage():
    """Monitor token usage."""
    print("\n=== Token Usage Tracking ===")
    
    model = OpenAIChatModel(model_name="gpt-3.5-turbo")
    response = model.chat("Explain machine learning in one sentence")
    
    print(f"Response: {response.content}")
    print(f"\nToken Usage:")
    print(f"  Prompt tokens: {response.usage['prompt_tokens']}")
    print(f"  Completion tokens: {response.usage['completion_tokens']}")
    print(f"  Total tokens: {response.usage['total_tokens']}")


if __name__ == "__main__":
    print("=" * 60)
    print("OpenAI Chat Examples")
    print("=" * 60)
    
    try:
        example_simple_chat()
        example_with_system_message()
        example_conversation()
        example_streaming()
        example_with_parameters()
        example_token_usage()
        
        print("\n" + "=" * 60)
        print("✅ All examples completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure:")
        print("1. You have installed dependencies: pip install -r requirements.txt")
        print("2. Your .env file has OPENAI_API_KEY set")

# Made with Bob
