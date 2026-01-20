# Chat Models Demo & Usage Examples

This directory contains comprehensive examples demonstrating how to use the chat models framework.

## Prerequisites

Before running the examples, make sure you have:

1. Installed required dependencies (see main README.md)
2. Set up API keys as environment variables:
   ```bash
   export OPENAI_API_KEY="your-openai-key"
   export ANTHROPIC_API_KEY="your-anthropic-key"
   export GOOGLE_API_KEY="your-google-key"
   ```

## Examples Overview

### 01_basic_chat.py
**Basic Chat Interactions**

Demonstrates:
- Simple single-turn conversations
- Using different providers (OpenAI, Anthropic, Google)
- System messages
- Conversation history
- Basic error handling

Run:
```bash
python demo/usages/01_basic_chat.py
```

### 02_streaming_chat.py
**Streaming Responses**

Demonstrates:
- Synchronous streaming
- Asynchronous streaming
- Real-time response generation
- Concurrent async requests

Run:
```bash
python demo/usages/02_streaming_chat.py
```

### 03_conversation_management.py
**Multi-turn Conversations**

Demonstrates:
- Managing conversation history
- Context-aware conversations
- Role-playing scenarios
- Temperature effects on creativity
- Conversation history management

Run:
```bash
python demo/usages/03_conversation_management.py
```

## Quick Start

### Basic Usage

```python
from src.chat_models import OpenAIChatModel

# Initialize model
model = OpenAIChatModel(
    model_name="gpt-3.5-turbo",
    temperature=0.7
)

# Simple chat
response = model.chat(
    message="Hello, how are you?",
    system_message="You are a helpful assistant."
)

print(response.content)
```

### Streaming Usage

```python
from src.chat_models import OpenAIChatModel
from src.chat_models.base import ChatMessage, MessageRole

model = OpenAIChatModel(model_name="gpt-3.5-turbo")

messages = [
    ChatMessage(role=MessageRole.USER, content="Tell me a story.")
]

# Stream the response
for chunk in model.stream_generate(messages):
    print(chunk, end="", flush=True)
```

### Async Usage

```python
import asyncio
from src.chat_models import OpenAIChatModel
from src.chat_models.base import ChatMessage, MessageRole

async def main():
    model = OpenAIChatModel(model_name="gpt-3.5-turbo")
    
    messages = [
        ChatMessage(role=MessageRole.USER, content="What is AI?")
    ]
    
    response = await model.agenerate(messages)
    print(response.content)

asyncio.run(main())
```

## Supported Models

### OpenAI
- gpt-3.5-turbo
- gpt-4
- gpt-4-turbo
- gpt-4o

### Anthropic
- claude-3-opus-20240229
- claude-3-sonnet-20240229
- claude-3-haiku-20240307
- claude-2.1

### Google
- gemini-pro
- gemini-pro-vision
- gemini-ultra

## Common Parameters

All chat models support these common parameters:

- `model_name`: The specific model to use
- `temperature`: Controls randomness (0.0 to 1.0+)
- `max_tokens`: Maximum tokens to generate
- `api_key`: API key for authentication

## Tips

1. **Temperature**: Lower values (0.0-0.3) for focused/deterministic responses, higher values (0.7-1.0) for creative responses
2. **Max Tokens**: Set appropriate limits to control costs and response length
3. **System Messages**: Use them to set the behavior and personality of the assistant
4. **History Management**: Keep conversation history manageable to avoid token limits
5. **Error Handling**: Always wrap API calls in try-except blocks

## Troubleshooting

### API Key Issues
```
ValueError: API key must be provided or set in environment variable
```
**Solution**: Set the appropriate environment variable for your provider.

### Import Errors
```
ModuleNotFoundError: No module named 'openai'
```
**Solution**: Install required dependencies: `pip install -r requirements.txt`

### Rate Limits
If you encounter rate limit errors, consider:
- Adding delays between requests
- Using exponential backoff
- Upgrading your API plan

## Next Steps

- Explore the source code in `src/chat_models/`
- Check out the main README.md for more information
- Try modifying the examples to suit your needs
- Build your own applications using these chat models

## Support

For issues or questions:
1. Check the documentation
2. Review the example code
3. Consult the provider's API documentation