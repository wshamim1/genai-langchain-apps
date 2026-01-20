# Anthropic (Claude) Examples

Examples for using Anthropic's Claude models.

## Setup

### 1. Install Dependencies
```bash
pip install langchain-core langchain-anthropic python-dotenv
```

### 2. Set API Key
Add to your `.env` file:
```
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

## Run Examples

```bash
python3 demo/chat_models/providers/anthropic/examples.py
```

## Available Models

- `claude-3-haiku-20240307` - Fast and compact
- `claude-3-sonnet-20240229` - Balanced (default)
- `claude-3-opus-20240229` - Most capable
- `claude-2.1` - Previous generation

## Quick Code

```python
from src.chat_models import AnthropicChatModel

model = AnthropicChatModel(model_name="claude-3-sonnet-20240229")
response = model.chat("Your question")
print(response.content)
```

## Examples Included

- ✅ Simple chat
- ✅ System messages
- ✅ Conversation history
- ✅ Streaming
- ✅ Different Claude models
- ✅ Token tracking