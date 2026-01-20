# OpenAI Examples

Examples for using OpenAI's GPT models (GPT-3.5, GPT-4, etc.)

## Setup

### 1. Install Dependencies
```bash
pip install langchain-core langchain-openai python-dotenv
```

### 2. Set API Key
Add to your `.env` file:
```
OPENAI_API_KEY=sk-proj-your-key-here
```

## Run Examples

```bash
python3 demo/chat_models/providers/openai/examples.py
```

## Available Models

- `gpt-3.5-turbo` - Fast and cost-effective (default)
- `gpt-4` - Most capable
- `gpt-4-turbo` - Faster GPT-4
- `gpt-4o` - Optimized GPT-4

## Quick Code

```python
from src.chat_models import OpenAIChatModel

model = OpenAIChatModel(model_name="gpt-3.5-turbo")
response = model.chat("Your question")
print(response.content)
```

## Examples Included

- ✅ Simple chat
- ✅ System messages
- ✅ Conversation history
- ✅ Streaming
- ✅ Temperature control
- ✅ Token tracking