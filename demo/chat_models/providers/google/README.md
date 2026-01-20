# Google (Gemini) Examples

Examples for using Google's Gemini models.

## Setup

### 1. Install Dependencies
```bash
pip install langchain-core langchain-google-genai python-dotenv
```

### 2. Set API Key
Add to your `.env` file:
```
GOOGLE_API_KEY=your-key-here
```

## Run Examples

```bash
python3 demo/chat_models/providers/google/examples.py
```

## Available Models

- `gemini-pro` - General purpose (default)
- `gemini-pro-vision` - Multimodal capabilities
- `gemini-ultra` - Most capable (when available)

## Quick Code

```python
from src.chat_models import GoogleChatModel

model = GoogleChatModel(model_name="gemini-pro")
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