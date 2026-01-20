# Provider-Specific Examples

This directory contains organized examples for each AI provider. Each provider has its own folder with dedicated examples and documentation.

## Structure

```
demo/chat_models/providers/
├── openai/          # OpenAI (GPT) examples
│   ├── examples.py  # All OpenAI examples
│   └── README.md    # OpenAI setup guide
├── anthropic/       # Anthropic (Claude) examples
│   ├── examples.py  # All Claude examples
│   └── README.md    # Anthropic setup guide
└── google/          # Google (Gemini) examples
    ├── examples.py  # All Gemini examples
    └── README.md    # Google setup guide
```

## Quick Start

### Today: Using OpenAI
```bash
# Run OpenAI examples
python3 demo/chat_models/providers/openai/examples.py
```

### Tomorrow: Switch to Anthropic
```bash
# 1. Install Anthropic package
pip install langchain-anthropic

# 2. Add API key to .env
echo "ANTHROPIC_API_KEY=sk-ant-your-key" >> .env

# 3. Run Anthropic examples
python3 demo/chat_models/providers/anthropic/examples.py
```

### Later: Try Google Gemini
```bash
# 1. Install Google package
pip install langchain-google-genai

# 2. Add API key to .env
echo "GOOGLE_API_KEY=your-key" >> .env

# 3. Run Google examples
python3 demo/chat_models/providers/google/examples.py
```

## Benefits of This Structure

✅ **Easy to Switch** - Just run a different file
✅ **No Code Changes** - Same API across providers
✅ **Clean Organization** - Each provider isolated
✅ **Quick Testing** - Compare providers easily
✅ **No Conflicts** - Provider dependencies separate

## Switching Providers

### Method 1: Run Different Files
```bash
# Today
python3 demo/chat_models/providers/openai/examples.py

# Tomorrow
python3 demo/chat_models/providers/anthropic/examples.py
```

### Method 2: Copy and Modify
```bash
# Copy OpenAI example as template
cp demo/chat_models/providers/openai/examples.py my_app.py

# Change just the import:
# from src.chat_models import OpenAIChatModel
# to
# from src.chat_models import AnthropicChatModel

# Change model initialization:
# model = OpenAIChatModel(model_name="gpt-3.5-turbo")
# to
# model = AnthropicChatModel(model_name="claude-3-sonnet-20240229")
```

## Comparing Providers

Run all providers to compare:

```bash
# OpenAI
python3 demo/chat_models/providers/openai/examples.py > openai_results.txt

# Anthropic (when ready)
python3 demo/chat_models/providers/anthropic/examples.py > anthropic_results.txt

# Google (when ready)
python3 demo/chat_models/providers/google/examples.py > google_results.txt

# Compare results
diff openai_results.txt anthropic_results.txt
```

## Code Similarity

All providers use the same API:

```python
# OpenAI
from src.chat_models import OpenAIChatModel
model = OpenAIChatModel(model_name="gpt-3.5-turbo")

# Anthropic
from src.chat_models import AnthropicChatModel
model = AnthropicChatModel(model_name="claude-3-sonnet-20240229")

# Google
from src.chat_models import GoogleChatModel
model = GoogleChatModel(model_name="gemini-pro")

# Everything else is the same!
response = model.chat("Your question")
print(response.content)
```

## Current Status

- ✅ **OpenAI** - Ready to use (you have API key)
- ⏳ **Anthropic** - Ready when you add API key
- ⏳ **Google** - Ready when you add API key

## Next Steps

1. **Today**: Use OpenAI examples
2. **Tomorrow**: Add Anthropic key and try Claude
3. **Later**: Add Google key and try Gemini
4. **Compare**: See which provider works best for your use case

No need to change your code structure - just switch which file you run!