**LLM Creation Guide**

This short guide shows different ways to create and invoke LLMs in Python, with concise examples and trade-offs. Use the examples to pick the right approach for your application.

**Install**:

- **Python packages (common)**:

```bash
pip install langchain openai transformers huggingface_hub
```

**Environment**:

- Set provider API keys in environment variables, e.g. `OPENAI_API_KEY`, `HUGGINGFACEHUB_API_TOKEN`.

**1. LangChain non-chat LLM (prompt -> text)**

- Import: `from langchain.llms import OpenAI`
- Use when you only need simple prompt-to-text behavior.

Example:

```python
from langchain.llms import OpenAI

llm = OpenAI(temperature=0.7, model_name="gpt-3.5-turbo")
out = llm("Write a one-line haiku about coffee")
print(out)
```

**2. LangChain chat model (role-based messages)**

- Import: `from langchain.chat_models import ChatOpenAI`
- Use when you need system/user/assistant roles, function calling, or chat features.

Example:

```python
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

chat = ChatOpenAI(temperature=0.2, model_name="gpt-4o-mini")
resp = chat([HumanMessage(content="Tell a short joke about computers")])
print(resp.generations[0][0].text)
```

**3. Provider-specific package vs LangChain wrapper**

- `langchain.chat_models.ChatOpenAI` (LangChain core): stable, integrated with LangChain chains, agents, callbacks.
- `from langchain_openai import ChatOpenAI` (provider package): may expose provider-specific features earlier; ties you to the provider distribution.

Use LangChain for portability across providers; use provider package for bleeding-edge provider features.

**4. Direct provider SDK (no LangChain)**

- Example: OpenAI Python SDK. Useful for minimal dependencies or when you need provider-specific features not wrapped by LangChain.

```python
import os
import openai

openai.api_key = os.environ["OPENAI_API_KEY"]
resp = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[{"role":"user","content":"Write a one-line haiku about coffee"}],
)
print(resp["choices"][0]["message"]["content"].strip())
```

**5. Azure OpenAI**

- LangChain provides `AzureChatOpenAI` / `AzureOpenAI` wrappers. Alternatively, use the provider SDK with Azure endpoint and key.

**6. Anthropic / Cohere / HuggingFaceHub**

- LangChain supports provider wrappers like `ChatAnthropic`, `Anthropic`, `Cohere`, `HuggingFaceHub`.
- Example (HuggingFaceHub):

```python
from langchain.llms import HuggingFaceHub
llm = HuggingFaceHub(repo_id="google/flan-t5-large", huggingfacehub_api_token="...")
print(llm("Summarize in one sentence: LangChain"))
```

**7. Self-hosted / Transformers locally**

- Use `transformers` pipelines for local inference or self-hosted models.

```python
from transformers import pipeline
pipe = pipeline("text-generation", model="gpt2")
print(pipe("Write a one-line haiku about coffee", max_length=30))
```

**8. Unified helper in this repo**

- See the helper module created here: [genai-langchain/llms/generic_llm.py](genai-langchain/llms/generic_llm.py)
- Example using the helper:

```python
from genai_langchain.llms.generic_llm import create_client, invoke_sync

client = create_client("openai", model="gpt-3.5-turbo", chat=False, temperature=0.7)
text = invoke_sync(client, "One-line haiku about coffee")
print(text)
```

**Comparison (high-level)**

- **LangChain wrappers**: + Portable, integrated with Chains/Agents/Callbacks. - Extra dependency and abstraction.
- **Provider SDK (direct)**: + Lightweight and full provider features. - Tighter coupling to provider, more glue code for Chains/Agents.
- **Local Transformers**: + No external API cost; full control. - Requires hardware and model management.

**When to choose**

- Small utilities or scripts → direct SDK.
- Production multi-step pipelines with tools → LangChain chat models + agents.
- Low-latency local inference → transformers or on-premise endpoint.

**Best practices**

- Keep API keys in env vars, not in code.
- Prefer chat wrappers when using tools, function calling, or role context.
- Use async methods for concurrent high-throughput workloads.
- Use streaming/callbacks for UI/low-latency token updates.

**Run examples**

```bash
python genai-langchain/llms/example_direct_llm.py
python genai-langchain/llms/example_chat.py
python genai-langchain/llms/example_direct_without_langchain.py
```

**Need more?**

- I can expand this into a longer README, add unit tests, or generate snippets for specific providers (Azure, Anthropic, Cohere). Tell me which you'd like next.
