# LangChain Agent Project

A collection of LangChain examples demonstrating AI agents, tool calling, and real-time web search with Tavily.

## 🚀 Features

- **Text Summarization** with OpenAI and local Ollama
- **Tool Calling** with static functions
- **Real-time Web Search** with Tavily API
- **Structured Responses** with Pydantic models
- **Job Search Agent** example

## 📋 Prerequisites

- Python 3.9+
- [uv](https://github.com/astral-sh/uv) package manager
- [Ollama](https://ollama.ai/) (for local LLM)
- API Keys:
  - OpenAI API key (optional, for GPT models)
  - Tavily API key (for real-time search)

## 🔧 Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd LangChainProject
```

2. **Install dependencies**
```bash
uv sync
```

3. **Set up environment variables**

Create a `.env` file in the project root:
```env
OPENAI_API_KEY=sk-proj-your-key-here
TAVILY_API_KEY=tvly-dev-your-key-here
```

4. **Install and run Ollama** (for local models)
```bash
# Install Ollama from https://ollama.ai/
# Pull the llama3.1 model
ollama pull llama3.1:8b
```

## 📁 Project Structure

```
LangChainProject/
├── main_1summerization.py              # Text summarization with OpenAI
├── main_2summerization_with_local_ollama.py  # Summarization with Ollama
├── main_3tool_call.py                  # Static tool calling example
├── main_4tool_call_tavily.py           # Multiple tools + Tavily search
├── main_5simple_tavily.py              # Simple Tavily search with structured output
├── main_6job_search_example.py         # Job search agent example
├── TOOL_CALL_GUIDE.md                  # Detailed guide on tool calling
├── .env                                # Environment variables (create this)
├── pyproject.toml                      # Project dependencies
└── README.md                           # This file
```

## 🎯 Examples

### 1. Text Summarization with OpenAI

Summarizes text about Ruby Learner platform using GPT-3.5-turbo.

```bash
uv run python main_1summerization.py
```

**Output:** A concise 2-bullet summary of the input text.

---

### 2. Text Summarization with Local Ollama

Same as above but uses local Ollama model (no API key needed).

```bash
uv run python main_2summerization_with_local_ollama.py
```

**Advantages:**
- 🆓 Free - no API costs
- 🔒 Private - runs locally
- ⚡ Fast - no network latency

---

### 3. Static Tool Calling

Demonstrates how AI agents can call custom functions to retrieve information.

```bash
uv run python main_3tool_call.py
```

**Features:**
- Static course database
- `get_course_info()` function
- `get_student_count()` function
- LLM decides which tool to use

**Example Query:** "Tell me about the Flutter course and how many students it has."

---

### 4. Multiple Tools + Tavily Search

Combines static tools with real-time web search using Tavily.

```bash
uv run python main_4tool_call_tavily.py
```

**Features:**
- Tests 3 different queries:
  1. Latest AI news (uses Tavily)
  2. Course information (uses static function)
  3. Bitcoin price (uses Tavily)
- LLM automatically chooses the right tool

**Requires:** `TAVILY_API_KEY` in `.env` file

---

### 5. Simple Tavily Search ⭐ (Recommended)

Clean, simple example of real-time web search with structured responses.

```bash
uv run python main_5simple_tavily.py
```

**Features:**
- Real-time web search
- Structured output with Pydantic models
- Includes sources (URLs and titles)
- Easy to customize for your queries

**Example Output:**
```
Answer:
The latest news about AI in December 2025 includes Google's new Gemini 3 Deep Think mode...

Sources:
  1. AI News Today, December 5, 2025...
     https://ts2.tech/en/ai-news-today-december-5-2025...
  2. Anthropic News Today...
     https://ts2.tech/en/anthropic-news-today...
```

---

### 6. Job Search Agent

Searches for job postings on LinkedIn and returns structured results.

```bash
uv run python main_6job_search_example.py
```

**Features:**
- Searches for AI Engineer jobs
- Returns company, title, location, description
- Includes source URLs
- Similar to the reference example pattern

**Example Query:** "Search for 3 job postings for an AI engineer using langchain in the bay area on linkedin"

---

## 🔑 API Keys

### Get Tavily API Key

1. Visit [https://tavily.com/](https://tavily.com/)
2. Sign up for free account
3. Get your API key
4. Add to `.env` file:
   ```env
   TAVILY_API_KEY=tvly-dev-your-key-here
   ```

### Get OpenAI API Key (Optional)

1. Visit [https://platform.openai.com/](https://platform.openai.com/)
2. Create account and add payment method
3. Generate API key
4. Add to `.env` file:
   ```env
   OPENAI_API_KEY=sk-proj-your-key-here
   ```

**Note:** You can use Ollama (free, local) instead of OpenAI for most examples.

---

## 🛠️ Development

### Run with uv

```bash
# Run any script
uv run python <script-name>.py

# Install new dependency
uv add <package-name>

# Sync dependencies
uv sync
```

### IDE Configuration (PyCharm)

1. Set Python interpreter to `.venv/bin/python`
2. Use "uv run" for running scripts
3. Or configure run configuration to use the virtual environment

---

## 📚 Key Concepts

### What is LangChain?

LangChain is a framework for building applications with Large Language Models (LLMs). It provides:
- Tool calling capabilities
- Prompt templates
- Memory management
- Agent workflows

### What are Tools?

Tools are functions that AI agents can call to:
- Search the web (Tavily)
- Query databases
- Perform calculations
- Access external APIs

### What is Structured Output?

Using Pydantic models to get consistent, typed responses from LLMs:
```python
class AgentResponse(BaseModel):
    answer: str
    sources: List[Source]
```

### Local vs Cloud LLMs

| Feature | Ollama (Local) | OpenAI (Cloud) |
|---------|----------------|----------------|
| Cost | Free | Pay per token |
| Privacy | Runs locally | Sends to API |
| Speed | Fast (no network) | Depends on network |
| Quality | Good | Better for complex tasks |

---

## 🐛 Troubleshooting

### "ModuleNotFoundError: No module named 'langchain_community'"

```bash
uv add langchain-community
```

### "TAVILY_API_KEY not found"

Make sure `.env` file exists in project root with:
```env
TAVILY_API_KEY=tvly-dev-your-actual-key-here
```

### "Connection error" with Ollama

1. Make sure Ollama is running:
   ```bash
   ollama serve
   ```
2. Check if model is installed:
   ```bash
   ollama list
   ollama pull llama3.1:8b
   ```

### Run button in IDE uses wrong Python

Configure PyCharm to use `.venv/bin/python` or run with `uv run` command.

---

## 📖 Learn More

- [LangChain Documentation](https://python.langchain.com/)
- [Tavily API Docs](https://docs.tavily.com/)
- [Ollama Documentation](https://ollama.ai/docs)
- [Pydantic Documentation](https://docs.pydantic.dev/)

---

## 🎓 Learning Path

1. **Start here:** `main_2summerization_with_local_ollama.py` - Understand basic LangChain usage
2. **Learn tools:** `main_3tool_call.py` - See how agents call functions
3. **Add search:** `main_5simple_tavily.py` - Real-time web search
4. **Advanced:** `main_6job_search_example.py` - Build practical agents

---

## 🤝 Contributing

Feel free to:
- Add more examples
- Improve documentation
- Report issues
- Suggest features

---

## 📝 License

MIT License

---

## 👨‍💻 Author

**Ruby Learner Myanmar**

---

**Happy Coding! 🚀**

*Last updated: December 7, 2025*

