# LangChain Agent Project

A comprehensive collection of LangChain and LangGraph examples demonstrating AI agents, tool calling, real-time web search, and RAG (Retrieval Augmented Generation) with vector databases.

## 🚀 Features

- **Text Summarization** with OpenAI and local Ollama
- **Tool Calling** with static functions
- **Real-time Web Search** with Tavily API
- **Structured Responses** with Pydantic models
- **LangGraph ReAct Agents** with multi-step reasoning
- **RAG with Vector Databases** using ChromaDB
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
├── main_7langgraph_react_agent.py      # LangGraph ReAct agent (NEW! ⭐)
├── main_8chromadb_rag.py               # Knowledge base RAG (in-memory)
├── main_9real_chromadb.py              # Real ChromaDB vector database (NEW! ⭐)
├── main_10_mongo_vector_search.py      # MongoDB Atlas Vector Search (NEW! ⭐)
├── flow_7.png                          # LangGraph visualization
├── .env                                # Environment variables (create this)
├── pyproject.toml                      # Project dependencies
├── chroma_db/                          # ChromaDB storage (created on first run)
│   ├── chroma.sqlite3                  # SQLite metadata
│   └── [UUID]/                         # Vector embeddings
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

### 7. LangGraph ReAct Agent ⭐ (NEW!)

Multi-step reasoning agent that can use multiple tools and decide when to stop.

```bash
uv run python main_7langgraph_react_agent.py
```

**Features:**
- 🔄 **ReAct Pattern**: Reasoning → Action → Observation loop
- 🛠️ **Multiple Tools**: Calculator + Tavily web search
- 🤖 **Agent Decides**: When to use tools, which tools, and when to stop
- 📊 **Visualization**: Generates `flow_7.png` showing the agent workflow

**Key Difference from LangChain:**
- **LangChain**: Single pass, one-shot tool calling
- **LangGraph**: Multi-step reasoning, can call tools multiple times

**Example Queries:**
```python
# Query 1: Math calculation
"What is 1234 multiplied by 5678?"

# Query 2: Web search + reasoning  
"What is the temperature in Tokyo? List it and then triple it"
```

**Why LangGraph?**
- More control over agent flow
- Can create complex multi-agent systems
- Better for tasks requiring multiple steps
- Similar to how ChatGPT/Claude work internally

---

### 8. Knowledge Base RAG (In-Memory)

Simple RAG example using in-memory knowledge base (no database setup needed).

```bash
uv run python main_8chromadb_rag.py
```

**Features:**
- 📚 Stores course information (Flutter, Kotlin, LangChain, AI/ML)
- 🔍 Agent searches knowledge base before answering
- 👥 Includes metadata (student counts, topics)
- 💡 Perfect for learning RAG concepts

**Example Queries:**
```python
# Query 1
"Tell me about Flutter and how many students are enrolled?"

# Query 2  
"What courses are available about mobile development and AI?"
```

---

### 9. Real ChromaDB Vector Database ⭐ (NEW!)

Production-ready RAG with persistent vector database using ChromaDB.

```bash
uv run python main_9real_chromadb.py
```

**Features:**
- 💾 **Persistent Storage**: Data saved to disk (`./chroma_db/`)
- 🔢 **Vector Embeddings**: Uses Ollama to generate 4096-dimensional vectors
- 🎯 **Semantic Search**: Finds similar content, not just keyword matching
- ⚡ **Fast**: HNSW algorithm for O(log n) similarity search
- 🗄️ **SQLite Backend**: Metadata stored in SQLite, vectors in binary files

**What Gets Created:**
```
chroma_db/
├── chroma.sqlite3          # Metadata (258 KB)
└── [UUID]/
    └── data_level0.bin     # Vector embeddings (1.6 MB)
```

**How It Works:**
1. First run: Creates ChromaDB, generates embeddings (takes ~30 seconds)
2. Later runs: Loads from disk instantly
3. Agent searches by semantic similarity
4. Returns relevant documents with metadata

**Example Output:**
```
🔧 Creating new ChromaDB...
⏳ Generating embeddings with Ollama...
✅ ChromaDB created at: /path/to/chroma_db
📊 Stored 5 documents with vector embeddings

Query: Tell me about mobile app development frameworks

🔍 Searching ChromaDB for: 'mobile app development frameworks'
✅ Found: Flutter, Kotlin, LangChain courses
```

**Why Use ChromaDB?**
- ✅ Store your company's knowledge
- ✅ Semantic search (understands meaning, not just keywords)
- ✅ Scales to millions of documents
- ✅ Perfect for chatbots, documentation search, Q&A systems

See `chromadb_structure.md` for detailed architecture explanation.

---

### 10. MongoDB Atlas Vector Search ⭐ (NEW!)

Cloud-based vector search using MongoDB Atlas with LangChain integration.

```bash
uv run python main_10_mongo_vector_search.py
```

**Features:**
- ☁️ **Cloud Database**: MongoDB Atlas (no local setup needed)
- 🔢 **Vector Embeddings**: Uses Ollama `nomic-embed-text` model
- 🌐 **Atlas Vector Search**: Native MongoDB vector search capabilities
- 🔄 **Fallback Search**: Manual cosine similarity if Atlas search index not configured
- 📊 **Metadata Support**: Stores and queries course info, topics, student counts

**Prerequisites:**
1. MongoDB Atlas account (free tier available)
2. Connection string with password in `.env`:
   ```env
   MONGODB_URI=mongodb+srv://user:password@cluster0.mongodb.net/...
   MONGO_DB_NAME=rab_test
   MONGO_COLLECTION=rab_test
   ```

**What Gets Created:**
```
MongoDB Atlas:
└── rab_test (database)
    └── rab_test (collection)
        ├── Document 1 (page_content, embedding, course, topic, students)
        ├── Document 2 (...)
        └── Document 3 (...)
```

**How It Works:**
1. Connects to MongoDB Atlas cloud database
2. Generates embeddings using Ollama locally
3. Stores documents with vector embeddings in MongoDB
4. Performs semantic similarity search
5. Returns relevant documents with scores

**Example Output:**
```
📌 Query: Tell me about mobile app development frameworks

   Result 1 (similarity: 0.8120):
   Course: Flutter
   Topic: Mobile Development
   Students: 150
   Content: Flutter is a cross-platform mobile development framework...

   Result 2 (similarity: 0.7243):
   Course: Kotlin
   Topic: Android Development
   Students: 120
   Content: Kotlin is a modern programming language...
```

**ChromaDB vs MongoDB Atlas:**

| Feature | ChromaDB (main_9) | MongoDB Atlas (main_10) |
|---------|------------------|------------------------|
| **Storage** | Local files | Cloud database |
| **Setup** | Zero config | Requires MongoDB account |
| **Scalability** | Limited to disk | Unlimited cloud storage |
| **Multi-user** | Single machine | Multiple users/apps |
| **Cost** | Free | Free tier available |
| **Best For** | Development, prototypes | Production, team collaboration |

**When to Use MongoDB Atlas:**
- ✅ Production applications
- ✅ Team collaboration (shared database)
- ✅ Already using MongoDB for other data
- ✅ Need backup/replication
- ✅ Want managed infrastructure

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

## 📊 Quick Comparison

| Feature | LangChain (main_1-6) | LangGraph (main_7) | RAG (main_8-9) | MongoDB Atlas (main_10) |
|---------|---------------------|-------------------|----------------|------------------------|
| **Purpose** | Single-pass tasks | Multi-step reasoning | Knowledge retrieval | Cloud vector search |
| **Tool Calls** | One-shot | Multiple, iterative | Search database | Search cloud DB |
| **State** | Stateless | Stateful graph | Persistent data | Cloud persistent |
| **Storage** | N/A | N/A | Local files | MongoDB Atlas |
| **Use Case** | Simple queries | Complex workflows | Company knowledge | Production RAG |
| **Example** | "What's 2+2?" | "Search, then calculate" | "Find in our docs" | "Team knowledge base" |

### When to Use What?

- **LangChain (main_1-6)**: Quick tasks, single tool calls, real-time data
- **LangGraph (main_7)**: Multi-step reasoning, agent workflows, complex decisions
- **RAG ChromaDB (main_8-9)**: Local development, prototypes, single-user apps
- **RAG MongoDB (main_10)**: Production apps, team collaboration, cloud deployment

---

## 🧠 Learning Path

**Beginner:**
1. `main_1` → Learn basic LangChain
2. `main_2` → Try local Ollama
3. `main_3` → Understand tool calling

**Intermediate:**
4. `main_5` → Real-time search with Tavily
5. `main_7` → Multi-step agents with LangGraph
6. `main_8` → Understand RAG concepts

**Advanced:**
7. `main_9` → Production RAG with ChromaDB (local)
8. `main_10` → Cloud RAG with MongoDB Atlas (production)

---

## 🛠️ Troubleshooting

### Ollama Connection Error
```
httpcore.ConnectError: [Errno 61] Connection refused
```
**Solution:** Start Ollama first
```bash
ollama serve
```

### Tavily API Error
```
ValueError: TAVILY_API_KEY not found
```
**Solution:** Add key to `.env` file
```env
TAVILY_API_KEY=tvly-dev-your-key-here
```

### ChromaDB Not Found
```
ModuleNotFoundError: No module named 'chromadb'
```
**Solution:** Reinstall dependencies
```bash
uv sync
```

### MongoDB Connection Error
```
❌ Error: MONGODB_URI not properly set in .env file
```
**Solution:** Add your MongoDB Atlas connection string to `.env` with actual password
```env
MONGODB_URI=mongodb+srv://username:YOUR_PASSWORD@cluster0.mongodb.net/...
MONGO_DB_NAME=rab_test
MONGO_COLLECTION=rab_test
```

---

## 📚 Additional Resources

- **LangChain Docs**: https://python.langchain.com/
- **LangGraph Docs**: https://langchain-ai.github.io/langgraph/
- **ChromaDB Docs**: https://docs.trychroma.com/
- **MongoDB Atlas**: https://www.mongodb.com/atlas
- **LangChain-MongoDB**: https://github.com/langchain-ai/langchain-mongodb
- **Ollama Models**: https://ollama.ai/library
- **Tavily Search**: https://tavily.com/

---

## 🤝 Contributing

Feel free to add more examples or improve existing ones!

## 📝 License

MIT

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

*Last updated: December 19, 2025*

