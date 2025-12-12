from typing import Literal
from dotenv import load_dotenv
from pathlib import Path
import shutil

env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langgraph.graph import MessagesState, StateGraph, END
from langgraph.prebuilt import ToolNode

AGENT_REASON = "agent_reason"
ACT = "act"
LAST = -1
CHROMA_DB_PATH = "./chroma_db"


def setup_chromadb(reset=False):
    """Create and populate real ChromaDB with course information"""
    
    # Reset database if requested
    if reset and Path(CHROMA_DB_PATH).exists():
        print("🗑️  Deleting existing ChromaDB...")
        shutil.rmtree(CHROMA_DB_PATH)
    
    # Check if database already exists
    if Path(CHROMA_DB_PATH).exists():
        print("📂 Loading existing ChromaDB from disk...")
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        vectorstore = Chroma(
            persist_directory=CHROMA_DB_PATH,
            embedding_function=embeddings
        )
        print(f"✅ Loaded ChromaDB from: {Path(CHROMA_DB_PATH).absolute()}\n")
        return vectorstore
    
    print("🔧 Creating new ChromaDB...")
    print("⏳ Generating embeddings with nomic-embed-text (this may take a moment)...")

    # Course information
    documents = [
        "Flutter is a cross-platform mobile development framework created by Google. It uses Dart programming language and allows building apps for iOS, Android, and web from a single codebase. Flutter has hot reload, rich widgets, and excellent performance.",
        "Kotlin is a modern programming language for Android development. It's officially supported by Google and offers null safety, coroutines, and concise syntax. Kotlin is 100% interoperable with Java.",
        "LangChain is a framework for developing applications powered by language models. It provides tools for chains, agents, memory management, and RAG (Retrieval Augmented Generation).",
        "AI and Machine Learning courses cover neural networks, deep learning, and practical applications using Python and TensorFlow. Topics include supervised learning, unsupervised learning, and reinforcement learning.",
        "Ruby Learner offers tech courses in Burmese language, making education accessible to Myanmar learners who want to learn Flutter, Kotlin, and AI technologies."
    ]
    
    metadatas = [
        {"course": "Flutter", "topic": "Mobile Development", "students": 150, "id": "flutter"},
        {"course": "Kotlin", "topic": "Android Development", "students": 120, "id": "kotlin"},
        {"course": "LangChain", "topic": "AI Development", "students": 80, "id": "langchain"},
        {"course": "AI/ML", "topic": "Artificial Intelligence", "students": 200, "id": "ai"},
        {"course": "General", "topic": "Platform Info", "students": 550, "id": "general"}
    ]
    
    # Create embeddings using nomic-embed-text (optimized for embeddings)
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    # Create ChromaDB vector store
    vectorstore = Chroma.from_texts(
        texts=documents,
        embedding=embeddings,
        metadatas=metadatas,
        persist_directory=CHROMA_DB_PATH
    )
    
    print(f"✅ ChromaDB created at: {Path(CHROMA_DB_PATH).absolute()}")
    print(f"📊 Stored {len(documents)} documents with vector embeddings\n")
    
    return vectorstore


# Initialize ChromaDB
print("🚀 Initializing Real ChromaDB...\n")
vectorstore = setup_chromadb(reset=True)  # Reset to use new embedding model


@tool
def search_courses(query: str) -> str:
    """Search courses using semantic similarity in ChromaDB vector database"""
    print(f"🔍 Searching ChromaDB for: '{query}'")
    
    # Use similarity search (finds semantically similar content)
    results = vectorstore.similarity_search(query, k=3)
    
    if results:
        formatted_results = []
        for doc in results:
            meta = doc.metadata
            result = f"**{meta.get('course', 'Unknown')}** (Topic: {meta.get('topic', 'N/A')}, Students: {meta.get('students', 0)})\n{doc.page_content}"
            formatted_results.append(result)
        
        return "\n\n".join(formatted_results)
    
    return "No courses found in ChromaDB."


@tool
def get_course_details(course_name: str) -> str:
    """Get detailed information for a specific course from ChromaDB"""
    print(f"📖 Getting details for: '{course_name}'")
    
    # Search for the specific course
    results = vectorstore.similarity_search(course_name, k=1)
    
    if results:
        doc = results[0]
        meta = doc.metadata
        return f"Course: {meta.get('course', 'Unknown')}\nTopic: {meta.get('topic', 'N/A')}\nEnrolled Students: {meta.get('students', 0)}\n\nDescription: {doc.page_content}"
    
    return "Course not found in ChromaDB."


def run_agent_reasoning(state: MessagesState) -> MessagesState:
    """Agent decides what to do based on the current state"""
    llm = ChatOllama(temperature=0, model="llama3.1:8b-instruct-q8_0")
    llm_with_tools = llm.bind_tools([search_courses, get_course_details])
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


def should_continue(state: MessagesState) -> Literal["act", "end"]:
    """Decide whether to continue using tools or end"""
    messages = state.get("messages", [])
    if not messages:
        return "end"

    last_message = messages[LAST]

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        print(f"\n🔧 Using {len(last_message.tool_calls)} tool(s)\n")
        return "act"
    else:
        print("\n✅ Done reasoning\n")
        return "end"


def build_graph() -> StateGraph:
    """Build the LangGraph workflow"""
    flow = StateGraph(MessagesState)

    flow.add_node(AGENT_REASON, run_agent_reasoning)

    tool_node = ToolNode([search_courses, get_course_details])
    flow.add_node(ACT, tool_node)

    flow.set_entry_point(AGENT_REASON)
    flow.add_conditional_edges(AGENT_REASON, should_continue, {"act": ACT, "end": END})
    flow.add_edge(ACT, AGENT_REASON)

    return flow.compile()


def main():
    print("="*60)
    print("🚀 LangGraph with REAL ChromaDB Vector Database")
    print("="*60)
    print("⚠️  Make sure Ollama is running: 'ollama serve'\n")

    app = build_graph()

    # Query 1: Test semantic search
    query = "Tell me about mobile app development frameworks and student enrollment?"

    # Query 2: Test specific course
    # query = "What is Kotlin and how many students?"

    # Query 3: Test semantic similarity
    # query = "I want to learn about AI and machine learning"

    print(f"Query: {query}\n")

    result = app.invoke({"messages": [HumanMessage(content=query)]})

    messages = result.get("messages", [])
    if messages:
        print("\n" + "="*60)
        print(messages[LAST].content)
        print("="*60)
        print(f"\nTotal messages: {len(messages)}")

    print("\n✅ Done!")
    print(f"\n💾 ChromaDB stored at: {Path(CHROMA_DB_PATH).absolute()}")


if __name__ == "__main__":
    main()

