from pathlib import Path
import shutil
from typing import Literal

from security_utils import load_project_env, sanitize_error_message

env_path = load_project_env(__file__)

from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

AGENT_REASON = "agent_reason"
ACT = "act"
LAST = -1
CHROMA_DB_PATH = "./chroma_db_main_12"
MEMORY_WINDOW = 12


def build_chromadb(embeddings: OllamaEmbeddings, db_path: Path) -> Chroma:
    documents = [
        "Flutter is a cross-platform mobile development framework created by Google. It uses Dart and supports iOS, Android, web, and desktop apps from one codebase. Flutter is known for hot reload, rich widgets, and strong UI performance.",
        "Kotlin is a modern language for Android development. It offers null safety, coroutines, concise syntax, and full Java interoperability. Google officially supports Kotlin for Android.",
        "LangChain is a framework for building applications with language models. It provides tools for prompts, chains, agents, retrieval, structured output, and memory.",
        "AI and Machine Learning courses cover neural networks, deep learning, practical Python workflows, supervised learning, unsupervised learning, and reinforcement learning.",
        "Ruby Learner offers Burmese-language tech education for Myanmar learners focused on Flutter, Kotlin, and AI topics through accessible online lessons.",
    ]

    metadatas = [
        {"course": "Flutter", "topic": "Mobile Development", "students": 150, "id": "flutter"},
        {"course": "Kotlin", "topic": "Android Development", "students": 120, "id": "kotlin"},
        {"course": "LangChain", "topic": "AI Development", "students": 80, "id": "langchain"},
        {"course": "AI/ML", "topic": "Artificial Intelligence", "students": 200, "id": "ai"},
        {"course": "General", "topic": "Platform Info", "students": 550, "id": "general"},
    ]

    vectorstore = Chroma.from_texts(
        texts=documents,
        embedding=embeddings,
        metadatas=metadatas,
        persist_directory=str(db_path),
    )

    print(f"✅ ChromaDB created at: {db_path.absolute()}")
    return vectorstore


def setup_chromadb() -> Chroma:
    """Load an existing ChromaDB or create one if missing."""
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    db_path = Path(CHROMA_DB_PATH)

    if db_path.exists():
        print("📂 Loading existing ChromaDB from disk...")
        try:
            return Chroma(
                persist_directory=CHROMA_DB_PATH,
                embedding_function=embeddings,
            )
        except Exception as exc:
            print(f"⚠️  Existing ChromaDB is incompatible: {sanitize_error_message(exc)}")
            print("🗑️  Rebuilding local store for main_12...")
            shutil.rmtree(db_path, ignore_errors=True)

    print("🔧 Creating ChromaDB for RAG chat memory demo...")
    print("⏳ Generating embeddings with nomic-embed-text...")
    return build_chromadb(embeddings, db_path)


vectorstore = setup_chromadb()


@tool
def retrieve_course_context(query: str) -> str:
    """Retrieve relevant course context for the user's question. Use this for course facts, comparisons, enrollment, or follow-up questions."""
    print(f"🔍 Retrieving context for: {query!r}")
    results = vectorstore.similarity_search(query, k=3)

    if not results:
        return "No relevant course context found."

    formatted_results = []
    for index, doc in enumerate(results, start=1):
        meta = doc.metadata
        formatted_results.append(
            "\n".join(
                [
                    f"Source {index}",
                    f"Course: {meta.get('course', 'Unknown')}",
                    f"Topic: {meta.get('topic', 'N/A')}",
                    f"Students: {meta.get('students', 'N/A')}",
                    f"Content: {doc.page_content}",
                ]
            )
        )

    return "\n\n".join(formatted_results)


def run_agent_reasoning(state: MessagesState) -> MessagesState:
    llm = ChatOllama(temperature=0, model="gemma4:e2b")
    llm_with_tools = llm.bind_tools([retrieve_course_context])

    system_message = SystemMessage(
        content=(
            "You are a helpful RAG chat assistant for Ruby Learner course data. "
            "Use the retrieval tool whenever the user asks about courses, topics, student counts, "
            "comparisons, or follow-up questions that depend on prior context. "
            "If the retrieved context is insufficient, say so plainly. "
            "When you answer from retrieved context, mention the course names you used."
        )
    )

    response = llm_with_tools.invoke([system_message] + state["messages"])
    return {"messages": [response]}


def should_continue(state: MessagesState) -> Literal["act", "end"]:
    messages = state.get("messages", [])
    if not messages:
        return "end"

    last_message = messages[LAST]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        print(f"\n🔧 Using {len(last_message.tool_calls)} tool(s)\n")
        return "act"

    print("\n✅ Response ready\n")
    return "end"


def build_graph():
    flow = StateGraph(MessagesState)
    flow.add_node(AGENT_REASON, run_agent_reasoning)
    flow.add_node(ACT, ToolNode([retrieve_course_context]))
    flow.set_entry_point(AGENT_REASON)
    flow.add_conditional_edges(AGENT_REASON, should_continue, {"act": ACT, "end": END})
    flow.add_edge(ACT, AGENT_REASON)
    return flow.compile()


def trim_conversation(messages: list, max_messages: int = MEMORY_WINDOW) -> list:
    """Keep recent conversation context without letting memory grow forever."""
    if len(messages) <= max_messages:
        return messages
    return messages[-max_messages:]


def print_ai_reply(messages: list) -> None:
    for message in reversed(messages):
        if isinstance(message, AIMessage) and message.content:
            print("\nAI:")
            print(message.content)
            return
    print("\nAI:")
    print("No final response generated.")


def main():
    print("=" * 72)
    print("RAG Chat + Short-Term Memory Agent")
    print("=" * 72)
    print("Requirements:")
    print("  1. Ollama must be running: ollama serve")
    print("  2. Models should exist locally: gemma4:e2b and nomic-embed-text")
    print()
    print("Try questions like:")
    print("  - Tell me about Flutter")
    print("  - How many students does it have?")
    print("  - Compare it with Kotlin")
    print("Type 'quit' to exit.\n")

    app = build_graph()
    conversation_messages = []

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue

        if user_input.lower() == "quit":
            print("Goodbye!")
            break

        current_turn = conversation_messages + [HumanMessage(content=user_input)]
        result = app.invoke({"messages": current_turn})
        conversation_messages = trim_conversation(result["messages"])

        print_ai_reply(conversation_messages)
        print("\n" + "-" * 72)

    print("\nFinal memory snapshot:")
    for message in conversation_messages:
        if isinstance(message, HumanMessage):
            print(f"Human: {message.content}")
        elif isinstance(message, ToolMessage):
            print(f"Tool: {message.content[:160]}")
        elif isinstance(message, AIMessage):
            print(f"AI: {message.content}")


if __name__ == "__main__":
    main()
