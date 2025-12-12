from typing import Literal
from dotenv import load_dotenv
from pathlib import Path

env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import MessagesState, StateGraph, END
from langgraph.prebuilt import ToolNode

AGENT_REASON = "agent_reason"
ACT = "act"
LAST = -1

# Simple in-memory knowledge base (simulating vector DB)
KNOWLEDGE_BASE = {
    "flutter": {
        "content": "Flutter is a cross-platform mobile development framework created by Google. It uses Dart programming language and allows building apps for iOS, Android, and web from a single codebase. Flutter has hot reload, rich widgets, and excellent performance.",
        "metadata": {"course": "Flutter", "topic": "Mobile Development", "students": 150}
    },
    "kotlin": {
        "content": "Kotlin is a modern programming language for Android development. It's officially supported by Google and offers null safety, coroutines, and concise syntax. Kotlin is 100% interoperable with Java.",
        "metadata": {"course": "Kotlin", "topic": "Android Development", "students": 120}
    },
    "langchain": {
        "content": "LangChain is a framework for developing applications powered by language models. It provides tools for chains, agents, memory management, and RAG (Retrieval Augmented Generation).",
        "metadata": {"course": "LangChain", "topic": "AI Development", "students": 80}
    },
    "ai": {
        "content": "AI and Machine Learning courses cover neural networks, deep learning, and practical applications using Python and TensorFlow. Topics include supervised learning, unsupervised learning, and reinforcement learning.",
        "metadata": {"course": "AI/ML", "topic": "Artificial Intelligence", "students": 200}
    }
}


@tool
def search_courses(query: str) -> str:
    """Search ALL courses from the knowledge base that match the query. Always use this to get course information."""
    query_lower = query.lower()
    results = []

    # Keywords that might indicate what user is looking for
    search_terms = ["mobile", "android", "flutter", "kotlin", "ai", "machine", "learning", "langchain"]

    # Check all courses for matches
    for key, data in KNOWLEDGE_BASE.items():
        meta = data["metadata"]
        topic_lower = meta["topic"].lower()
        course_lower = meta["course"].lower()

        # Match against query or topic
        if (key in query_lower or
            course_lower in query_lower or
            any(term in topic_lower for term in query_lower.split()) or
            any(term in query_lower for term in [key, course_lower])):

            result = f"**{meta['course']}** (Topic: {meta['topic']}, Students: {meta['students']})\n{data['content']}"
            results.append(result)

    if results:
        return "\n\n".join(results)
    return "No courses found matching your query. Available courses: Flutter, Kotlin, LangChain, AI/ML"


@tool
def get_course_details(course_name: str) -> str:
    """Get detailed information including student count for a specific course"""
    course_lower = course_name.lower()

    for key, data in KNOWLEDGE_BASE.items():
        if key in course_lower or data["metadata"]["course"].lower() in course_lower:
            meta = data["metadata"]
            return f"Course: {meta['course']}\nTopic: {meta['topic']}\nEnrolled Students: {meta['students']}\n\nDescription: {data['content']}"

    return "Course not found in knowledge base."


def run_agent_reasoning(state: MessagesState) -> MessagesState:
    """Agent decides what to do based on the current state"""
    llm = ChatOllama(temperature=0, model="llama3.1:8b")
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
    print("🚀 LangGraph with Knowledge Base (RAG)\n")
    print("⚠️  Make sure Ollama is running: 'ollama serve'\n")

    app = build_graph()

    # Query 1: Get detailed course info with student count
    query = "Tell me about Flutter and how many students are enrolled?"

    # Query 2: Search for multiple courses
    # query = "What courses are available about mobile development and AI?"

    print(f"Query: {query}\n")

    result = app.invoke({"messages": [HumanMessage(content=query)]})

    messages = result.get("messages", [])
    if messages:
        print("\n" + "="*60)
        print(messages[LAST].content)
        print("="*60)
        print(f"\nTotal messages: {len(messages)}")

    print("\n✅ Done!")


if __name__ == "__main__":
    main()

