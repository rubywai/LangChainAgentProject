import os
from typing import Literal
from dotenv import load_dotenv
from pathlib import Path

env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch
from langgraph.graph import MessagesState, StateGraph, END
from langgraph.prebuilt import ToolNode

AGENT_REASON = "agent_reason"
ACT = "act"
LAST = -1


@tool
def calculator(operation: str, x: float, y: float) -> str:
    """Perform basic math operations: add, subtract, multiply, divide"""
    if operation == "add":
        return str(x + y)
    elif operation == "subtract":
        return str(x - y)
    elif operation == "multiply":
        return str(x * y)
    elif operation == "divide":
        return str(x / y) if y != 0 else "Cannot divide by zero"
    else:
        return "Unknown operation"


def validate_env() -> None:
    if not os.getenv("TAVILY_API_KEY"):
        raise EnvironmentError("TAVILY_API_KEY not found in .env file")


def run_agent_reasoning(state: MessagesState) -> MessagesState:
    llm = ChatOllama(temperature=0, model="llama3.1:8b")
    tavily_search = TavilySearch(max_results=3)
    llm_with_tools = llm.bind_tools([tavily_search, calculator])
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


def should_continue(state: MessagesState) -> Literal["act", "end"]:
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
    flow = StateGraph(MessagesState)

    flow.add_node(AGENT_REASON, run_agent_reasoning)

    tavily_search = TavilySearch(max_results=3)
    tool_node = ToolNode([tavily_search, calculator])
    flow.add_node(ACT, tool_node)

    flow.set_entry_point(AGENT_REASON)
    flow.add_conditional_edges(AGENT_REASON, should_continue, {"act": ACT, "end": END})
    flow.add_edge(ACT, AGENT_REASON)

    return flow.compile()


def main():
    print("🚀 LangGraph ReAct Agent\n")

    validate_env()
    app = build_graph()

    # Prompt 1: Uses Tavily search for real-time info
    # query = "What is the current weather in Tokyo and what's twice that temperature?"

    # Prompt 2: Uses calculator for math operations
    query = "What is 1234 multiplied by 5678?"

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

