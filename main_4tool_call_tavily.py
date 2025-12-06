from langchain.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_ollama import ChatOllama
from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv
import os
from pathlib import Path

# Load .env file
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)


@tool
def get_course_info(course_name: str) -> str:
    """Get information about Ruby Learner courses (Flutter, Kotlin).

    Args:
        course_name: The name of the course (e.g., 'Flutter', 'Kotlin')
    """
    courses = {
        "Flutter": "Flutter course teaches cross-platform mobile app development. Duration: 12 weeks, 45 students enrolled.",
        "Kotlin": "Kotlin course teaches Android app development. Duration: 10 weeks, 32 students enrolled."
    }
    return courses.get(course_name, f"Course '{course_name}' not found.")


def main():
    # Check for Tavily API key
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if not tavily_api_key:
        raise ValueError(
            f"TAVILY_API_KEY not found in environment variables.\n"
            f"Checked .env file at: {env_path}\n"
            f"Please ensure the file exists and contains:\n"
            f"TAVILY_API_KEY=your-actual-key-here"
        )

    # Initialize local Ollama model
    llm = ChatOllama(temperature=0, model="llama3.1:8b")

    # Create Tavily search tool for real-time web search
    tavily_search = TavilySearchResults(
        max_results=3,
        api_key=tavily_api_key,
        description="Search the web for real-time information. Use this when you need current news, facts, or information from the internet."
    )

    # Define available tools
    tools = [get_course_info, tavily_search]

    # Bind tools to LLM
    llm_with_tools = llm.bind_tools(tools)

    # Test queries
    queries = [
        "What is the latest news about AI in December 2025?",
        "Tell me about the Flutter course at Ruby Learner",
        "What is the current price of Bitcoin?"
    ]

    for query in queries:
        print(f"\n{'='*80}")
        print(f"Query: {query}")
        print(f"{'='*80}\n")

        # First call - LLM decides which tool to use
        result = llm_with_tools.invoke([HumanMessage(content=query)])

        # Handle tool calls
        if hasattr(result, 'tool_calls') and result.tool_calls:
            print("🔧 Tools called by LLM:\n")

            tool_messages = []
            for tool_call in result.tool_calls:
                tool_name = tool_call['name']
                tool_args = tool_call['args']
                print(f"  📌 Tool: {tool_name}")
                print(f"     Args: {tool_args}")

                # Execute the tool
                if tool_name == 'get_course_info':
                    tool_result = get_course_info.invoke(tool_args)
                elif tool_name == 'tavily_search_results_json':
                    tool_result = tavily_search.invoke(tool_args)
                else:
                    tool_result = f"Unknown tool: {tool_name}"

                print(f"     Result: {tool_result}\n")

                # Create tool message for next LLM call
                tool_messages.append(
                    ToolMessage(
                        content=str(tool_result),
                        tool_call_id=tool_call['id']
                    )
                )

            # Second call - LLM synthesizes final answer using tool results
            print("💡 Final Answer:\n")
            messages = [HumanMessage(content=query), result] + tool_messages
            final_result = llm.invoke(messages)
            print(final_result.content)

        else:
            print("💬 Direct answer (no tools used):\n")
            print(result.content)

        print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()

