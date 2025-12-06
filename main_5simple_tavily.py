"""
Simple example of using Tavily for real-time web search with Ollama.
This example shows how to get current information from the internet with structured responses.
"""
from typing import List
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from pathlib import Path

# Load .env file
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch


class Source(BaseModel):
    """Schema for a source used by the agent"""
    url: str = Field(description="The URL of the source")
    title: str = Field(description="The title of the source")


class AgentResponse(BaseModel):
    """Schema for agent response with answer and sources"""
    answer: str = Field(description="The agent's answer to the query")
    sources: List[Source] = Field(
        default_factory=list, description="List of sources used to generate the answer"
    )


def main():
    print("🚀 Initializing Ollama and Tavily search agent...\n")

    # Initialize local Ollama model
    llm = ChatOllama(temperature=0, model="llama3.1:8b")

    # Create Tavily search tool
    tavily_search = TavilySearch(max_results=3)

    # Bind tools to LLM
    llm_with_tools = llm.bind_tools([tavily_search])

    # Example query - you can change this to any real-time question
    query = "What are the latest news about AI in December 2025?"

    print(f"❓ Query: {query}\n")
    print("="*80 + "\n")

    # Step 1: LLM decides which tools to use
    print("🔍 Agent is thinking and searching...\n")
    result = llm_with_tools.invoke([HumanMessage(content=query)])

    # Step 2: Execute tool calls if any
    sources = []
    tool_messages = []

    if hasattr(result, 'tool_calls') and result.tool_calls:
        print(f"📌 Agent is using {len(result.tool_calls)} tool(s)\n")

        for tool_call in result.tool_calls:
            # Execute Tavily search - only pass the 'query' parameter
            tool_args = tool_call['args']
            # Filter to only include valid parameters
            search_query = tool_args.get('query', '') if isinstance(tool_args, dict) else str(tool_args)
            search_results = tavily_search.invoke({'query': search_query})

            # Extract sources
            if isinstance(search_results, list):
                for item in search_results:
                    if isinstance(item, dict) and 'url' in item:
                        sources.append(Source(
                            url=item.get('url', ''),
                            title=item.get('title', 'Untitled')
                        ))
                        print(f"   📄 {item.get('title', 'Untitled')}")
                        print(f"      {item.get('url', '')}\n")

            # Create tool message for next LLM call
            tool_messages.append(
                ToolMessage(
                    content=str(search_results),
                    tool_call_id=tool_call['id']
                )
            )

    # Step 3: LLM synthesizes final answer with structured output
    print("="*80 + "\n")
    print("💡 Generating structured response...\n")

    # Use structured output with Pydantic model
    structured_llm = llm.with_structured_output(AgentResponse)

    # Format sources for the prompt
    sources_text = "\n".join([f"- {s.title} ({s.url})" for s in sources])

    # Create final prompt with tool results
    final_messages = [HumanMessage(content=query), result] + tool_messages
    final_messages.append(HumanMessage(content=f"""
Based on the search results above, provide a comprehensive answer to the query: "{query}"

Important: 
1. Write a clear, concise answer in the "answer" field.
2. List ALL the sources you used in the "sources" field with their exact URLs and titles.

Available sources from search:
{sources_text}

Respond with a structured JSON containing "answer" and "sources" fields.
"""))

    # Get structured response
    try:
        structured_response = structured_llm.invoke(final_messages)

        print("="*80)
        print("\n📊 STRUCTURED RESPONSE:\n")
        print(f"Answer:\n{structured_response.answer}\n")

        # Always show sources (either from structured response or from search)
        print("Sources:")
        if structured_response.sources and len(structured_response.sources) > 0:
            for idx, source in enumerate(structured_response.sources, 1):
                print(f"  {idx}. {source.title}")
                print(f"     {source.url}\n")
        else:
            # Fallback to sources extracted from search
            for idx, source in enumerate(sources, 1):
                print(f"  {idx}. {source.title}")
                print(f"     {source.url}\n")

    except Exception as e:
        print(f"❌ Error generating structured response: {e}")
        print("\nFallback - Showing search results:")
        for idx, source in enumerate(sources, 1):
            print(f"  {idx}. {source.title}")
            print(f"     {source.url}\n")



if __name__ == "__main__":
    main()

