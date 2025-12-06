"""
Example using Tavily search with structured responses - similar to the reference example.
This searches for job postings and returns structured data with sources.
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


class AgentResponse(BaseModel):
    """Schema for agent response with answer and sources"""
    answer: str = Field(description="The agent's answer to the query")
    sources: List[Source] = Field(
        default_factory=list, description="List of sources used to generate the answer"
    )


def main():
    print("🚀 Hello from langchain-course!\n")
    print("Initializing Ollama (local) and Tavily search agent...\n")

    # Initialize local Ollama model (instead of OpenAI gpt-5)
    llm = ChatOllama(temperature=0, model="llama3.1:8b")

    # Create Tavily search tool
    tavily_search = TavilySearch(max_results=3)

    # Bind tools to LLM
    llm_with_tools = llm.bind_tools([tavily_search])

    # Query similar to the reference example
    query = "Search for 3 job postings for an AI engineer using langchain in the bay area on linkedin and list their details"

    print(f"❓ Query: {query}\n")
    print("="*80 + "\n")

    # Step 1: LLM decides to use search tool
    print("🔍 Agent is searching for job postings...\n")
    result = llm_with_tools.invoke([HumanMessage(content=query)])

    # Step 2: Execute tool calls
    sources = []
    tool_messages = []

    if hasattr(result, 'tool_calls') and result.tool_calls:
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
                        sources.append(Source(url=item.get('url', '')))

            # Create tool message for next LLM call
            tool_messages.append(
                ToolMessage(
                    content=str(search_results),
                    tool_call_id=tool_call['id']
                )
            )

    # Step 3: Get structured response
    print("💡 Generating structured response...\n")

    structured_llm = llm.with_structured_output(AgentResponse)

    # Format sources for prompt
    sources_text = "\n".join([f"- {s.url}" for s in sources])

    final_messages = [HumanMessage(content=query), result] + tool_messages
    final_messages.append(HumanMessage(content=f"""
Based on the search results above, provide a comprehensive answer listing the job posting details.

Important: 
1. In the "answer" field, list each job posting with its details (company, title, location, etc.)
2. In the "sources" field, include the URLs where these jobs were found

Available source URLs:
{sources_text}

Respond with structured JSON containing "answer" and "sources" fields.
"""))

    # Get structured response
    structured_response = structured_llm.invoke(final_messages)

    print("="*80)
    print("\n📊 RESULT:\n")
    print(f"Answer:\n{structured_response.answer}\n")

    print("\nSources:")
    if structured_response.sources:
        for idx, source in enumerate(structured_response.sources, 1):
            print(f"  {idx}. {source.url}")
    else:
        # Fallback
        for idx, source in enumerate(sources[:3], 1):
            print(f"  {idx}. {source.url}")

    print("\n" + "="*80)
    print("\n✅ Done!")


if __name__ == "__main__":
    main()

