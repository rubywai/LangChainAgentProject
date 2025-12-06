from dotenv import load_dotenv
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

@tool
def do_print(message: str):
    """Print a message to the console."""
    print(f"The function is calling {message}")


def main():
    llm = ChatOllama(temperature=0, model="llama3.1:8b")
    tools = [do_print]
    llm_with_tools = llm.bind_tools(tools)
    result = llm_with_tools.invoke([HumanMessage(content="Hello, please print a message using the tool.")])

    if hasattr(result, 'tool_calls') and result.tool_calls:
        print(f"Tool calls found: {result.tool_calls}")
        for tool_call in result.tool_calls:
            if tool_call['name'] == 'do_print':
                do_print.invoke(tool_call['args'])
    else:
        print(f"No tool calls. Content: {result.content}")

if __name__ == "__main__":
    main()
