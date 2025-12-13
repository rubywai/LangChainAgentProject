from langchain.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage
from langchain_ollama import ChatOllama


# Static course database
COURSES_DB = {
    "Flutter": {
        "name": "Flutter",
        "description": "Cross-platform mobile app development with Flutter",
        "instructor": "Ruby Learner",
        "duration": "12 weeks",
        "students": 45
    },
    "Kotlin": {
        "name": "Kotlin",
        "description": "Android app development with Kotlin",
        "instructor": "Ruby Learner",
        "duration": "10 weeks",
        "students": 32
    }
}


@tool
def get_course_info(course_name: str) -> str:
    """Get detailed information about a course. Use this when the user asks about course details.

    Args:
        course_name: The name of the course (e.g., 'Flutter', 'Kotlin')
    """
    course = COURSES_DB.get(course_name)
    if course:
        return f"Course: {course['name']}\nDescription: {course['description']}\nInstructor: {course['instructor']}\nDuration: {course['duration']}"
    return f"Course '{course_name}' not found."


@tool
def get_student_count(course_name: str) -> str:
    """Get the number of students enrolled in a course.

    Args:
        course_name: The name of the course (e.g., 'Flutter', 'Kotlin')
    """
    course = COURSES_DB.get(course_name)
    if course:
        return f"The {course_name} course has {course['students']} students enrolled."
    return f"Course '{course_name}' not found."


def main():
    # Initialize local Ollama model
    llm = ChatOllama(temperature=0, model="llama3.1:8b")

    # Define available tools
    tools = [get_course_info, get_student_count]

    # Bind tools to LLM with auto mode (model decides when to use tools)
    llm_with_tools = llm.bind_tools(tools, tool_choice="auto")

    # Test query
    query = "Hello Sir, tell me more about Flutter course and how many students are enrolled?"
    print(f"Query: {query}\n")

    # Step 1: Initial message history with system instructions and user query
    messages = [
        SystemMessage(content="You are a helpful AI assistant that can answer any question. You have some tools available for getting course information, but you should answer general knowledge questions directly without using tools. Use tools ONLY when asked about course details or student counts."),
        HumanMessage(content=query)
    ]

    # Step 2: Get AI response (may contain tool calls)
    ai_message = llm_with_tools.invoke(messages)
    messages.append(ai_message)

    # Step 3: Check if AI wants to use tools
    if hasattr(ai_message, 'tool_calls') and ai_message.tool_calls:
        print("🔧 Tools used:")

        # Create a mapping of tool names to tool functions
        tool_map = {
            'get_course_info': get_course_info,
            'get_student_count': get_student_count
        }

        # Execute each tool and collect results as ToolMessages
        for tool_call in ai_message.tool_calls:
            tool_name = tool_call['name']
            tool_args = tool_call['args']
            tool_call_id = tool_call['id']

            print(f"  - {tool_name}({tool_args})")

            # Execute the tool
            tool_function = tool_map.get(tool_name)
            if tool_function:
                tool_result = tool_function.invoke(tool_args)
                print(f"    ✓ Result: {tool_result}")

                # Create ToolMessage with the result
                tool_message = ToolMessage(
                    content=tool_result,
                    tool_call_id=tool_call_id
                )
                messages.append(tool_message)

        # Step 4: Send tool results back to LLM for final answer
        print("\n🤖 Generating final answer...\n")
        final_response = llm_with_tools.invoke(messages)

        print("=" * 60)
        print("Final Answer:")
        print("=" * 60)
        print(final_response.content)
        print("=" * 60)
    else:
        # No tool calls needed, use direct response
        print("=" * 60)
        print("Direct Answer (no tools needed):")
        print("=" * 60)
        print(ai_message.content)
        print("=" * 60)


if __name__ == "__main__":
    main()
