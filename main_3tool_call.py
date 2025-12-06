from langchain.tools import tool
from langchain_core.messages import HumanMessage
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

    # Bind tools to LLM
    llm_with_tools = llm.bind_tools(tools)

    # Test query
    query = "Tell me about the Flutter course and how many students it has."
    print(f"Query: {query}\n")

    result = llm_with_tools.invoke([HumanMessage(content=query)])

    print("Answer: \n")

    # Handle tool calls
    if hasattr(result, 'tool_calls') and result.tool_calls:
        print("Tools used:")
        for tool_call in result.tool_calls:
            tool_name = tool_call['name']
            tool_args = tool_call['args']
            print(f"  - {tool_name}: {tool_args}")

            # Execute the tool
            if tool_name == 'get_course_info':
                tool_result = get_course_info.invoke(tool_args)
                print(f"    Result: {tool_result}")
            elif tool_name == 'get_student_count':
                tool_result = get_student_count.invoke(tool_args)
                print(f"    Result: {tool_result}")
    else:
        print(f"No tool calls. Direct answer: {result.content}")


if __name__ == "__main__":
    main()
