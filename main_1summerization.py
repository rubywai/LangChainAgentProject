from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()


def main():
    information = """
    Ruby Learner in Myanmar offers online tech training, primarily focusing on Flutter and Kotlin for app development, with video lessons available on YouTube, teaching local learners how to build mobile apps in Burmese. 
Key Offerings & Platform:
Focus: Mobile development with Flutter (cross-platform) and Kotlin (Android).
Format: Video-based courses, accessible through a dedicated YouTube playlist.
Language: Lessons are in Burmese, making tech education accessible in Myanmar. 
"""


    summary_template = """
1. Give a short summary of Ruby Learner using the information: {information}
2. Provide two important facts about it.
"""

    summary_prompt = PromptTemplate(
        input_variables=["information"],
        template=summary_template,
    )

    # Choose a model available to you (change if needed)
    llm = ChatOpenAI(temperature=1, model="gpt-3.5-turbo")

    # Use modern LCEL syntax instead of deprecated LLMChain
    chain = summary_prompt | llm

    # Run the chain and print the response
    try:
        result = chain.invoke({"information": information})
        print(result.content)
    except Exception as e:
        # Print a helpful error message but keep the script from crashing silently
        print("Failed to run LLM chain:", str(e))


if __name__ == "__main__":
    main()
