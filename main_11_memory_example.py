import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_ollama import ChatOllama

load_dotenv(dotenv_path=Path(__file__).parent / '.env')

# Configuration
OLLAMA_MODEL = os.environ.get('OLLAMA_MODEL', 'llama3.1:8b')
MEMORY_K = int(os.environ.get('MEMORY_K', 5))  # Remember last 5 interactions

def main():
    print("LangChain Short-Term Memory Example")
    print("=" * 60)
    print(f"Using Ollama model: {OLLAMA_MODEL}")
    print(f"Memory window size: {MEMORY_K} interactions")
    print()

    # Initialize LLM
    llm = ChatOllama(model=OLLAMA_MODEL, temperature=0.7)

    # Set up memory - ConversationBufferWindowMemory keeps last K interactions
    memory = ConversationBufferWindowMemory(k=MEMORY_K)

    # Create conversation chain with memory
    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=False
    )

    print("Starting conversation with memory...")
    print("Type 'quit' to exit")
    print("-" * 60)

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() == 'quit':
            print("Goodbye!")
            break

        if not user_input:
            continue

        # Get response from chain (memory is automatically updated)
        try:
            response = conversation.run(input=user_input)
            print(f"AI: {response}")
        except Exception as e:
            print(f"Error: {e}")

        print("-" * 60)

    print("\nConversation ended!")
    print(f"Final memory buffer:\n{memory.buffer}")

if __name__ == '__main__':
    main()
