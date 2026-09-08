from chains.general_chain import get_llm, get_conversation_chain
from memory.conversation_memory import get_memory

def main():
    llm = get_llm()
    memory = get_memory()
    chain = get_conversation_chain(llm, memory)

    print("Chatbot ready! Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Bye!")
            break

        response = chain.predict(input=user_input)
        print(f"Bot: {response}\n")

if __name__ == "__main__":
    main()