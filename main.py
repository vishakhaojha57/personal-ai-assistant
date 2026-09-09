from chains.general_chain import get_llm, get_conversation_chain
from chains.academic_rag import build_vector_store, get_academic_chain, academic_query
from memory.conversation_memory import get_memory
from notes.notes_store import save_note , get_all_notes
import os

def main():
    llm = get_llm()
    memory = get_memory()
    general_chain = get_conversation_chain(llm, memory)

    academic_chain = None
    academic_docs_dir = "data/academic_docs"
    os.makedirs(academic_docs_dir, exist_ok=True)
    
    has_pdfs = any(f.endswith('.pdf') for f in os.listdir(academic_docs_dir))

    if has_pdfs:
        print(f"Bot: Found PDFs in {academic_docs_dir}, preparing document store...")
        build_vector_store(academic_docs_dir)
        academic_chain = get_academic_chain(llm)
        print("Bot: Got your documents. I'll answer using them.\n")
    else:
        print("Bot: No academic documents found. Let's chat normally.\n")

    print("Chatbot ready! Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Bye!")
            break

        if user_input.lower().startswith("remember:"):
            note_content = user_input[len("remember:"):].strip()
            if note_content:
                save_note(note_content)
                print("Bot: Got it! I' have saved that to your personal notes.\n")
                continue

        elif user_input.lower() in ["what do you remember?", "show my notes"]:
            notes = get_all_notes()
            print(f"Bot:\n{notes}\n")
            continue




        if academic_chain:
            response = academic_query(academic_chain, user_input)
        else:
            response = general_chain.predict(input=user_input)

        print(f"Bot: {response}\n")

if __name__ == "__main__":
    main()