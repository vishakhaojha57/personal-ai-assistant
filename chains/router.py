from langchain_core.prompts import PromptTemplate
from chains.general_chain import get_llm


ROUTER_PROMPT = """You are a smart routing assistant. Analyze the user's input and classify it into EXACTLY ONE of these categories:
1. ACADEMIC: The user is asking a question that requires searching through their uploaded PDFs, documents, or study materials.
2. NOTES: The user is asking you to remember something, save a personal note, or asking about what you remember from their personal notes.
3. GENERAL: The user is just chatting normally, saying hi, or asking a general knowledge question.
Output ONLY the category name (ACADEMIC, NOTES, or GENERAL) and nothing else.
User Input: {user_input}
Category:"""

def get_router_chain():
    llm = get_llm()
    prompt = PromptTemplate.from_template(ROUTER_PROMPT)
    router_chain = prompt | llm
    return router_chain

def classify_input(router_chain , user_input : str) -> str:
    response = router_chain.invoke({"user_input" : user_input})

    category = response.content.strip().upper()
    if category not in ["ACADEMIC", "NOTES", "GENERAL"]:
        return "GENERAL"

    return category
