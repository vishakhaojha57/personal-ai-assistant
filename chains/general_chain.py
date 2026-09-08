from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from config import GOOGLE_API_KEY , MODEL_NAME , TEMPERATURE
from langchain_classic.chains import ConversationChain


SYSTEM_PROMPT = """You are a helpful personal assistant for a  student.
You answer both academic and general questions clearly and concisely.
Keep responses friendly and to the point."""


def get_llm():
    return ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key = GOOGLE_API_KEY,
        temperature = TEMPERATURE
    )

def get_conversation_chain(llm, memory):
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{input}")
    ])

    return ConversationChain(
        llm=llm,
        memory=memory,
        prompt=prompt,
        verbose=False
    )

    
