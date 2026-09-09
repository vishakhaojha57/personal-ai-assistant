from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import FileChatMessageHistory
import os

HISTORY_FILE = "history/chat_history.json"
def get_memory():
    os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)

    message_history = FileChatMessageHistory(file_path=HISTORY_FILE)
    
    return ConversationBufferMemory(
        memory_key="chat_history",
        chat_memory=message_history, 
        return_messages=True
    )

    
