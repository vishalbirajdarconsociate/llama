import os
import re

from langchain.chains import LLMChain
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq


def main():
    groq_api_key = os.environ['GROQ_API_KEY']
    model = 'llama-3.3-70b-versatile'
    groq_chat = ChatGroq(
            groq_api_key=groq_api_key, 
            model_name=model
    )
    

    system_prompt = 'You are a unfriendly, miserable conversational chatbot'
    conversational_memory_length = 5
    memory = ConversationBufferWindowMemory(k=conversational_memory_length, memory_key="chat_history", return_messages=True)

    while True:
        user_question = input("Ask a question: ")
        if user_question:
            prompt = ChatPromptTemplate.from_messages(
                [
                    SystemMessage(
                        content=system_prompt
                    ), 
                    MessagesPlaceholder(
                        variable_name="chat_history"
                    ), 
                    HumanMessagePromptTemplate.from_template(
                        "{human_input}"
                    ),  
                ]
            )
            conversation = LLMChain(
                llm=groq_chat,  
                prompt=prompt, 
                verbose=False,  
                memory=memory, 
            )
            response = conversation.predict(human_input=user_question)
            print("Chatbot:", re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL))

if __name__ == "__main__":
    main()