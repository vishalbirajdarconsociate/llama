import time
import re
from fastapi import FastAPI
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def train_chatbot_llama():
    prompt_template = """{query}"""
    prompt = PromptTemplate(
        input_variables=["context", "query"], template=prompt_template
    )
    llm = ChatGroq(model_name="deepseek-r1-distill-llama-70b", temperature=0.7)
    return prompt | llm


chatbot = train_chatbot_llama()


def remove_think_tags(text):

    pattern = r"<think>.*?</think>"
    return re.sub(pattern, "", text, flags=re.DOTALL)


@app.get("/chat/")
async def chat_endpoint(q: str | None = None):
    start_time = time.time()
    response = chatbot.invoke({"query": q})
    end_time = time.time()
    time_taken = end_time - start_time
    return {"response": remove_think_tags(response.content), "time_taken": time_taken}
