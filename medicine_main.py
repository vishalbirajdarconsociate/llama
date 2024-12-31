from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from langchain.prompts import PromptTemplate
from sqlalchemy import create_engine, text
from langchain_ollama import ChatOllama

app = FastAPI()

import requests


def fetch_product_api_data():
    api_url = "http://62.72.56.145:5154/api/product_list/"
    try:
        payload = {"jsonrpc": "2.0", "params": {"page_no": 1, "page_size": 200}}
        headers = {
            "Content-Type": "application/json",
        }
        response = requests.post(api_url, json=payload, headers=headers)
        response.raise_for_status()
        products = response.json()
        data = ""
        for product in products["result"]["products"]:
            name = product.get("name", "Unknown").replace("\n", "")
            price = product.get("uom_list", 0)[0]["sales_price"]
            cat = product.get("category", "Unknown")
            data += f"Product Name: {name}, Price: ${price}, Category: {cat};\n"
        return data
    except requests.exceptions.RequestException as e:
        return "Error fetching product data."


def train_chatbot_llama(product_data):
    prompt_template = f"""
You are an intelligent assistant for an e-commerce platform specializing in medicinal products. Your role is to provide accurate and helpful answers to user queries about medicines and related products, based solely on the information provided below:

Product Data:
{product_data}

User Query:
{{query}}

Guidelines:
1. Use the provided product data to craft your responses. Do not include any information that is not present in the product data.
2. Be concise yet informative. If the query cannot be answered using the product data, respond with: 
    "I'm sorry, I can only answer queries based on the available product data." Otherwise, provide a direct and relevant answer based on the product data.
3. Maintain a professional and empathetic tone, as users may be inquiring about health-related products.
4. Do not provide medical advice, dosage recommendations, or warnings unless explicitly stated in the product data. Instead, recommend consulting a healthcare professional for such information.
5. Do not introduce yourself; just answer the query directly.
6. Highlight any key product details, such as its purpose, active ingredients, or availability, when relevant.

Now, provide a response to the query:
    """
    print(prompt_template)
    prompt = PromptTemplate(input_variables=["query"], template=prompt_template)
    llm = ChatOllama(
        model="llama3.2:1b",
        temperature=0,
    )
    chain = prompt | llm
    print("Model is done")
    return chain


product_data = fetch_product_api_data()
chatbot = train_chatbot_llama(product_data)


class Query(BaseModel):
    query: str


@app.post("/chat-get")
async def chat(query: Query):
    try:
        response = chatbot.invoke(query.query)
        return {"response": response.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.get("/chat/")
async def read_items(q: str | None = None):
    response = chatbot.invoke(q)
    return {"response": response.content}
