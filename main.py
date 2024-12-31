from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.prompts import PromptTemplate
from sqlalchemy import create_engine, text
from langchain_ollama import ChatOllama

app = FastAPI()


def fetch_product_data():
    engine = create_engine("postgresql://vishal:vishal@localhost:5432/jumla-prod")    # local
    query = "SELECT name, list_price FROM product_template  LIMIT 10"
    with engine.connect() as connection:
        result = connection.execute(text(query))
        data = ""
        for i in result:
            data += f"Product Name: {i[0]['en_US']},Price: ${i[1] or 0};\n"
        print(data)
    return data

import requests

def fetch_product_api_data():
    api_url = "http://195.35.6.131:5154/api/product_list/"
    try:
        # If the API requires a body, define it here
        payload = {
    "jsonrpc": "2.0",
    "params": {
        "page_no": 1,
        "page_size": 100
    }
}
        headers = {
            "Content-Type": "application/json",
        }
        response = requests.post(api_url, json=payload, headers=headers)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        # Assuming the API returns a JSON response
        products = response.json()  # Parse the JSON response
        
        data = ""
        for product in products['result']['products']:  
            name = product.get('name', 'Unknown')
            price = product.get('uom_list', 0)[0]['sales_price']
            cat = product.get('category', 'Unknown')
            data += f"Product Name: {name}, Price: ${price}, Category: {cat};\n"
        
        print(data)  # Print the data for debugging purposes
        return data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from API: {e}")
        return "Error fetching product data."

def train_chatbot_llama(product_data):
    prompt_template = f"""
    You are an intelligent assistant for an e-commerce platform. Your role is to provide accurate and helpful answers to user queries about products, based solely on the information provided below:

    Product Data:
    {product_data}

    User Query:
    {{query}}

    Guidelines:
    1. Use the provided product data to craft your responses. Do not include any information that is not present in the product data.
    2. Be concise yet informative. If the product data does not contain enough information to answer the query, respond with: 
    "I'm sorry, I can only answer queries based on the available product data."
    3. Maintain a friendly and professional tone in your responses.
    4. Do not introduce yourself , just answer the query

    Now, provide a response to the query:
    """
    prompt = PromptTemplate(input_variables=["query"], template=prompt_template)
    llm = ChatOllama(
    model="llama3.2:1b",
    temperature=0,
    )
    chain = prompt | llm
    print("Model is done")
    return chain


# product_data = fetch_product_data()
product_data = fetch_product_api_data()
chatbot = train_chatbot_llama(product_data)


class Query(BaseModel):
    query: str


@app.post("/chat")
async def chat(query: Query):
    try:
        response = chatbot.invoke(query.query)
        return {"response": response.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
