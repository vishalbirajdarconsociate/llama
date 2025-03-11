"""
* * Without embedings and no pickle file  
"""

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from langchain.prompts import PromptTemplate
from langchain_ollama import ChatOllama

app = FastAPI()

import requests



def fetch_product_csv_data(file_path):
    try:
        df = pd.read_csv(file_path)
        required_columns = [
            "Medicine Name", "Composition", "Uses", "Side_effects", 
            "Image URL", "Manufacturer", "Excellent Review %", 
            "Average Review %", "Poor Review %"
        ]
        if not all(col in df.columns for col in required_columns):
            raise ValueError("CSV file is missing one or more required columns.")
        data = []
        for _, row in df.iterrows():
            medicine_name = row["Medicine Name"]
            composition = row["Composition"]
            uses = row["Uses"]
            side_effects = row["Side_effects"]
            manufacturer = row["Manufacturer"]
            reviews = f"Excellent: {row['Excellent Review %']}%, Average: {row['Average Review %']}%, Poor: {row['Poor Review %']}%"
            formatted = (
                f"Medicine Name: {medicine_name}, Composition: {composition}, "
                f"Uses: {uses}, Side Effects: {side_effects}, Manufacturer: {manufacturer}, "
                f"Reviews: {reviews}"
            )
            print(formatted)
            data.append(formatted)
        return data
    except FileNotFoundError:
        return ["Error: CSV file not found."]
    except ValueError as ve:
        return [f"Error: {str(ve)}"]
    except Exception as e:
        return [f"Error: {str(e)}"]



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


product_data = fetch_product_csv_data("/home/vishal/Desktop/opencv/chatbot/llama/Medicine_Details.csv")
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
