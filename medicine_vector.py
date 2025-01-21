"""
* * With Hugging Face Embeddings and pickle file (not recommended)
"""



import pandas as pd
import time
from fastapi import FastAPI
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document
import requests
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def fetch_product_api_data():
    api_url = "http://62.72.56.145:5154/api/product_list/"
    try:
        payload = {"jsonrpc": "2.0", "params": {"page_no": 1, "page_size": 200}}
        headers = {"Content-Type": "application/json"}
        response = requests.post(api_url, json=payload, headers=headers)
        response.raise_for_status()
        products = response.json()
        data = []
        for product in products["result"]["products"]:
            name = product.get("name", "Unknown").replace("\n", "")
            price = product.get("uom_list", [{"sales_price": "N/A"}])[0]["sales_price"]
            cat = product.get("category", "Unknown")
            data.append(f"Product Name: {name}, Price: ${price}, Category: {cat}")
        print(data)
        return data
    except requests.exceptions.RequestException as e:
        return ["Error fetching product data."]

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



def create_vector_store(product_data):
    embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    documents = [Document(page_content=desc) for desc in product_data]
    vector_store = FAISS.from_documents(documents, embedder)
    vector_store.save_local("faiss_product_store")
    return vector_store


def fetch_relevant_product_data(query):
    vector_store = FAISS.load_local(
        "faiss_product_store",
        HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
        allow_dangerous_deserialization=True
    )
    retriever = vector_store.as_retriever()
    relevant_docs = retriever.invoke(query)
    return "\n".join([doc.page_content for doc in relevant_docs])


def train_chatbot_llama():
    prompt_template = """
You are an intelligent assistant for an e-commerce platform specializing in medicinal products. Your role is to provide accurate and helpful answers to user queries about medicines and related products, based solely on the information provided below:

Product Data:
{context}

User Query:
{query}

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
    prompt = PromptTemplate(input_variables=["context", "query"], template=prompt_template)
    llm = ChatOllama(
        model="llama3.2:1b",
        temperature=0,
    )
    return prompt | llm


# product_data = fetch_product_api_data()
product_data = fetch_product_csv_data("/home/vishal/Desktop/opencv/chatbot/llama/Medicine_Details.csv")
vector_store = create_vector_store(product_data)
chatbot = train_chatbot_llama()


@app.get("/chat/")
async def chat_endpoint(q: str | None = None):
    start_time = time.time()

    relevant_data = fetch_relevant_product_data(q)
    response = chatbot.invoke({"context": relevant_data, "query": q})
    end_time = time.time()
    time_taken = end_time - start_time
    return {"response": response.content, "time_taken": time_taken}


@app.get("/update-product-data/")
async def update_product_data():
    start_time = time.time()
    product_data = fetch_product_api_data()
    if "Error fetching product data." in product_data:
        return {"error": "Failed to fetch product data from API."}
    create_vector_store(product_data)
    end_time = time.time()
    time_taken = end_time - start_time
    return {
        "message": "Product data updated successfully.",
        "time_taken_seconds": time_taken,
    }
