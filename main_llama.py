from fastapi import FastAPI
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain.docstore.document import Document
import requests
import time
import pickle

app = FastAPI()


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
            print(f"Product Name: {name}, Price: ${price}, Category: {cat}")
            data.append(f"Product Name: {name}, Price: ${price}, Category: {cat}")
        return data
    except requests.exceptions.RequestException as e:
        return ["Error fetching product data."]


def create_and_save_vector_store(product_data):
    embedder = OllamaEmbeddings(model = "llama3.2:3b")
    documents = [Document(page_content=desc) for desc in product_data]
    vector_store = FAISS.from_documents(documents, embedder)
    with open("faiss_product_store.pkl", "wb") as f:
        pickle.dump(vector_store, f)


def load_vector_store():
    with open("faiss_product_store.pkl", "rb") as f:
        vector_store = pickle.load(f)
    return vector_store


def fetch_relevant_product_data(query):
    vector_store = load_vector_store()
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
    prompt = PromptTemplate(
        input_variables=["context", "query"], template=prompt_template
    )
    llm = ChatOllama(
        model = "llama3.2:3b",
        temperature=0,
    )
    return prompt | llm


product_data = fetch_product_api_data()
create_and_save_vector_store(product_data)
chatbot = train_chatbot_llama()


@app.get("/chat/")
async def chat_endpoint(q: str):
    start_time = time.time()
    relevant_data = fetch_relevant_product_data(q)
    response = chatbot.invoke({"context": relevant_data, "query": q})
    end_time = time.time()
    return {"response": response.content, "time_taken_seconds": end_time - start_time}


@app.get("/update-product-data/")
async def update_product_data():
    start_time = time.time()
    product_data = fetch_product_api_data()
    if "Error fetching product data." in product_data:
        return {"error": "Failed to fetch product data from API."}
    create_and_save_vector_store(product_data)
    end_time = time.time()
    time_taken = end_time - start_time
    return {
        "message": "Product data updated successfully.",
        "time_taken_seconds": time_taken,
    }
