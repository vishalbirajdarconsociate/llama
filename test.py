import requests
import time

BASE_URL = "http://127.0.0.1:8000"  
questions = [
    "What is the price of Bioderma Sebium Pore Refiner?",
    "Can you tell me the uses of Minimalist Vitamin C Face Serum?",
    "What are the side effects of CETAPHIL BRIGHT HEALTHY RADIANCE BRIGHTENING NIGHT COMFORT Cream?",
    "Does the bot know about Garnier Skin Black Serum Mask?",
    "What is the composition of Dabur Shilajit Gold Capsule?",
    "Are there any fitness-related products?",
    "Which products fall under 'Personal Care' category?",
    "Tell me about COQ 300mg Softgel.",
    "Is there a gift card available?",
    "What are the reviews for Pentasure 2.0 Vanilla Powder?",
]

def test_chatbot(questions):
    results = []
    for question in questions:
        start_time = time.time()
        response = requests.get(f"{BASE_URL}/chat/", params={"q": question})
        end_time = time.time()
        if response.status_code == 200:
            result = response.json()
            results.append({
                "question": question,
                "response": result.get("response", "No response"),
                "time_taken_seconds": result.get("time_taken_seconds", end_time - start_time)
            })
        else:
            results.append({
                "question": question,
                "error": f"Error {response.status_code}: {response.text}"
            })
    
    return results

if __name__ == "__main__":
    print("Testing chatbot with sample questions...")
    test_results = test_chatbot(questions)
    
    for result in test_results:
        print(f"Question: {result['question']}")
        if "response" in result:
            print(f"Response: {result['response']}")
            print(f"Time Taken: {result['time_taken_seconds']} seconds\n")
        else:
            print(f"Error: {result['error']}\n")
