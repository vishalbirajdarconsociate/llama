import requests
import time

# Base URL of your FastAPI chatbot
BASE_URL = "http://127.0.0.1:8000"  # Update with your actual API URL if different

# List of test questions
questions = [
    "What is the price of Bioderma Sebium Pore Refiner?",
    "Can you tell me the uses of Minimalist Vitamin C Face Serum?",
    "What are the side effects of CETAPHIL BRIGHT HEALTHY RADIANCE BRIGHTENING NIGHT COMFORT Cream?",
    "Does the bot know about Garnier Skin Black Serum Mask?",
    "What is the composition of Dabur Shilajit Gold Capsule?",
    "Are there any fitness-related products?",
    "Which products fall under 'Personal Care' category?",
    "Tell me about COQ 300mg Softgel.",
    "What are the reviews for Pentasure 2.0 Vanilla Powder?",
]

# Test function to call the chatbot API
def test_chatbot(questions):
    total_time_start = time.time()  # Start total test time
    total_query_time = 0
    results = []
    
    for question in questions:
        start_time = time.time()
        response = requests.get(f"{BASE_URL}/chat/", params={"q": question})
        end_time = time.time()
        query_time = end_time - start_time
        total_query_time += query_time

        if response.status_code == 200:
            result = response.json()
            results.append({
                "question": question,
                "response": result.get("response", "No response"),
                "time_taken_seconds": result.get("time_taken_seconds", query_time)
            })
        else:
            results.append({
                "question": question,
                "error": f"Error {response.status_code}: {response.text}"
            })
    
    total_time_end = time.time()  # End total test time
    total_test_time = total_time_end - total_time_start
    average_time_per_query = total_query_time / len(questions)
    
    return results, average_time_per_query, total_test_time

# Run the test
if __name__ == "__main__":
    print("Testing chatbot with sample questions...")
    test_results, avg_time, total_time = test_chatbot(questions)
    
    # Print the results
    for result in test_results:
        print(f"Question: {result['question']}")
        if "response" in result:
            print(f"Response: {result['response']}")
            print(f"Time Taken: {result['time_taken_seconds']} seconds\n")
        else:
            print(f"Error: {result['error']}\n")
    
    # Print summary statistics
    print(f"Average Time Per Query: {avg_time:.2f} seconds")
    print(f"Total Test Time: {total_time:.2f} seconds")
