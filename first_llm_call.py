import os
import requests

# Hugging Face API endpoint and token
HF_API_URL = "https://api-inference.huggingface.co/models/mixtral-8x7b-instruct-v0.1"  # Replace with desired model
HF_TOKEN = os.environ.get("HF_TOKEN")  # Ensure HF_TOKEN is set in your environment

# Headers for authentication
headers = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

# Function to query the model
def query_huggingface(prompt):
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 100,  # Adjust as needed
            "return_full_text": False
        }
    }
    response = requests.post(HF_API_URL, headers=headers, json=payload)
    
    if response.status_code == 200:
        return response.json()[0]["generated_text"]
    else:
        raise Exception(f"Error: {response.status_code}, {response.text}")

# Example usage
try:
    prompt = "Hello, how can I assist you today?"  # Replace with your prompt
    response = query_huggingface(prompt)
    print(response)
except Exception as e:
    print(e)