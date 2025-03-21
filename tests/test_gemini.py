import os

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]

import requests
import json

# Use the API key from environment variable
url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

headers = {
    "Content-Type": "application/json"
}

data = {
    "contents": [{
        "parts": [{"text": "Explain how AI works"}]
    }]
}

response = requests.post(url, headers=headers, json=data)
print(response.json())