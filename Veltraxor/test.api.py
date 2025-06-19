import os
import requests
from dotenv import load_dotenv

# Load all variables in .env from the project root directory
load_dotenv()

# API Key in the interface address and environment variable
url = "http://127.0.0.1:8000/chat"
api_key = os.getenv("XAI_API_KEY")

# Injecting keys via formatted strings
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}
payload = {"prompt": "hello"}

response = requests.post(url, json=payload, headers=headers)

print(f"Status Code: {response.status_code}")
print(f"Raw Text: {response.text}")
try:
    print(f"Parsed JSON: {response.json()}")
except ValueError as decode_error:
    print(f"JSON Decode Error: {decode_error}")
