import requests

url = "http://127.0.0.1:8000/chat"
headers = {"Content-Type": "application/json"}
payload = {
    "prompt": "Hello",
    "history": [
        {"role": "user", "content": "Hi"}
    ]
}

response = requests.post(url, json=payload)
print("Status code:", response.status_code)
print("Response:", response.text)
