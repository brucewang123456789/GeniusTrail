import os
import requests

TOKEN = os.getenv("AUTH_HEADER", "Bearer real_valid_token").split(" ", 1)[1]
URL = "http://localhost:8000/chat"
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {TOKEN}",
}
data = {"messages": [{"role": "user", "content": "ping"}]}

resp = requests.post(URL, json=data, headers=headers)
print("Status:", resp.status_code)
print("Response:", resp.text)
