from fastapi import FastAPI
from pydantic import BaseModel

# Creating a FastAPI instance
app = FastAPI()


# Defining the request body data model
class ChatRequest(BaseModel):
    input: str


# POST route, receiving chat request
@app.post("/chat")
async def chat(request: ChatRequest):
    return {"message": f"Received input: {request.input}"}


# Visit http://127.0.0.1:8000/docs to view the automatically generated documentation
