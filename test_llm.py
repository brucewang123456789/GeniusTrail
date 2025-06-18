from llm_client import LLMClient

def main():
    client = LLMClient(model="grok-3-latest")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",   "content": "Hello, what can you do?"}
    ]

    try:
        result = client.chat(messages)
        assistant_msg = result["choices"][0]["message"]["content"]
        print("Success:", assistant_msg)
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    main()
