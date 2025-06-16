from llm_client import LLMClient

if __name__ == "__main__":
    # Replace "gpt-3.5-turbo" or your chosen model name
    client = LLMClient(model="gpt-3.5-turbo")
    # Simple test message
    messages = [{"role": "user", "content": "Hello"}]
    try:
        resp = client.chat(messages)
        # Expect a JSON with 'choices' etc.
        print("Success:", resp.get("choices")[0].get("message").get("content"))
    except Exception as e:
        print("Error:", e)
