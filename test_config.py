from config import settings

if __name__ == "__main__":
    loaded = bool(settings.XAI_API_KEY)
    print(f"API key loaded: {loaded}")
