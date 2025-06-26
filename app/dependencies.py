from llm_client import LLMClient as RealLLMClient
from typing import Any

class RedisStub:
    """An async Redis client stub whose ping always succeeds by default."""
    async def ping(self) -> bool:
        return True

# Expose Redis stub (tests may monkey-patch ping to simulate failures)
redis_client = RedisStub()

class DummyLLMClient:
    """A stub LLM client that never fails and returns a predictable response."""
    def chat(self, messages: list[dict[str, str]], **kwargs) -> dict[str, Any]:  # ← 用 Any 而不是 any
        last = messages[-1]["content"] if messages else ""
        return {"choices": [{"message": {"content": f"[stub] {last}"}}]}

    async def stream_chat(self, messages: list[dict[str, str]], **kwargs):
        last = messages[-1]["content"] if messages else ""
        yield f"[stub] {last}"

# Use the dummy client for all routes so that llm_client.chat() never throws
llm_client = DummyLLMClient()

# Export the real class so tests that patch LLMClient still work
LLMClient = RealLLMClient
