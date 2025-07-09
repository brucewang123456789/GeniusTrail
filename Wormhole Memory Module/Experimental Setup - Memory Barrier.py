class MemoryBarrier:
    def __init__(self):
        self.memory_store = {}
        self.active_conversation = None

    def switch_conversation(self, conversation_id):
        """
        Switch to a specific conversation by setting it as active.
        """
        self.active_conversation = conversation_id
        if conversation_id not in self.memory_store:
            self.memory_store[conversation_id] = []

    def add_memory(self, conversation_id, memory):
        """
        Add memory to a specific conversation.
        """
        if self.active_conversation != conversation_id:
            raise PermissionError("No active conversation. Use 'switch_conversation' to activate one.")
        self.memory_store[conversation_id].append(memory)

    def retrieve_memory(self, conversation_id):
        """
        Retrieve memory from a specific conversation.
        """
        if self.active_conversation != conversation_id:
            raise PermissionError("No active conversation. Use 'switch_conversation' to activate one.")
        return self.memory_store.get(conversation_id, [])

    def clear_memory(self, conversation_id):
        """
        Clear all memory from a specific conversation.
        """
        if self.active_conversation != conversation_id:
            raise PermissionError("No active conversation. Use 'switch_conversation' to activate one.")
        self.memory_store[conversation_id] = []

# Example usage for testing
if __name__ == "__main__":
    memory_barrier = MemoryBarrier()

    # Switch to Conversation 1 and add memory
    memory_barrier.switch_conversation("Conversation 1")
    memory_barrier.add_memory("Conversation 1", "This is a memory for Conversation 1.")

    # Retrieve memory from Conversation 1
    print("Conversation 1 Memory:", memory_barrier.retrieve_memory("Conversation 1"))

    # Switch to Conversation 2 and add memory
    memory_barrier.switch_conversation("Conversation 2")
    memory_barrier.add_memory("Conversation 2", "This is a memory for Conversation 2.")

    # Retrieve memory from Conversation 2
    print("Conversation 2 Memory:", memory_barrier.retrieve_memory("Conversation 2"))

    # Try to retrieve memory from an inactive conversation (should raise an error)
    try:
        print("Conversation 1 Memory (Inactive):", memory_barrier.retrieve_memory("Conversation 1"))
    except PermissionError as e:
        print("Error:", e)

    # Clear memory from Conversation 2
    memory_barrier.clear_memory("Conversation 2")
    print("Conversation 2 Memory after clearing:", memory_barrier.retrieve_memory("Conversation 2"))
