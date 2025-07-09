class MemoryBarrier:
    def __init__(self, dataset):
        """
        Initializes the memory barrier with dataset partitioning.
        :param dataset: A list of data entries where each entry represents a conversation.
        """
        self.conversations = {}
        for i, entry in enumerate(dataset):
            self.conversations[f"conversation_{i+1}"] = entry
        self.current_conversation = None

    def open_conversation(self, conversation_id):
        """
        Switch to a specific conversation.
        :param conversation_id: ID of the conversation to open.
        """
        if conversation_id in self.conversations:
            self.current_conversation = conversation_id
            print(f"Switched to {conversation_id}")
        else:
            print(f"Error: {conversation_id} does not exist.")

    def retrieve_memory(self):
        """
        Retrieve memory from the current conversation.
        """
        if self.current_conversation:
            memory = self.conversations[self.current_conversation]
            print(f"Retrieved memory from {self.current_conversation}: {memory}")
            return memory
        else:
            print("Error: No conversation is currently open.")
            return None

# Example dataset (CoQA-style question-answer pairs)
data = [
    "Question 1: What is the capital of France? Answer: Paris.",
    "Question 2: Who wrote Hamlet? Answer: William Shakespeare.",
    "Question 3: What is the boiling point of water? Answer: 100 degrees Celsius."
]

# Initialize the memory barrier with the dataset
memory_barrier = MemoryBarrier(data)

# Simulating conversation switching and memory retrieval
memory_barrier.open_conversation("conversation_1")
memory_barrier.retrieve_memory()

memory_barrier.open_conversation("conversation_2")
memory_barrier.retrieve_memory()

memory_barrier.open_conversation("conversation_3")
memory_barrier.retrieve_memory()

# Attempting to retrieve memory without switching to a conversation
memory_barrier.current_conversation = None  # Resetting the current conversation
memory_barrier.retrieve_memory()
