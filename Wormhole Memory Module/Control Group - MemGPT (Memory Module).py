class MemGPTMemoryModule:
    def __init__(self):
        self.memory = {}

    def store_memory(self, key, value):
        self.memory[key] = value

    def retrieve_memory(self, query):
        # Simplistic retrieval logic
        results = []
        for key, value in self.memory.items():
            if query in str(value):
                results.append(value)
        return results

    def clear_memory(self):
        self.memory.clear()

    def display_memory(self):
        # Adjusted output format
        output = {"Retrieved Memory": {}, "Memory After Clearing": {}}
        output["Retrieved Memory"] = {key: value for key, value in self.memory.items()}
        return output


# Create an instance of MemGPTMemoryModule
control_group = MemGPTMemoryModule()

# Store memory
control_group.store_memory("Key 1", "The cat is on the mat.")
control_group.store_memory("Key 2", "The dog chased the cat.")

# Retrieve memory
query = "cat"
retrieved = control_group.retrieve_memory(query)

# Output adjusted for consistency
print("Query:", query)
print("Results:", retrieved)
print("Memory After Clearing:", control_group.display_memory())

# Clear memory
control_group.clear_memory()
print("Memory after clearing:", control_group.memory)
