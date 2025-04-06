class MemoryStore:
    def __init__(self):
        self.memory_store = {}

    def store_memory(self, key, value):
        """Store memory using a hashable key and its associated value."""
        self.memory_store[tuple(key)] = value

    def retrieve_memory(self, key):
        """Retrieve memory based on the closest matching key."""
        closest_key = min(self.memory_store.keys(), key=lambda stored_key: sum((sk - k)**2 for sk, k in zip(stored_key, key))**0.5)
        return self.memory_store[closest_key]

    def display_memory(self):
        """Display all stored memory in a formatted way."""
        print("Retrieved Memory:")
        for key, value in self.memory_store.items():
            print(f"Key: {key}, Value: {value}")

# Instantiate the MemoryStore class
control_memory = MemoryStore()

# Store some example memories
control_memory.store_memory([0.5, -0.3], [0.7, 0.2])
control_memory.store_memory([0.6, -0.2], [0.9, -0.1])

# Retrieve a memory based on a query
retrieved_memory = control_memory.retrieve_memory([0.6, -0.25])

# Display the retrieved memory
print("Retrieved Memory:", retrieved_memory)

# Display all memory contents for comparison
control_memory.display_memory()
