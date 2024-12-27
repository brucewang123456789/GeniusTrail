import random

class MemoryAugmentedReasoning:
    def __init__(self, stm_capacity, ltm_capacity):
        """Initialize Short-Term Memory (STM) and Long-Term Memory (LTM)."""
        self.short_term_memory = []  # Stores recent scenarios
        self.long_term_memory = []  # Stores historical scenarios
        self.stm_capacity = stm_capacity  # Maximum size of STM
        self.ltm_capacity = ltm_capacity  # Maximum size of LTM
    
    def update_stm(self, scenario):
        """Update short-term memory with a new scenario."""
        if len(self.short_term_memory) >= self.stm_capacity:
            self.short_term_memory.pop(0)  # Remove oldest entry if STM is full
        self.short_term_memory.append(scenario)
        print(f"STM Updated: {self.short_term_memory}")

    def update_ltm(self, scenario):
        """Add a new scenario to long-term memory."""
        if len(self.long_term_memory) >= self.ltm_capacity:
            self.long_term_memory.pop(0)  # Remove oldest entry if LTM is full
        self.long_term_memory.append(scenario)
        print(f"LTM Updated: {self.long_term_memory}")

    def retrieve_ltm(self, query):
        """Retrieve relevant scenarios from long-term memory."""
        relevance_scores = [self.similarity(query, memory) for memory in self.long_term_memory]
        if relevance_scores:
            # Retrieve the most relevant memory entry
            max_index = relevance_scores.index(max(relevance_scores))
            print(f"Relevant LTM Retrieved: {self.long_term_memory[max_index]}")
            return self.long_term_memory[max_index]
        print("No Relevant LTM Found")
        return None

    def sparse_attention(self, query):
        """Apply sparse attention to prioritize memory entries."""
        stm_scores = [self.similarity(query, memory) for memory in self.short_term_memory]
        ltm_scores = [self.similarity(query, memory) for memory in self.long_term_memory]

        # Combine STM and LTM scores
        all_scores = stm_scores + ltm_scores
        all_memories = self.short_term_memory + self.long_term_memory

        if all_scores:
            # Select top-k entries based on sparse attention
            k = min(3, len(all_scores))  # Top-3 selection for simplicity
            top_indices = sorted(range(len(all_scores)), key=lambda i: all_scores[i], reverse=True)[:k]
            prioritized_memories = [all_memories[i] for i in top_indices]
            print(f"Prioritized Memories: {prioritized_memories}")
            return prioritized_memories
        print("No Prioritized Memories Found")
        return []

    def similarity(self, query, memory):
        """Calculate similarity between query and memory (randomized for demonstration)."""
        return random.random()  # Replace with an actual similarity metric in practice

    def process(self, scenario_output):
        """Execute the memory reasoning process with Scenario Output."""
        # Preprocess Scenario Output
        query = scenario_output.get("scenario", "Default Scenario")
        print(f"Processing Query: {query}")

        # Update STM with the query as the latest scenario
        self.update_stm(query)

        # Retrieve relevant memory from LTM
        relevant_ltm = self.retrieve_ltm(query)

        # Apply sparse attention across STM and LTM
        prioritized_memories = self.sparse_attention(query)

        # Generate refined output
        output = {
            "query": query,
            "relevant_ltm": relevant_ltm,
            "prioritized_memories": prioritized_memories
        }
        print("Final Output:", output)
        return output

# Test Case for Memory-Augmented Reasoning
if __name__ == "__main__":
    # Initialize the module with STM and LTM capacities
    mar_module = MemoryAugmentedReasoning(stm_capacity=5, ltm_capacity=10)

    # Populate LTM with historical data
    for i in range(7):
        scenario = f"Historical Scenario {i}"
        mar_module.update_ltm(scenario)

    # Simulate Scenario Output from Part 3
    scenario_output = {"scenario": "Example Scenario from Part 3"}

    # Process the Scenario Output
    result = mar_module.process(scenario_output)

    print("\nProcessed Output:")
    print("Query:", result["query"])
    print("Relevant LTM:", result["relevant_ltm"])
    print("Prioritized Memories:", result["prioritized_memories"])
