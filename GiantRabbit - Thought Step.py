# Chain of Thought (CoT) Layer with Multi-Step Reasoning
class SparseCoTLayer:
    def __init__(self, d_model, num_steps, sparsity_factor):
        self.d_model = d_model
        self.num_steps = num_steps
        self.sparsity_factor = sparsity_factor
        self.weights = [[random.uniform(-0.1, 0.1) for _ in range(d_model)] for _ in range(d_model)]

    def apply_sparsity(self, data):
        sparse_data = []
        for value in data:
            if isinstance(value, list):
                sparse_row = [v if random.uniform(0, 1) < self.sparsity_factor else 0.0 for v in value]
                sparse_data.append(sparse_row)
            else:
                sparse_data.append(value if random.uniform(0, 1) < self.sparsity_factor else 0.0)
        return sparse_data

    def thought_step(self, current_state, input_data):
        next_state = []
        for i in range(len(current_state)):
            # Process if current state element is a list and there are corresponding weights
            if isinstance(current_state[i], list) and i < len(self.weights):
                combined_value = []
                for j in range(len(current_state[i])):
                    if j < len(self.weights[i]) and j < len(input_data):
                        sum_value = 0
                        # Ensure input_data[j] is iterable (like a list)
                        if isinstance(input_data[j], list) and j < len(current_state[i]):
                            for k in range(len(input_data[j])):
                                if k < len(self.weights[i]):
                                    # Only proceed if within bounds of both input_data and weights
                                    sum_value += input_data[j][k] * self.weights[i][j]
                        combined_value.append(current_state[i][j] + sum_value)
                    else:
                        combined_value.append(current_state[i][j])
                next_state.append(combined_value)

            # Process if current state element is an int/float (scalar value)
            elif isinstance(current_state[i], (int, float)):
                sum_value = sum(
                    input_data[j] * self.weights[i][j]
                    for j in range(min(len(input_data), len(self.weights[i])))
                    if isinstance(input_data[j], (int, float)) and j < len(self.weights[i])
                )
                next_state.append(current_state[i] + sum_value)

            # If it's neither a list nor a scalar, just append as is
            else:
                next_state.append(current_state[i])
                
        return next_state

    def call(self, input_sequence):
        current_state = input_sequence
        all_states = []
        for step in range(self.num_steps):
            sparse_state = self.apply_sparsity(current_state)
            next_state = self.thought_step(sparse_state, input_sequence)
            current_state = next_state
            all_states.append(current_state)
        return all_states

    def call(self, input_sequence):
       current_state = input_sequence
    all_states = []
    for step in range(self.num_steps):
           sparse_state = self.apply_sparsity(current_state)
           next_state = self.thought_step(sparse_state, input_sequence)
           current_state = next_state
        all_states.append(current_state)
       # Add this to print the final output
       print("Final Thought Process Output:", all_states)
       return all_states






