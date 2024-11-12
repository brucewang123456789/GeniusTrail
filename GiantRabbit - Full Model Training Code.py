import random

# Sparsemax Activation Layer
class Sparsemax:
    def call(self, logits):
        logits_sorted = sorted(logits, reverse=True)
        cumsum_logits = []
        current_sum = 0
        for logit in logits_sorted:
            current_sum += logit
            cumsum_logits.append(current_sum)

        k_array = list(range(1, len(logits) + 1))
        threshold = [(cumsum_logits[i] - 1) / k_array[i] for i in range(len(logits))]
        valid_entries = [i for i, l in enumerate(logits_sorted) if l > threshold[i]]
        k = len(valid_entries)
        tau = (cumsum_logits[k - 1] - 1) / k
        sparsemax_output = [max(logit - tau, 0) for logit in logits]

        return sparsemax_output


# Sparse Embedding Layer with Masking
class SparseEmbeddingLayer:
    def __init__(self, vocab_size, embed_dim, sparsity_factor):
        self.embedding = [[random.uniform(-1, 1) for _ in range(embed_dim)] for _ in range(vocab_size)]
        self.sparsity_factor = sparsity_factor

    def call(self, input_tokens):
        embedded_tokens = [self.embedding[token] for token in input_tokens]
        sparse_embeddings = []
        for row in embedded_tokens:
            sparse_row = [value if random.uniform(0, 1) < self.sparsity_factor else 0.0 for value in row]
            sparse_embeddings.append(sparse_row)
        return sparse_embeddings


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
            if isinstance(current_state[i], list) and i < len(self.weights):
                combined_value = []
                for j in range(len(current_state[i])):
                    if j < len(self.weights[i]):
                        sum_value = 0
                        for k in range(min(len(input_data[j]), len(self.weights[i]))):
                            if isinstance(input_data[j], list) and k < len(input_data[j]):
                                sum_value += input_data[j][k] * self.weights[i][j]
                        combined_value.append(current_state[i][j] + sum_value)
                    else:
                        combined_value.append(current_state[i][j])
                next_state.append(combined_value)
            elif isinstance(current_state[i], (int, float)):
                sum_value = sum(
                    input_data[j] * self.weights[i][j]
                    for j in range(min(len(input_data), len(self.weights[i])))
                    if isinstance(input_data[j], (int, float))
                )
                next_state.append(current_state[i] + sum_value)
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


# Full Model: GiantRabbitModel with all layers
class GiantRabbitModel:
    def __init__(self, vocab_size, embed_dim, d_model, num_heads, sparsity_factor, seq_length):
        self.sparse_embedding = SparseEmbeddingLayer(vocab_size, embed_dim, sparsity_factor)
        self.sparse_cot = SparseCoTLayer(d_model, 5, sparsity_factor)

    def call(self, input_tokens):
        sparse_embeddings = self.sparse_embedding.call(input_tokens)
        cot_output = self.sparse_cot.call(sparse_embeddings)
        return cot_output


# Training Code for GiantRabbitModel
def train_giant_rabbit_model():
    vocab_size = 1000
    embed_dim = 64
    d_model = 64
    num_heads = 8
    sparsity_factor = 0.5
    seq_length = 10
    epochs = 10

    giant_rabbit_model = GiantRabbitModel(vocab_size, embed_dim, d_model, num_heads, sparsity_factor, seq_length)
    for epoch in range(epochs):
        input_tokens = [random.randint(0, vocab_size - 1) for _ in range(seq_length)]
        target_data = [[random.uniform(-1, 1) for _ in range(d_model)] for _ in range(seq_length)]
        predictions = giant_rabbit_model.call(input_tokens)
        
        # Ensure we are not exceeding bounds during loss calculation
        min_length = min(len(predictions[-1]), len(target_data))
        loss = sum(
            (predictions[-1][i][j] - target_data[i][j]) ** 2
            for i in range(min_length)
            for j in range(min(len(predictions[-1][i]), len(target_data[i])))
        ) / (seq_length * d_model)
        
        print(f"Epoch {epoch + 1}, Loss: {loss:.4f}")


# Run training
train_giant_rabbit_model()
