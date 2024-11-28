import random

class SparseEmbeddingLayer:
    def __init__(self, vocab_size, embed_dim, sparsity_factor, token_importance):
        # Initialize vocabulary size, embedding dimension, and sparsity factor
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.sparsity_factor = sparsity_factor
        self.embedding = [[random.uniform(-1, 1) for _ in range(embed_dim)] for _ in range(vocab_size)]
        self.token_importance = token_importance  # Dictionary to hold importance of each token

    def call(self, input_tokens):
        embedded_tokens = [self.embedding[token] for token in input_tokens]
        
        # Apply dynamic sparsity mask to the embeddings
        sparse_embeddings = []
        for i, row in enumerate(embedded_tokens):
            sparse_row = []
            for j, value in enumerate(row):
                # Adjust sparsity factor based on token importance
                importance_factor = self.token_importance.get(input_tokens[i], 1.0)
                adjusted_sparsity_factor = min(1.0, self.sparsity_factor * importance_factor)
                
                # Use adjusted sparsity factor to determine whether to mask
                sparse_row.append(value if random.uniform(0, 1) < adjusted_sparsity_factor else 0.0)
            sparse_embeddings.append(sparse_row)
        
        return sparse_embeddings

# Example usage with dynamic masking
def run_sparse_embedding():
    vocab_size = 1000
    embed_dim = 64
    sparsity_factor = 0.5
    token_importance = {  # Example token importance values
        0: 1.2,
        5: 0.8,
        7: 1.5,
    }
    input_tokens = [random.randint(0, vocab_size - 1) for _ in range(10)]
    
    # Instantiate the sparse embedding layer with dynamic masking strategy
    sparse_embedding_layer = SparseEmbeddingLayer(vocab_size, embed_dim, sparsity_factor, token_importance)
    output = sparse_embedding_layer.call(input_tokens)
    print("Revised Sparse Embedding Output with Dynamic Masking:\n", output)

# Run the function
run_sparse_embedding()
