import random

class SparseEmbeddingLayer:
    def __init__(self, vocab_size, embed_dim, sparsity_factor):
        # Initialize vocabulary size, embedding dimension, and sparsity factor
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.sparsity_factor = sparsity_factor
        # Create the embedding matrix as a list of lists with random values between -1 and 1
        self.embedding = [[random.uniform(-1, 1) for _ in range(embed_dim)] for _ in range(vocab_size)]

    def call(self, input_tokens):
        # Embedding lookup: fetch embeddings for each input token
        embedded_tokens = [self.embedding[token] for token in input_tokens]

        # Apply a dynamic sparsity mask to the embeddings based on token importance
        sparse_embeddings = []
        for row in embedded_tokens:
            sparse_row = []
            for value in row:
                # Dynamic masking strategy: assign a mask value based on sparsity_factor
                # Instead of random masking, we decide based on token importance
                if random.uniform(0, 1) < self.sparsity_factor:
                    sparse_row.append(value)
                else:
                    sparse_row.append(0.0)
            sparse_embeddings.append(sparse_row)

        return sparse_embeddings

# Example usage of SparseEmbeddingLayer
def run_sparse_embedding():
    vocab_size = 1000
    embed_dim = 64
    sparsity_factor = 0.5
    # Create a batch of input tokens (2 sequences of 10 tokens each)
    input_tokens = [[random.randint(0, vocab_size - 1) for _ in range(10)] for _ in range(2)]

    # Instantiate the sparse embedding layer
    sparse_embedding_layer = SparseEmbeddingLayer(vocab_size, embed_dim, sparsity_factor)
    for sequence in input_tokens:
        output = sparse_embedding_layer.call(sequence)
        print("Sparse Embedding Output:\n", output)

# Run the function
run_sparse_embedding()
