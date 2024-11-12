import random
import math

# Define the Sparse Encoding Layer
class SparseEncodingLayer:
    def __init__(self, vocab_size, embed_dim, seq_length, sparsity_factor):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.seq_length = seq_length
        self.sparsity_factor = sparsity_factor

        # Creating embedding matrix randomly
        random.seed(42)
        self.embedding = [[random.random() for _ in range(embed_dim)] for _ in range(vocab_size)]
        self.positional_encoding = self._get_positional_encoding(seq_length, embed_dim)

    def _get_positional_encoding(self, seq_length, embed_dim):
        # Create positional encoding for the input
        pos_encoding = []
        for pos in range(seq_length):
            pos_vector = []
            for i in range(embed_dim):
                if i % 2 == 0:
                    pos_vector.append(math.sin(pos / (10000 ** (i / embed_dim))))
                else:
                    pos_vector.append(math.cos(pos / (10000 ** (i / embed_dim))))
            pos_encoding.append(pos_vector)
        return pos_encoding

    def call(self, input_tokens):
        # Step 1: Embedding Lookup
        embedded_tokens = []
        for token_sequence in input_tokens:
            embedded_sequence = [self.embedding[token] for token in token_sequence]
            embedded_tokens.append(embedded_sequence)

        # Step 2: Apply Sparsity Mask to Embeddings
        sparse_embeddings = []
        for embedded_sequence in embedded_tokens:
            sparsity_mask = [
                [1 if random.uniform(0, 1) < self.sparsity_factor else 0 for _ in range(self.embed_dim)]
                for _ in range(self.seq_length)
            ]
            sparse_sequence = [
                [embedded_vector[i] * sparsity_mask[j][i] for i in range(self.embed_dim)]
                for j, embedded_vector in enumerate(embedded_sequence)
            ]
            sparse_embeddings.append(sparse_sequence)

        # Step 3: Apply Sparse Positional Encoding
        sparse_pos_encoded_output = []
        for i, sparse_sequence in enumerate(sparse_embeddings):
            pos_encoded_sequence = [
                [sparse_sequence[j][k] + self.positional_encoding[j][k] * sparsity_mask[j][k]
                 for k in range(self.embed_dim)]
                for j in range(self.seq_length)
            ]
            sparse_pos_encoded_output.append(pos_encoded_sequence)

        return sparse_pos_encoded_output

# Example usage
def run_sparse_encoding_layer():
    vocab_size = 1000  # Vocabulary size
    embed_dim = 64  # Embedding dimension
    seq_length = 10  # Sequence length
    sparsity_factor = 0.5  # Sparsity factor (50% of dimensions are active)

    batch_size = 2
    input_tokens = [[random.randint(0, vocab_size - 1) for _ in range(seq_length)] for _ in range(batch_size)]

    # Create and run the Sparse Encoding Layer
    sparse_encoding_layer = SparseEncodingLayer(vocab_size, embed_dim, seq_length, sparsity_factor)
    output = sparse_encoding_layer.call(input_tokens)

    print("Sparse Encoding Layer Output:")
    for batch in output:
        for sequence in batch:
            print(sequence)

run_sparse_encoding_layer()
