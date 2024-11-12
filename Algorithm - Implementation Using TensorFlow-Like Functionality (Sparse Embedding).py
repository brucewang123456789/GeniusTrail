import random

# Step 1: Define Inputs
# Embedding Matrix (E), Vocabulary Size = 6, Embedding Dimension = 4
V = 6  # Vocabulary size
D = 4  # Embedding dimension

# Generating the embedding matrix with random values
random.seed(42)
embedding_matrix = [[random.random() for _ in range(D)] for _ in range(V)]

# Input Token Indices (x), Batch Size = 1, Sequence Length = 5
batch_size = 1
sequence_length = 5
input_tokens = [[random.randint(0, V - 1) for _ in range(sequence_length)] for _ in range(batch_size)]

# Sparsity Mask (M), Sparsity Factor alpha = 0.5 (i.e., 50% of dimensions are active)
alpha = 0.5
active_dimensions = int(alpha * D)
mask = [0] * D
for i in range(active_dimensions):
    mask[i] = 1
random.shuffle(mask)  # Shuffle to randomize active dimensions

# Step 2: Embedding Lookup
# Embed(x) = E[x] -> This performs the embedding lookup
embedded_tokens = []
for token_sequence in input_tokens:
    embedded_sequence = []
    for token in token_sequence:
        embedded_sequence.append(embedding_matrix[token])
    embedded_tokens.append(embedded_sequence)

# Step 3: Apply Sparsity Mask to the Embedding
# SparseEmbed(x) = Embed(x) * M -> Applying mask to reduce dimensions
sparse_embeddings = []
for embedded_sequence in embedded_tokens:
    sparse_sequence = []
    for embedded_vector in embedded_sequence:
        sparse_vector = [embedded_vector[i] * mask[i] for i in range(D)]
        sparse_sequence.append(sparse_vector)
    sparse_embeddings.append(sparse_sequence)

# Step 4: Output
print("Embedding Matrix (E):")
for row in embedding_matrix:
    print(row)

print("\nInput Tokens (x):")
for row in input_tokens:
    print(row)

print("\nEmbedded Tokens (Embed(x)):")
for sequence in embedded_tokens:
    for vector in sequence:
        print(vector)

print("\nSparsity Mask (M):")
print(mask)

print("\nSparse Embeddings (SparseEmbed(x)):")
for sequence in sparse_embeddings:
    for vector in sequence:
        print(vector)
