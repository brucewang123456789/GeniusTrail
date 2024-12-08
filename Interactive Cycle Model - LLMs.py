import math

class Tokenizer:
    def __init__(self):
        self.vocab = {"hello": 0, "how": 1, "can": 2, "i": 3, "assist": 4, "you": 5, "?": 6}

    def tokenize(self, input_text):
        try:
            tokens = [self.vocab.get(word, len(self.vocab)) for word in input_text.lower().split()]
            print(f"Tokenized input: {tokens}")
            return tokens
        except Exception as e:
            raise RuntimeError(f"Error in tokenization: {str(e)}")

class Embedding:
    def __init__(self, vocab_size, embedding_dim):
        self.embedding_matrix = [[(i + j) / embedding_dim for j in range(embedding_dim)] for i in range(vocab_size)]

    def embed(self, tokens):
        try:
            embedded_tokens = [self.embedding_matrix[token] for token in tokens]
            print(f"Embedded tokens: {embedded_tokens}")
            return embedded_tokens
        except Exception as e:
            raise RuntimeError(f"Error in embedding: {str(e)}")

class PositionalEncoding:
    def __init__(self, max_len, embedding_dim):
        self.positional_encoding = [
            [
                math.sin(pos / (10000 ** (2 * (i // 2) / embedding_dim))) if i % 2 == 0 else math.cos(pos / (10000 ** (2 * (i // 2) / embedding_dim)))
                for i in range(embedding_dim)
            ]
            for pos in range(max_len)
        ]

    def add_positional_encoding(self, embeddings):
        try:
            encoded = [
                [emb + pos for emb, pos in zip(embedding, self.positional_encoding[idx])]
                for idx, embedding in enumerate(embeddings)
            ]
            print(f"Positional Encoded tokens: {encoded}")
            return encoded
        except Exception as e:
            raise RuntimeError(f"Error in positional encoding: {str(e)}")

class ForwardPass:
    def __init__(self, embedding_dim):
        self.attention_weights = [[0.1 * (i + j) for j in range(embedding_dim)] for i in range(embedding_dim)]

    def process(self, encoded_embeddings):
        try:
            forward_output = [
                [sum(embedding[idx] * weight for idx, weight in enumerate(self.attention_weights[i])) for i in range(len(embedding))]
                for embedding in encoded_embeddings
            ]
            print(f"Forward Pass output: {forward_output}")
            return forward_output
        except Exception as e:
            raise RuntimeError(f"Error in forward pass: {str(e)}")

class Decoder:
    def __init__(self, vocab_size, embedding_dim):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.weights = [[(i + j) / vocab_size for j in range(vocab_size)] for i in range(embedding_dim)]

    def decode(self, embeddings):
        try:
            logits = [
                [sum(emb * weight for emb, weight in zip(embedding, self.weights[idx % self.vocab_size])) for idx in range(self.vocab_size)]
                for embedding in embeddings
            ]
            print(f"Logits: {logits}")
            probabilities = [
                [math.exp(logit) / sum(math.exp(log) for log in logits[idx]) for logit in logits[idx]]
                for idx in range(len(logits))
            ]
            print(f"Probabilities: {probabilities}")
            return probabilities
        except Exception as e:
            raise RuntimeError(f"Error in decoding: {str(e)}")

class LargeLanguageModel:
    def __init__(self):
        self.tokenizer = Tokenizer()
        self.embedding = Embedding(vocab_size=10, embedding_dim=8)
        self.positional_encoding = PositionalEncoding(max_len=50, embedding_dim=8)
        self.forward_pass = ForwardPass(embedding_dim=8)
        self.decoder = Decoder(vocab_size=10, embedding_dim=8)

    def generate_response(self, input_text):
        try:
            tokens = self.tokenizer.tokenize(input_text)
            embeddings = self.embedding.embed(tokens)
            encoded = self.positional_encoding.add_positional_encoding(embeddings)
            forward_output = self.forward_pass.process(encoded)
            probabilities = self.decoder.decode(forward_output)
            response = "Response generated based on token probabilities"
            return response
        except Exception as e:
            raise RuntimeError(f"Error in LLM processing: {str(e)}")

class LLMsPipeline:
    def __init__(self):
        self.llm = LargeLanguageModel()

    def run_llm_pipeline(self, input_text):
        try:
            print("Running LLM pipeline...")
            response = self.llm.generate_response(input_text)
            print("LLM pipeline completed.")
            return response
        except Exception as e:
            raise RuntimeError(f"Error in LLM pipeline: {str(e)}")

# Example Usage
if __name__ == "__main__":
    llm_pipeline = LLMsPipeline()

    try:
        input_text = "Hello, how can I assist you?"  # Example input text from ASR pipeline
        result = llm_pipeline.run_llm_pipeline(input_text)
        print(f"Final LLM Output: {result}")
    except Exception as e:
        print(f"Pipeline error: {e}")
