import random

# -------------------------------
# 1) USER INPUT (Query/Context)
# -------------------------------
def get_user_input():
    """
    Emulates receiving a user query or context as a string.
    """
    # For demonstration, we'll just hard-code a sample query.
    return "Compute the sum of the following numbers: 12 and 7."

# -------------------------------
# 2) TOKENIZATION & EMBEDDING
# -------------------------------
def simple_tokenizer(text):
    """
    Splits text into tokens (naive approach).
    """
    # A basic split on whitespace and punctuation
    # just for demonstration (no illusions).
    tokens = []
    current = ""
    for char in text:
        if char.isalnum():
            current += char
        else:
            if current:
                tokens.append(current.lower())
                current = ""
            if char.strip():
                # record punctuation as separate token
                tokens.append(char)
    if current:
        tokens.append(current.lower())
    return tokens

def positional_encoding(index):
    """
    Returns a basic numeric position value or small vector.
    We'll keep it scalar for demonstration.
    """
    # In real systems, you might have sine/cosine or a learned vector.
    # We'll return a simple integer for clarity.
    return index

def build_embedding_table():
    """
    Creates a trivial embedding lookup (word->random vector).
    In practice, you'd have a learned dictionary.
    """
    table = {}
    # We'll fill this on the fly in 'embed_tokens' if a new token is found.
    return table

def embed_tokens(tokens, embedding_table):
    """
    Returns a list of (token_embedding) for each token,
    combining a random vector with a simple positional encoding.
    """
    embedded_sequence = []
    for i, tok in enumerate(tokens):
        if tok not in embedding_table:
            # Create a random vector of dimension d=4 for demonstration
            # (no illusions of real dimension).
            vec = [random.uniform(-0.1, 0.1) for _ in range(4)]
            embedding_table[tok] = vec
        # Summation of embedding vector + position
        position_val = positional_encoding(i)
        embed_vec = embedding_table[tok]
        # We'll just store them as a tuple for demonstration
        combined = [val + position_val for val in embed_vec]
        embedded_sequence.append(combined)
    return embedded_sequence

# -------------------------------
# 3) MOE-ENABLED TRANSFORMER STACK
# -------------------------------
def router_and_experts(hidden_state):
    """
    Approximates the 'Router & Expert Blocks' stage.
    We pick top-K experts based on random scores,
    then produce partial outputs + gating logs.
    """
    # Suppose we have N=3 experts for demonstration
    # We'll generate random gating scores for each
    gating_scores = [random.random() for _ in range(3)]
    # We'll pick top-K=2 experts
    sorted_indices = sorted(range(3), key=lambda i: gating_scores[i], reverse=True)
    active_experts = sorted_indices[:2]

    # Combine outputs from the chosen experts.
    # We'll just do a naive sum of the hidden_state scaled by gating.
    partial_output = [0.0, 0.0, 0.0, 0.0]  # dimension 4 to match embedding
    for i in active_experts:
        factor = gating_scores[i]
        for d in range(len(partial_output)):
            partial_output[d] += factor * hidden_state[d]

    # Return gating logs & partial outputs
    return {
        "gating_scores": gating_scores,
        "partial_output": partial_output
    }

def feedforward_layers_enhanced_gating(token_embeddings):
    """
    Applies the router_and_experts to each embedded token in a loop,
    returning a list of dictionary outputs.
    """
    outputs = []
    for emb in token_embeddings:
        moe_result = router_and_experts(emb)
        outputs.append(moe_result)
    return outputs

def residual_norm_module(moe_outputs):
    """
    For demonstration, assume we do a minimal 'residual + scaling' for each output.
    """
    # We'll store updated outputs in a new list
    updated = []
    for out in moe_outputs:
        # out['partial_output'] is dimension=4
        res = []
        for val in out["partial_output"]:
            # a trivial 'resnorm' approach
            # e.g., y = val * 0.95 (just a placeholder)
            res.append(val * 0.95)
        new_out = {
            "gating_scores": out["gating_scores"],
            "partial_output": res
        }
        updated.append(new_out)
    return updated

def moe_transformer_stack(embedded_sequence):
    """
    MoE Transformer: Self-Attention -> Feed-Forward Layers -> Residual-Norm
    For simplicity, we skip explicit attention steps and show the MoE approach.
    """
    # We do a single pass for demonstration
    moe_stage_output = feedforward_layers_enhanced_gating(embedded_sequence)
    # Apply a trivial 'residual norm' stage
    final_outputs = residual_norm_module(moe_stage_output)
    return final_outputs

# -------------------------------
# 4) DYNAMIC CoT CONTROLLER
# -------------------------------
def compute_importance_score(gating_scores, partial_output):
    """
    Mimics I_t = gamma * sum_{active} alpha_t,e + (1-gamma)* attention_val
    We'll use partial_output magnitude as a stand-in for attention_val here.
    """
    gamma = 0.6
    sum_gating = sum(gating_scores)
    attention_val = sum(abs(x) for x in partial_output) / len(partial_output)
    importance = gamma * sum_gating + (1 - gamma) * attention_val
    return importance

def pruning_and_summarization(importance, partial_output, threshold):
    """
    If importance < threshold, remove or compress.
    We'll just 'remove' by returning None,
    or keep it if importance >= threshold.
    """
    if importance < threshold:
        return None
    else:
        # 'Compress' is trivially the same partial_output for demonstration
        return partial_output

def dynamic_cot_controller(moe_outputs):
    """
    Loops over each MoE output, computing importance and applying partial pruning.
    We'll store any kept tokens in a progressive buffer.
    """
    progressive_buffer = []
    partial_reward = 0.1  # placeholder
    threshold_base = 0.5
    updated_tokens = []

    for out in moe_outputs:
        gating_scores = out["gating_scores"]
        partial_out = out["partial_output"]

        # Step 1: compute importance
        importance_val = compute_importance_score(gating_scores, partial_out)

        # Step 2: partial reward modifies threshold
        # e.g., threshold = threshold_base + alpha * reward
        alpha = 0.2
        threshold = threshold_base + alpha * partial_reward

        # Step 3: prune or compress
        result = pruning_and_summarization(importance_val, partial_out, threshold)
        if result is not None:
            updated_tokens.append(result)
            progressive_buffer.append({
                "importance": importance_val,
                "payload": result
            })

        # optionally update partial reward if something was kept
        # for demonstration, let's do partial_reward += 0.05
        if result is not None:
            partial_reward += 0.05

    return {
        "progressive_buffer": progressive_buffer,
        "updated_tokens": updated_tokens
    }

# -------------------------------
# MAIN PIPELINE (User Input -> Dynamic CoT)
# -------------------------------
def main():
    # 1) USER INPUT
    user_text = get_user_input()

    # 2) TOKENIZATION & EMBEDDING
    embedding_table = build_embedding_table()
    tokens = simple_tokenizer(user_text)
    embedded_seq = embed_tokens(tokens, embedding_table)

    # 3) MOE-ENABLED TRANSFORMER STACK
    moe_stack_out = moe_transformer_stack(embedded_seq)

    # 4) DYNAMIC CoT CONTROLLER
    cot_result = dynamic_cot_controller(moe_stack_out)

    # Print out final results for demonstration
    print("Tokens:", tokens)
    print("Embedded Seq (first 2 vectors only for brevity):", embedded_seq[:2])
    print("MoE Stack Output (2 examples):", moe_stack_out[:2])
    print("CoT Controller Progressive Buffer:", cot_result["progressive_buffer"])

if __name__ == "__main__":
    main()
