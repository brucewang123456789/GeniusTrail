#!/usr/bin/env python3

import random

# --------------------------------------------------
#        1) USER INPUT (Query/Context)
# --------------------------------------------------
def get_user_input():
    """
    Emulates receiving a user query or context as a string.
    """
    # Hard-coded sample query for demonstration:
    return "Compute the sum of the following numbers: 12 and 7."

# --------------------------------------------------
#        2) TOKENIZATION & EMBEDDING
# --------------------------------------------------
def simple_tokenizer(text):
    """
    Splits text into tokens (naive approach).
    """
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
                tokens.append(char)
    if current:
        tokens.append(current.lower())
    return tokens

def positional_encoding(index):
    """
    Returns a basic numeric position value or small vector.
    We'll keep it scalar for demonstration.
    """
    return index

def build_embedding_table():
    """
    Creates a trivial embedding lookup (word->random vector).
    In practice, you'd have a learned dictionary.
    """
    return {}

def embed_tokens(tokens, embedding_table):
    """
    Returns a list of (token_embedding) for each token,
    combining a random vector with a simple positional encoding.
    """
    embedded_sequence = []
    for i, tok in enumerate(tokens):
        if tok not in embedding_table:
            vec = [random.uniform(-0.1, 0.1) for _ in range(4)]
            embedding_table[tok] = vec
        pos_val = positional_encoding(i)
        embed_vec = embedding_table[tok]
        combined = [val + pos_val for val in embed_vec]
        embedded_sequence.append(combined)
    return embedded_sequence

# --------------------------------------------------
#        3) MOE-ENABLED TRANSFORMER STACK
# --------------------------------------------------
def router_and_experts(hidden_state):
    """
    Approximates 'Router & Expert Blocks': pick top-K=2 experts
    based on random gating scores, then produce partial outputs.
    """
    gating_scores = [random.random() for _ in range(3)]
    sorted_indices = sorted(range(3), key=lambda i: gating_scores[i], reverse=True)
    active_experts = sorted_indices[:2]

    partial_output = [0.0, 0.0, 0.0, 0.0]
    for i in active_experts:
        factor = gating_scores[i]
        for d in range(len(partial_output)):
            partial_output[d] += factor * hidden_state[d]

    return {
        "gating_scores": gating_scores,
        "partial_output": partial_output
    }

def feedforward_layers_enhanced_gating(token_embeddings):
    """
    Applies router_and_experts to each token, returning gating logs + partial outputs.
    """
    outputs = []
    for emb in token_embeddings:
        moe_result = router_and_experts(emb)
        outputs.append(moe_result)
    return outputs

def residual_norm_module(moe_outputs):
    """
    Minimal residual + scaling for demonstration.
    """
    updated = []
    for out in moe_outputs:
        partial = out["partial_output"]
        res = [val * 0.95 for val in partial]
        updated.append({
            "gating_scores": out["gating_scores"],
            "partial_output": res
        })
    return updated

def moe_transformer_stack(embedded_sequence):
    """
    MoE Transformer Stack: For simplicity, skip explicit attention
    and just do feed-forward + residual norm.
    """
    moe_out = feedforward_layers_enhanced_gating(embedded_sequence)
    final_out = residual_norm_module(moe_out)
    return final_out

# --------------------------------------------------
#        4) DYNAMIC CoT CONTROLLER
# --------------------------------------------------
def compute_importance_score(gating_scores, partial_output):
    """
    I_t = gamma * sum_gating + (1-gamma)* attention_val
    Using partial_output magnitude as a stand-in for attention.
    """
    gamma = 0.6
    sum_gating = sum(gating_scores)
    attention_val = sum(abs(x) for x in partial_output) / len(partial_output)
    importance = gamma * sum_gating + (1 - gamma) * attention_val
    return importance

def pruning_and_summarization(importance, partial_output, threshold):
    """
    Prune if importance < threshold; else keep (or 'compress').
    """
    if importance < threshold:
        return None
    else:
        # For demonstration, we keep partial_output unchanged
        return partial_output

def reasoning_discriminator(query_text):
    """
    Determines whether reasoning is required.
    Uses:
    - `P_fact`: Probability that input is a factual/common-sense statement.
    - `C_comp`: Computational complexity score.
    """
    P_fact = random.uniform(0, 1)  # Placeholder for a real classification model
    C_comp = random.randint(1, 5)  # Placeholder for a complexity estimation function

    if P_fact >= 0.85 and C_comp <= 3:
        return "N"  # No reasoning required, directly output answer
    return "Y"  # Requires CoT reasoning

def dynamic_cot_controller(moe_outputs):
    """
    Iterates over each MoE output, computing an importance value
    and applying partial pruning. Stores the result in progressive_buffer.
    """
    progressive_buffer = []
    partial_reward = 0.1
    threshold_base = 0.5
    updated_tokens = []

    for out in moe_outputs:
        gating_scores = out["gating_scores"]
        partial_out = out["partial_output"]

        importance_val = compute_importance_score(gating_scores, partial_out)

        alpha = 0.2
        threshold = threshold_base + alpha * partial_reward

        result = pruning_and_summarization(importance_val, partial_out, threshold)
        if result is not None:
            updated_tokens.append(result)
            progressive_buffer.append({
                "importance": importance_val,
                "payload": result
            })

        # Increase partial_reward if we kept something
        if result is not None:
            partial_reward += 0.05

    return {
        "progressive_buffer": progressive_buffer,
        "updated_tokens": updated_tokens,
        # We might store partial_reward here for use in the next stage
        "partial_reward": partial_reward
    }

# --------------------------------------------------
#        5) AUTO-REGRESSIVE DECODING (Iterative Blocks)
# --------------------------------------------------
def token_selection(buffered_coT, decode_params):
    """
    Chooses which tokens/blocks to decode next.
    Takes the last block from progressive_buffer for demonstration.
    """
    if not buffered_coT:
        return []
    return buffered_coT[-1]["payload"]

def iterative_generation(selected_block, decode_state):
    """
    Expands tokens in small increments. We add random offsets.
    """
    new_tokens = []
    for val in selected_block:
        offset = random.uniform(-0.05, 0.05)
        new_tokens.append(val + offset)
    return new_tokens

def adjusting_pruning_blocks(pruned_block, expanded_block, partial_reward):
    """
    Merge or discard blocks based on partial_reward threshold logic.
    """
    threshold = 0.15
    if partial_reward < threshold:
        return pruned_block
    else:
        return pruned_block + expanded_block

def auto_regressive_decoding(cot_controller_output, decode_params):
    """
    Main function for the Auto-Reg Decoding with small iterative blocks.
    Expects:
      - progressive_buffer
      - partial_reward
    """
    partial_reward = cot_controller_output.get("partial_reward", 0.05)
    progressive_buffer = cot_controller_output.get("progressive_buffer", [])
    decode_state = {}

    # 1) Token Selection
    selected_block = token_selection(progressive_buffer, decode_params)

    # 2) We define a trivial pruned_block (just scaled)
    pruned_block = [x * 0.9 for x in selected_block]

    # 3) Iterative Generation
    expanded_block = iterative_generation(selected_block, decode_state)

    # 4) Adjusting / Pruning (RL)
    final_blocks = adjusting_pruning_blocks(pruned_block, expanded_block, partial_reward)

    # Build an 'intermediate CoT'
    intermediate_cot = {
        "coT_intermediate": final_blocks,
        "partial_reward": partial_reward + 0.02  # nudge reward up
    }
    return intermediate_cot

# --------------------------------------------------
#        6) HIERARCHICAL CoT ASSEMBLY
# --------------------------------------------------
def macro_summaries(coT_data):
    """
    Splits top-level steps from coT_data. We'll just pick every other element.
    """
    return coT_data[::2]

def micro_details(coT_data):
    """
    Takes the 'in-between' elements as micro-level detail.
    """
    return coT_data[1::2]

def reward_context_map(macro, micro, partial_reward):
    """
    Associates reward signals with each macro/micro step.
    """
    r_map = {}
    idx = 0
    for val in macro + micro:
        r_map[idx] = partial_reward
        idx += 1
    return r_map

def reward_aligned_refinement(macro, micro, r_map):
    """
    Possibly reorder or highlight tokens based on reward.
    We'll just compress any token whose reward is below a threshold.
    """
    final_rep = []
    for i, val in enumerate(macro + micro):
        if r_map[i] > 0.1:
            final_rep.append(val)
        else:
            final_rep.append(val * 0.8)
    return final_rep

def hierarchical_cot_assembly(intermediate_cot):
    """
    Gathers the partial CoT, splits into macro+micro, applies reward-based refinement.
    """
    partial_reward = intermediate_cot["partial_reward"]
    coT_seq = intermediate_cot["coT_intermediate"]

    macro = macro_summaries(coT_seq)
    micro = micro_details(coT_seq)
    r_map = reward_context_map(macro, micro, partial_reward)

    final_structure = reward_aligned_refinement(macro, micro, r_map)

    return {
        "final_structure": final_structure,
        "partial_reward": partial_reward
    }

# --------------------------------------------------
#        7) FINAL OUTPUT GENERATION + RL LOOP
# --------------------------------------------------
def produce_final_answer(refined_structure):
    """
    Converts final chain-of-thought into a user-facing answer.
    For demonstration, we just join them as a string.
    """
    return " ".join(str(round(x, 3)) for x in refined_structure)

def integrated_rl_reward_loop(final_answer, partial_reward):
    """
    If continuing RL training, compute or simulate an episodic reward
    based on final answer and feed back to dynamic CoT gating logic.
    """
    max_token = 0.0
    tokens = final_answer.split()
    for t in tokens:
        # check if numeric
        if t.replace('.', '', 1).isdigit():
            val = float(t)
            if val > max_token:
                max_token = val

    episode_reward = partial_reward
    if max_token > 0.9:
        episode_reward += 0.1

    dynamic_cot_update = {
        "updated_reward": episode_reward
    }
    return dynamic_cot_update

def final_output_generation(hier_cot_result, enable_loop=False):
    """
    Produces final answer, optionally loops RL reward back to Dynamic CoT Controller.
    """
    partial_reward = hier_cot_result["partial_reward"]
    refined_structure = hier_cot_result["final_structure"]

    # 1) Final Answer
    answer = produce_final_answer(refined_structure)
    print("**Final Answer**:", answer)

    # 2) Integrated RL Reward (Optional)
    if enable_loop:
        feedback = integrated_rl_reward_loop(answer, partial_reward)
        return {
            "final_answer": answer,
            "rl_feedback": feedback
        }
    else:
        return {
            "final_answer": answer,
            "rl_feedback": None
        }

def dynamic_cot_update_controller(cot_controller_state, rl_feedback):
    """
    Takes gating/pruning thresholds in cot_controller_state
    and updates them with RL feedback for next run.
    """
    if not rl_feedback:
        return cot_controller_state
    if "partial_reward" in cot_controller_state:
        cot_controller_state["partial_reward"] += rl_feedback["updated_reward"]
    return cot_controller_state

# --------------------------------------------------
#        MAIN PIPELINE (From Start to End)
# --------------------------------------------------
def main():
    """
    Full pipeline from user query all the way to final output,
    with an optional RL loop back to the dynamic CoT controller.
    """

    # --- MODULE 1: USER INPUT ---
    user_text = get_user_input()

    # --- MODULE 2: TOKENIZATION & EMBEDDING ---
    embedding_table = build_embedding_table()
    tokens = simple_tokenizer(user_text)
    embedded_seq = embed_tokens(tokens, embedding_table)

    # --- MODULE 3: MoE-ENABLED TRANSFORMER STACK ---
    moe_stack_out = moe_transformer_stack(embedded_seq)

    # --- MODULE 4: DYNAMIC CoT CONTROLLER ---
    cot_result = dynamic_cot_controller(moe_stack_out)
    # We'll print partial results for demonstration
    print("Tokens:", tokens)
    print("Embedded Seq (2 examples):", embedded_seq[:2])
    print("MoE Stack Output (2 examples):", moe_stack_out[:2])
    print("CoT Controller Buffer:", cot_result["progressive_buffer"])

    # --- MODULE 5: AUTO-REGRESSIVE DECODING ---
    decode_params = {"some_decode_param": "placeholder"}
    intermediate_cot = auto_regressive_decoding(cot_result, decode_params)

    # --- MODULE 6: HIERARCHICAL CoT ASSEMBLY ---
    hier_cot_result = hierarchical_cot_assembly(intermediate_cot)

    # --- MODULE 7: FINAL OUTPUT GENERATION ---
    final_result = final_output_generation(hier_cot_result, enable_loop=True)

    # If RL feedback is present, update dynamic CoT gating
    if final_result["rl_feedback"]:
        updated_cot = dynamic_cot_update_controller(cot_result, final_result["rl_feedback"])
        print("Updated Dynamic CoT State:", updated_cot)

if __name__ == "__main__":
    main()
