#!/usr/bin/env python3

import random
import time

# ---------------------------------------------------------------------
# (1) USER INPUT (Query/Context)
# ---------------------------------------------------------------------
def get_user_input():
    """
    Accepts a user query or context string.
    In actual experiments, you can pass any text or math problem.
    """
    return "Compute the sum of 12 and 7."  # Example default

# ---------------------------------------------------------------------
# (2) TOKENIZATION & EMBEDDING
# ---------------------------------------------------------------------
def simple_tokenizer(text):
    """
    Naive tokenizer splitting on non-alphanumeric chars.
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
    Returns a basic scalar position encoding.
    """
    return index

def build_embedding_table():
    """
    Creates or loads a trivial embedding dictionary.
    """
    return {}

def embed_tokens(tokens, embedding_table):
    """
    Maps each token to a random 4D vector + positional encoding.
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

# ---------------------------------------------------------------------
# (3) MOE-ENABLED TRANSFORMER STACK
# ---------------------------------------------------------------------
def router_and_experts(hidden_state):
    """
    Approximates top-K gating in a multi-expert feed-forward scenario.
    """
    gating_scores = [random.random() for _ in range(3)]
    sorted_indices = sorted(range(3), key=lambda i: gating_scores[i], reverse=True)
    active_experts = sorted_indices[:2]

    partial_output = [0.0] * len(hidden_state)
    for i in active_experts:
        factor = gating_scores[i]
        for d in range(len(partial_output)):
            partial_output[d] += factor * hidden_state[d]

    return {
        "gating_scores": gating_scores,
        "partial_output": partial_output
    }

def feedforward_layers_enhanced_gating(token_embeddings):
    outputs = []
    for emb in token_embeddings:
        moe_result = router_and_experts(emb)
        outputs.append(moe_result)
    return outputs

def residual_norm_module(moe_outputs):
    """
    Minimal residual scaling approach for demonstration.
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
    The combined MoE stack: gating + a trivial 'residual norm'.
    """
    moe_out = feedforward_layers_enhanced_gating(embedded_sequence)
    final_out = residual_norm_module(moe_out)
    return final_out

# ---------------------------------------------------------------------
# (4) DYNAMIC CoT CONTROLLER
# ---------------------------------------------------------------------
def compute_importance_score(gating_scores, partial_output):
    """
    I_t = gamma * sum_gating + (1 - gamma)* attention_val
    """
    gamma = 0.6
    sum_gating = sum(gating_scores)
    attention_val = sum(abs(x) for x in partial_output) / len(partial_output)
    importance = gamma * sum_gating + (1 - gamma) * attention_val
    return importance

def pruning_and_summarization(importance, partial_output, threshold):
    """
    Prunes tokens if importance < threshold. Otherwise, keep them.
    """
    if importance < threshold:
        return None
    else:
        return partial_output

def dynamic_cot_controller(moe_outputs):
    """
    Applies partial pruning logic to each MoE output, storing results in a buffer.
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
            partial_reward += 0.05

    return {
        "progressive_buffer": progressive_buffer,
        "updated_tokens": updated_tokens,
        "partial_reward": partial_reward
    }

# ---------------------------------------------------------------------
# (5) AUTO-REGRESSIVE DECODING (Iterative Blocks)
# ---------------------------------------------------------------------
def token_selection(buffered_coT, decode_params):
    if not buffered_coT:
        return []
    return buffered_coT[-1]["payload"]

def iterative_generation(selected_block, decode_state):
    new_tokens = []
    for val in selected_block:
        offset = random.uniform(-0.05, 0.05)
        new_tokens.append(val + offset)
    return new_tokens

def adjusting_pruning_blocks(pruned_block, expanded_block, partial_reward):
    """
    Merges or discards expansions based on partial reward threshold.
    """
    threshold = 0.15
    if partial_reward < threshold:
        return pruned_block
    else:
        return pruned_block + expanded_block

def auto_regressive_decoding(cot_controller_output, decode_params):
    partial_reward = cot_controller_output.get("partial_reward", 0.05)
    progressive_buffer = cot_controller_output.get("progressive_buffer", [])
    decode_state = {}

    selected_block = token_selection(progressive_buffer, decode_params)
    pruned_block = [x * 0.9 for x in selected_block]
    expanded_block = iterative_generation(selected_block, decode_state)

    final_blocks = adjusting_pruning_blocks(pruned_block, expanded_block, partial_reward)
    intermediate_cot = {
        "coT_intermediate": final_blocks,
        "partial_reward": partial_reward + 0.02
    }
    return intermediate_cot

# ---------------------------------------------------------------------
# (6) HIERARCHICAL CoT ASSEMBLY
# ---------------------------------------------------------------------
def macro_summaries(coT_data):
    return coT_data[::2]

def micro_details(coT_data):
    return coT_data[1::2]

def reward_context_map(macro, micro, partial_reward):
    r_map = {}
    idx = 0
    for val in macro + micro:
        r_map[idx] = partial_reward
        idx += 1
    return r_map

def reward_aligned_refinement(macro, micro, r_map):
    final_rep = []
    for i, val in enumerate(macro + micro):
        if r_map[i] > 0.1:
            final_rep.append(val)
        else:
            final_rep.append(val * 0.8)
    return final_rep

def hierarchical_cot_assembly(intermediate_cot):
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

# ---------------------------------------------------------------------
# (7) FINAL OUTPUT GENERATION
# ---------------------------------------------------------------------
def produce_final_answer(refined_structure):
    """
    Summarize or convert final CoT to an answer string.
    """
    return " ".join(str(round(x, 3)) for x in refined_structure)

def integrated_rl_reward_loop(final_answer, partial_reward):
    max_token = 0.0
    tokens = final_answer.split()
    for t in tokens:
        if t.replace('.', '', 1).isdigit():
            val = float(t)
            if val > max_token:
                max_token = val
    episode_reward = partial_reward
    if max_token > 0.9:
        episode_reward += 0.1
    return {
        "updated_reward": episode_reward
    }

def final_output_generation(hier_cot_result, enable_loop=False):
    partial_reward = hier_cot_result["partial_reward"]
    refined_structure = hier_cot_result["final_structure"]

    answer = produce_final_answer(refined_structure)
    print("**Final Answer**:", answer)

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
    if not rl_feedback:
        return cot_controller_state
    if "partial_reward" in cot_controller_state:
        cot_controller_state["partial_reward"] += rl_feedback["updated_reward"]
    return cot_controller_state

# ---------------------------------------------------------------------
#             EXPERIMENTAL HARNESS (ALL MODULES INTACT)
# ---------------------------------------------------------------------
def run_dynamic_cot_full_pipeline(question_text, enable_loop=False):
    """
    This function runs the entire pipeline with a given question or
    problem statement, maintaining all subcomponents for the dynamic CoT.
    It returns a dict of metrics for experiment logging.
    """
    start_time = time.time()

    # 1) USER INPUT
    # If you want to override the user input function:
    # user_text = get_user_input()
    # but we pass question_text directly
    user_text = question_text

    # 2) TOKEN & EMBED
    embedding_table = build_embedding_table()
    tokens = simple_tokenizer(user_text)
    embedded_seq = embed_tokens(tokens, embedding_table)

    # 3) MoE Stack
    moe_out = moe_transformer_stack(embedded_seq)

    # 4) Dynamic CoT
    cot_result = dynamic_cot_controller(moe_out)

    # 5) Auto-Reg Decoding
    decode_params = {"some_decode_param": "placeholder"}
    intermediate_cot = auto_regressive_decoding(cot_result, decode_params)

    # 6) Hierarchical CoT Assembly
    hier_cot_result = hierarchical_cot_assembly(intermediate_cot)

    # 7) Final Output
    final_result = final_output_generation(hier_cot_result, enable_loop=enable_loop)

    end_time = time.time()
    total_time = end_time - start_time

    # gather minimal metrics
    metrics = {
        "question": user_text,
        "tokens": tokens,
        "final_answer": final_result["final_answer"],
        "time_spent": round(total_time, 4),
        "partial_reward": cot_result["partial_reward"],
        "rl_feedback": final_result["rl_feedback"]
    }

    return metrics

# ---------------------------------------------------------------------
# OPTIONAL: Example function to integrate MIT linear algebra questions
# ---------------------------------------------------------------------
def run_experiment_on_questions(questions, enable_loop=False):
    """
    A simple example that runs the entire dynamic CoT pipeline on a list
    of textual questions, returning the results for each.
    """
    results = []
    for q in questions:
        res = run_dynamic_cot_full_pipeline(q, enable_loop=enable_loop)
        results.append(res)
    return results

# ---------------------------------------------------------------------
# MAIN (Sample usage)
# ---------------------------------------------------------------------
def main():
    # Example: If you want to demonstrate the code with some linear algebra tasks:
    sample_questions = [
        "MIT Q1: Find the determinant of a 2x2 matrix with entries a,b,c,d.",
        "MIT Q2: Solve the eigenvalues of a 3x3 matrix with certain row structures."
    ]

    # run the dynamic CoT pipeline in full experimental mode
    experiment_results = run_experiment_on_questions(sample_questions, enable_loop=False)
    for r in experiment_results:
        print("\n=== Experiment Result ===")
        print("Question:", r["question"])
        print("Tokens:", r["tokens"])
        print("Answer:", r["final_answer"])
        print("Time Spent:", r["time_spent"], "sec")
        print("Partial Reward:", r["partial_reward"])
        print("RL Feedback:", r["rl_feedback"])

if __name__ == "__main__":
    main()
