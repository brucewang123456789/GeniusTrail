import random

# -------------------------------------------------------------------
#  AUTO-REGRESSIVE DECODING with FEEDBACK LOOP (Iterative Small Blocks)
# -------------------------------------------------------------------

def token_selection(buffered_coT, decode_params):
    """
    [Arrow: Progressive Buffer/CoT -> Token Selection]
    Chooses which tokens (or partial blocks) to decode next.
    'decode_params' can include gating logs, partial reward, etc.
    """
    # For demonstration, select tokens randomly from buffered_coT
    # We'll pretend each buffer entry is a 'block'
    if not buffered_coT:
        return []
    # Simple approach: pick the last token-block
    block = buffered_coT[-1]["payload"]
    return block

def iterative_generation(selected_block, decode_state):
    """
    [Arrow: Token Selection -> Iterative Generation]
    Generates expanded tokens in small increments.
    'decode_state' may hold partial reward updates, step counters, etc.
    """
    # For demonstration, we expand the block by appending random small data
    new_tokens = []
    for val in selected_block:
        # Add random offsets
        offset = random.uniform(-0.05, 0.05)
        new_tokens.append(val + offset)

    # Return an 'expanded' block
    return new_tokens

def adjusting_pruning_blocks(pruned_block, expanded_block, partial_reward):
    """
    [Arrow: Pruned/Expanded Blocks -> Adjust/Prune (RL)]
    Incorporates partial RL signals + gating to refine block sets.
    """
    # We'll do a trivial check: if partial_reward < threshold, keep only pruned_block
    # else combine pruned_block + expanded_block
    threshold = 0.15  # placeholder for demonstration
    if partial_reward < threshold:
        return pruned_block
    else:
        return pruned_block + expanded_block

def auto_regressive_decoding(cot_controller_output, decode_params):
    """
    Main function for the Auto-Regressive Decoding module (Iterative Small Blocks).
    [Arrow: Dynamic CoT Controller -> Auto-Reg Decoding]
    'cot_controller_output' should include:
       - progressive_buffer (list of pruned tokens)
       - partial_reward (float)
    """
    partial_reward = 0.05 if "partial_reward" not in cot_controller_output else cot_controller_output["partial_reward"]
    progressive_buffer = cot_controller_output.get("progressive_buffer", [])
    decode_state = {}

    # 1) Token Selection
    selected_block = token_selection(progressive_buffer, decode_params)

    # For demonstration, let's define a trivial 'pruned_block'
    # In real usage, you'd pass partial_out or pruned tokens from dynamic CoT
    pruned_block = [x * 0.9 for x in selected_block]

    # 2) Iterative Generation
    expanded_block = iterative_generation(selected_block, decode_state)

    # 3) Adjusting / Pruning (Incorporates RL + importance)
    final_blocks = adjusting_pruning_blocks(pruned_block, expanded_block, partial_reward)

    # Build an intermediate CoT from final_blocks
    intermediate_cot = {
        "coT_intermediate": final_blocks,
        "partial_reward": partial_reward + 0.02  # let's nudge reward up slightly
    }

    return intermediate_cot

# -------------------------------------------------------------------
#  HIERARCHICAL CoT ASSEMBLY
# -------------------------------------------------------------------

def macro_summaries(coT_data):
    """
    [Arrow: Intermediate CoT -> Macro Summaries]
    Extracts or constructs top-level reasoning steps from coT_data
    """
    # We'll pretend to pick every other element to form a 'macro' outline
    macro = coT_data[::2]
    return macro

def micro_details(coT_data):
    """
    [Arrow: Intermediate CoT -> Micro Details]
    Keeps fine-grained expansions
    """
    # We'll pick the 'in-between' elements as micro
    micro = coT_data[1::2]
    return micro

def reward_context_map(macro, micro, partial_reward):
    """
    [Arrow: Macro+Micro -> Reward-Context Map]
    Associates reward signals with macro/micro steps
    """
    # We'll just produce a dictionary mapping indices to partial_reward for demonstration
    r_map = {}
    idx = 0
    for val in macro + micro:
        r_map[idx] = partial_reward
        idx += 1
    return r_map

def reward_aligned_refinement(macro, micro, r_map):
    """
    [Arrow: Reward-Context Map -> Reward-Aligned Refinement]
    Potentially reorder or highlight tokens based on reward map
    """
    # We'll do a trivial approach: if reward > 0.1, keep them as is,
    # else reduce them (just a demonstration)
    final_rep = []
    for i, val in enumerate(macro + micro):
        if r_map[i] > 0.1:
            final_rep.append(val)
        else:
            # compress it
            final_rep.append(val * 0.8)
    return final_rep

def hierarchical_cot_assembly(intermediate_cot):
    """
    Main function for Hierarchical CoT Assembly.
    [Arrow: Auto-Reg Decoding -> Hierarchical CoT Assembly]
    'intermediate_cot' should hold coT_intermediate + partial_reward
    """
    partial_reward = intermediate_cot["partial_reward"]
    coT_seq = intermediate_cot["coT_intermediate"]

    # 1) Macro Summaries
    macro = macro_summaries(coT_seq)

    # 2) Micro Details
    micro = micro_details(coT_seq)

    # 3) Reward-Context Map
    r_map = reward_context_map(macro, micro, partial_reward)

    # 4) Reward-Aligned Refinement
    final_structure = reward_aligned_refinement(macro, micro, r_map)

    return {
        "final_structure": final_structure,
        "partial_reward": partial_reward
    }

# -------------------------------------------------------------------
#  FINAL OUTPUT GENERATION & Potential Loop to Dynamic CoT
# -------------------------------------------------------------------

def produce_final_answer(refined_structure):
    """
    [Arrow: Hierarchical CoT Assembly -> Final Answer]
    Convert refined CoT into a user-facing response.
    """
    # We'll just join them as a string for demonstration
    # (Pretend each float is a token or word index).
    answer_str = " ".join(str(round(x, 3)) for x in refined_structure)
    return answer_str

def integrated_rl_reward_loop(final_answer, partial_reward):
    """
    [Arrow: Final Output -> Dynamic CoT Controller (optional)]
    If continuing RL training, we compute or simulate an episodic reward
    and update gating/pruning parameters.
    """
    # We'll do a naive approach: if the final answer includes a large positive token, reward is bigger
    max_token = max(float(x) for x in final_answer.split() if x.replace('.', '', 1).isdigit()) if final_answer else 0.0
    episode_reward = partial_reward
    if max_token > 0.9:
        episode_reward += 0.1
    # This would feed back into dynamic CoT gating logic
    dynamic_cot_update = {
        "updated_reward": episode_reward
    }
    return dynamic_cot_update

def final_output_generation(hier_cot_result, enable_loop=False):
    """
    Main function for Final Output Generation.
    [Arrow: Hierarchical CoT Assembly -> Final Output]
    Optionally loops back to Dynamic CoT if enable_loop = True.
    """
    partial_reward = hier_cot_result["partial_reward"]
    refined_structure = hier_cot_result["final_structure"]

    # 1) Final Answer
    answer = produce_final_answer(refined_structure)

    # Print or store
    print("**Final Answer**:", answer)

    # 2) Integrated RL Reward
    if enable_loop:
        feedback = integrated_rl_reward_loop(answer, partial_reward)
        return {
            "final_answer": answer,
            "rl_feedback": feedback  # This can be used to update Dynamic CoT gating
        }
    else:
        return {
            "final_answer": answer,
            "rl_feedback": None
        }

# -------------------------------------------------------------------
#  OPTIONAL: Dynamic CoT Controller Loop-Back
#  If we want to simulate how the RL feedback updates gating/pruning
# -------------------------------------------------------------------
def dynamic_cot_update_controller(cot_controller_state, rl_feedback):
    """
    Takes gating/pruning thresholds in cot_controller_state
    and updates them with rl_feedback.
    [Arrow: Integrated RL Reward -> Dynamic CoT Controller]
    """
    if not rl_feedback:
        return cot_controller_state
    # Example: increment partial_reward for next run
    if "partial_reward" in cot_controller_state:
        cot_controller_state["partial_reward"] += rl_feedback["updated_reward"]
    return cot_controller_state

# -------------------------------------------------------------------
#  MAIN EXECUTION (Append to the prior pipeline)
# -------------------------------------------------------------------
def main():
    """
    This function assumes the pipeline up to 'dynamic_cot_controller'
    is already done. We resume from 'cot_result' that the controller output
    and carry it all the way to final output, plus RL loop.
    """
    # Mimic the result from dynamic_cot_controller
    cot_result = {
        "progressive_buffer": [
            {"payload": [0.55, -0.2, 0.9, 0.1]},
            {"payload": [0.15, 0.35, -0.05]}
        ],
        "partial_reward": 0.12
    }

    decode_params = {
        "some_decode_param": "placeholder"
    }

    # -------------------------------------------------------------------
    # 1) AUTO-REGRESSIVE DECODING
    # -------------------------------------------------------------------
    intermediate_cot = auto_regressive_decoding(cot_result, decode_params)

    # -------------------------------------------------------------------
    # 2) HIERARCHICAL CoT ASSEMBLY
    # -------------------------------------------------------------------
    hier_cot_result = hierarchical_cot_assembly(intermediate_cot)

    # -------------------------------------------------------------------
    # 3) FINAL OUTPUT GENERATION
    # -------------------------------------------------------------------
    final_result = final_output_generation(hier_cot_result, enable_loop=True)

    # If a RL loop is needed, update dynamic CoT gating
    if final_result["rl_feedback"]:
        updated_cot = dynamic_cot_update_controller(cot_result, final_result["rl_feedback"])
        print("Updated Dynamic CoT State:", updated_cot)

if __name__ == "__main__":
    main()
