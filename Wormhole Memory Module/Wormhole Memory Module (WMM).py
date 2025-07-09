"""
A plain Python 3.13 version of the Wormhole Memory Module (WMM),
with no external libraries (like numpy). This code demonstrates:

1) Momentum-based update & decay,
2) Multi-axis memory store (like a "Rubik's Cube" for user memory),
3) Simple retrieval mechanism,
4) Printing outputs to verify the memory is updated and retrievable.

Run it in your Python 3.13 IDLE without any plugins to see printed results.
"""

import math

class WormholeMemoryModule:
    def __init__(self, momentum_alpha=0.3, decay_lambda=0.01):
        """
        momentum_alpha: coefficient for momentum update (0 < alpha < 1).
        decay_lambda:   rate for simple decay after an update (0 < lambda < 1).

        The memory_store is a dictionary:
          key = (user_id, conv_id, topic_vec_as_tuple, time_stamp)
          val = a list or tuple representing the stored memory vector
        We do not rely on numpy; all operations are basic Python.
        """
        self.momentum_alpha = momentum_alpha
        self.decay_lambda = decay_lambda
        self.memory_store = {}

    def _l2_norm(self, vec1, vec2):
        """
        Compute L2 distance between two lists (or tuples) of equal length,
        using basic Python. No external libs needed.
        """
        dist_sq = 0.0
        length = min(len(vec1), len(vec2))
        for i in range(length):
            diff = vec1[i] - vec2[i]
            dist_sq += diff * diff
        return math.sqrt(dist_sq)

    def _apply_momentum_decay(self, old_vec, new_vec):
        """
        Combine old_vec with new_vec using momentum_alpha if "surprise" is large,
        then apply a decay factor. No external library used.
        Return the updated vector as a list of floats.
        """
        surprise_threshold = 1.0  # you can tune
        dist = self._l2_norm(old_vec, new_vec)
        updated = list(old_vec)  # make a copy

        if dist > surprise_threshold:
            # momentum update
            alpha = self.momentum_alpha
            for i in range(len(updated)):
                updated[i] = (1 - alpha) * updated[i] + alpha * new_vec[i]

        # apply weight decay
        for i in range(len(updated)):
            updated[i] *= (1.0 - self.decay_lambda)

        return updated

    def update_memory(self, key, data_vector):
        """
        If the key doesn't exist, we store data_vector directly.
        If the key does exist, we do momentum_decay merge with old memory.
        """
        if key not in self.memory_store:
            # directly store as a list
            self.memory_store[key] = list(data_vector)
        else:
            old_vec = self.memory_store[key]
            new_vec = self._apply_momentum_decay(old_vec, list(data_vector))
            self.memory_store[key] = new_vec

    def retrieve_multi_axis(self, query_key, top_k=2):
        """
        A simplified retrieval that tries to find the closest memory vectors
        to query_key's topic vector, using L2 distance.

        query_key might be a dict: {
           'user_id': 1,
           'conv_id': 10,
           'topic_vec': [1.2, -0.9, ...],
           'time': 0
        }

        We'll do:
         1) match user_id exactly
         2) compute L2 distance between stored topic_vec and query's topic_vec
         3) pick top_k
         4) average them as final result
        """
        if 'user_id' not in query_key or 'topic_vec' not in query_key:
            return None

        q_user = query_key['user_id']
        q_topic_vec = query_key['topic_vec']

        # build a list of (distance, memory_vector)
        candidates = []
        for k, stored_vec in self.memory_store.items():
            # k is (user_id, conv_id, topic_vec_as_tuple, time_stamp)
            if k[0] == q_user:  # same user
                stored_topic = k[2]  # a tuple
                # compute distance
                d = self._l2_norm(stored_topic, q_topic_vec)
                candidates.append((d, stored_vec))

        if not candidates:
            return None

        # sort by distance
        candidates.sort(key=lambda x: x[0])
        selected = candidates[:top_k]
        # average them
        out_len = len(selected[0][1])
        count = len(selected)
        merged = [0.0]*out_len

        for dist_val, memv in selected:
            for i in range(out_len):
                merged[i] += memv[i]

        for i in range(out_len):
            merged[i] /= float(count)

        return merged

    def merge_with_current(self, current_hidden, retrieved_embed):
        """
        Merging the retrieved memory into current hidden via a simple scale addition.
        E.g., h' = h + 0.5*retrieved
        """
        scale = 0.5
        out_len = min(len(current_hidden), len(retrieved_embed))
        merged = []
        for i in range(out_len):
            merged_val = current_hidden[i] + scale * retrieved_embed[i]
            merged.append(merged_val)
        return merged

    def debug_print_memory(self):
        """
        Print the entire memory store in a readable manner,
        so user can see what's inside.
        """
        print("=== Wormhole Memory Store ===")
        for k, val in self.memory_store.items():
            print(f"  Key={k}, Value(first 3)={val[:3]} ... (len={len(val)})")
        print("=== End of Memory ===\n")

# ------------------------- DEMO / TEST MAIN -------------------------
if __name__ == "__main__":
    # We'll create a WMM instance
    wmm = WormholeMemoryModule(momentum_alpha=0.3, decay_lambda=0.01)

    # We'll simulate a user query with a topic vector, and do some memory updates

    # 1) Prepare some data
    user_id = 1
    conv_id = 10
    topic_vec_example = [1.0, -1.0, 0.5]  # purely for demonstration
    time_stamp = 0
    key_for_memory = (user_id, conv_id, tuple(topic_vec_example), time_stamp)

    # 2) Let's store a 'data_vector' as memory
    old_data = [0.9, -0.8, 0.4]
    wmm.update_memory(key_for_memory, old_data)
    print("[STEP] Updated memory with old_data")
    wmm.debug_print_memory()

    # 3) Suppose we have a new_data that differs
    new_data = [2.0, 1.5, -0.3]  # quite different
    wmm.update_memory(key_for_memory, new_data)
    print("[STEP] Updated memory with new_data (momentum & decay merge expected)")
    wmm.debug_print_memory()

    # 4) Let's try retrieving
    query_key = {
       'user_id': user_id,
       'conv_id': conv_id,
       'topic_vec': [1.05, -0.95, 0.45],
       'time': 0
    }
    retrieved_vec = wmm.retrieve_multi_axis(query_key, top_k=1)
    print(f"[STEP] Retrieved vector = {retrieved_vec}")

    # 5) Let's pretend we have a current hidden state
    current_h = [0.5, 0.5, 0.5]
    if retrieved_vec is not None:
        merged_result = wmm.merge_with_current(current_h, retrieved_vec)
        print(f"[STEP] Merged result = {merged_result}")
    else:
        print("[STEP] Nothing retrieved, skipping merge.")
