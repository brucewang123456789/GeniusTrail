"""
Plain Python 3.x demonstration of "breaking memory barriers"
across multiple dialogue in a decoder-only style scenario.
No external plugins required. Prints out memory store changes
and retrieval results to illustrate cross-session usage.

Steps:
 1) We maintain a global memory (Rubik's Cube style).
 2) Each conversation has a user_id and conv_id, plus some "topic vector."
 3) Momentum & decay are used if new data conflicts with old memory.
 4) A retrieval step can unify data from multiple conv_ids
    belonging to the same user or related topics.

Run in Python 3.13 IDLE with no libraries installed.
"""

import math

class WormholeBarriers:
    def __init__(self, momentum_alpha=0.3, decay_lambda=0.01):
        self.momentum_alpha = momentum_alpha
        self.decay_lambda = decay_lambda
        # memory: dict
        #   key = (user_id, conv_id, topic_tuple)
        #   val = list representing memory vector
        self.memory = {}

    def _l2_dist(self, v1, v2):
        length = min(len(v1), len(v2))
        dist_sq = 0.0
        for i in range(length):
            d = v1[i] - v2[i]
            dist_sq += d*d
        return math.sqrt(dist_sq)

    def _momentum_decay_update(self, old_vec, new_vec):
        # Step1: measure surprise
        threshold = 1.0
        dist_val = self._l2_dist(old_vec, new_vec)
        updated = list(old_vec)

        # Step2: momentum if above threshold
        if dist_val > threshold:
            alpha = self.momentum_alpha
            for i in range(len(updated)):
                updated[i] = (1 - alpha)*updated[i] + alpha*new_vec[i]

        # Step3: decay
        for i in range(len(updated)):
            updated[i] *= (1.0 - self.decay_lambda)

        return updated

    def store_or_update(self, user_id, conv_id, topic_vec, data_vec):
        """
        If key not found, store data_vec directly.
        Else momentum+decay merge with old memory.
        """
        key = (user_id, conv_id, tuple(topic_vec))
        if key not in self.memory:
            self.memory[key] = list(data_vec)
        else:
            old_data = self.memory[key]
            new_data = self._momentum_decay_update(old_data, list(data_vec))
            self.memory[key] = new_data

    def retrieve(self, user_id, topic_vec, top_k=2):
        """
        Attempt cross-session retrieval: we look for all conv_ids
        belonging to 'user_id' and measure topic similarity.
        Then average top_k matches as final retrieval, simulating
        a 'decoder-only' approach that merges data from multiple convs.
        """
        candidates = []
        for k, val in self.memory.items():
            if k[0] == user_id:   # same user
                stored_topic = k[2]
                dist_val = self._l2_dist(stored_topic, topic_vec)
                candidates.append((dist_val, val))

        if not candidates:
            return None

        candidates.sort(key=lambda x: x[0])
        selected = candidates[:top_k]

        length = len(selected[0][1])
        merged = [0.0]*length
        for dist_val, memv in selected:
            for i in range(length):
                merged[i] += memv[i]

        for i in range(length):
            merged[i] /= float(len(selected))

        return merged

    def debug_print(self):
        print("=== Memory Barrier State ===")
        for k, v in self.memory.items():
            # just show partial
            print(" Key=", k, "Value(first2)=", v[:2], " length=", len(v))
        print("=== End Memory State ===\n")

# ------------------- Demo main -------------------
if __name__ == "__main__":
    wbb = WormholeBarriers(momentum_alpha=0.3, decay_lambda=0.01)

    # Suppose we have multiple conversation IDs for the same user
    print("[STEP] Insert data from conv_id=10, user=1, topic_vec=[1.0, -0.5]")
    wbb.store_or_update(user_id=1, conv_id=10, topic_vec=[1.0, -0.5], data_vec=[0.8, 0.9])
    wbb.debug_print()

    print("[STEP] Insert data from conv_id=11, user=1, topic_vec=[1.2, -0.4]")
    wbb.store_or_update(user_id=1, conv_id=11, topic_vec=[1.2, -0.4], data_vec=[2.0, -1.0])
    wbb.debug_print()

    print("[STEP] Insert conflicting data for conv_id=11, user=1")
    wbb.store_or_update(user_id=1, conv_id=11, topic_vec=[1.2, -0.4], data_vec=[-0.5, 2.0])
    wbb.debug_print()

    print("[STEP] Now retrieve cross-session (user=1, topic_vec=[1.05, -0.45])")
    retrieved = wbb.retrieve(user_id=1, topic_vec=[1.05, -0.45], top_k=2)
    print("Retrieved embed =", retrieved)

    print("[STEP] Insert data from conv_id=20, user=2, topic_vec=[0.5,0.5]")
    wbb.store_or_update(user_id=2, conv_id=20, topic_vec=[0.5, 0.5], data_vec=[1.0, 1.0])
    wbb.debug_print()

    print("[STEP] Cross-session retrieval again for user=2, topic_vec=[0.48,0.52]")
    ret2 = wbb.retrieve(user_id=2, topic_vec=[0.48,0.52], top_k=1)
    print("Retrieved embed for user=2:", ret2)

    print("Demo complete. You can see that conv_id=10,11 for user=1 share memory, likewise user=2 is separate but may be extended for multi convs.")
