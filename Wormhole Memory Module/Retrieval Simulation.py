# retrieval_simulation_decoder_only.py

import math

class DecoderWormholeMemory:
    def __init__(self, momentum_alpha=0.3, decay_lambda=0.01):
        self.momentum_alpha = momentum_alpha
        self.decay_lambda = decay_lambda
        # key = (user_id, conv_id, topic_tuple)
        # val = list representing memory vector
        self.memory_db = {}

    def _l2_dist(self, a, b):
        total = 0.0
        for x, y in zip(a, b):
            diff = x - y
            total += diff*diff
        return math.sqrt(total)

    def _momentum_decay(self, oldv, newv):
        dist = self._l2_dist(oldv, newv)
        threshold = 1.0
        updated = list(oldv)
        if dist > threshold:
            for i in range(len(updated)):
                updated[i] = (1.0 - self.momentum_alpha)*updated[i] + self.momentum_alpha*newv[i]
        for i in range(len(updated)):
            updated[i] *= (1.0 - self.decay_lambda)
        return updated

    def store_or_update(self, user_id, conv_id, topic_vec, data_vec):
        key = (user_id, conv_id, tuple(topic_vec))
        if key not in self.memory_db:
            self.memory_db[key] = list(data_vec)
        else:
            old_data = self.memory_db[key]
            new_data = self._momentum_decay(old_data, data_vec)
            self.memory_db[key] = new_data

    def retrieve_cross_session(self, user_id, topic_vec, top_k=2):
        # Only match same user; measure distance w.r.t. topic
        matches = []
        for (u, c, ttuple), val in self.memory_db.items():
            if u == user_id:
                d = self._l2_dist(list(ttuple), topic_vec)
                matches.append((d, val))

        if not matches:
            return None

        matches.sort(key=lambda x: x[0])
        selected = matches[:top_k]

        out_len = len(selected[0][1])
        merged = [0.0]*out_len
        for _, memv in selected:
            for i in range(out_len):
                merged[i] += memv[i]
        for i in range(out_len):
            merged[i] /= float(len(selected))

        return merged

    def debug_memory(self):
        print("===== [Decoder Wormhole Memory] =====")
        for k, v in self.memory_db.items():
            print(f" Key={k}, partial_val={v[:2]} len={len(v)}")
        print("=====================================\n")


# Minimal demonstration
if __name__ == "__main__":
    dwm = DecoderWormholeMemory(momentum_alpha=0.3, decay_lambda=0.01)

    # Example: user=1, conv_id=10 => store
    dwm.store_or_update(user_id=1, conv_id=10, topic_vec=[1.0, -0.5], data_vec=[0.9, 0.2])
    dwm.debug_memory()

    # Suppose new conflicting data for same conv_id
    dwm.store_or_update(user_id=1, conv_id=10, topic_vec=[1.0, -0.5], data_vec=[2.0, -1.0])
    dwm.debug_memory()

    # Another conv for the same user => conv_id=11
    dwm.store_or_update(user_id=1, conv_id=11, topic_vec=[0.8, -0.6], data_vec=[-0.3, 1.0])
    dwm.debug_memory()

    # Cross-session retrieval for user=1, new topic
    ret = dwm.retrieve_cross_session(user_id=1, topic_vec=[1.05, -0.55], top_k=2)
    print("Retrieval across conv10 & conv11 =>", ret)
    print("Demo complete.")
