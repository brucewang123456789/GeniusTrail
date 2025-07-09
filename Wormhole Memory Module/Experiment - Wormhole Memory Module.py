# Experimental Group Code

class WormholeMemoryModule:
    def __init__(self, momentum_alpha=0.3, decay_lambda=0.01):
        self.momentum_alpha = momentum_alpha
        self.decay_lambda = decay_lambda
        self.memory_store = {}

    def _l2_distance(self, a, b):
        return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5

    def _apply_momentum_decay(self, old_vec, new_vec):
        updated = [
            (1 - self.momentum_alpha) * o + self.momentum_alpha * n
            for o, n in zip(old_vec, new_vec)
        ]
        return [(1 - self.decay_lambda) * v for v in updated]

    def store_or_update(self, user_id, conv_id, topic_vec, data_vec):
        key = (user_id, conv_id, tuple(topic_vec))
        if key in self.memory_store:
            old_vec = self.memory_store[key]
            self.memory_store[key] = self._apply_momentum_decay(old_vec, data_vec)
        else:
            self.memory_store[key] = list(data_vec)

    def retrieve(self, user_id, topic_vec, top_k=2):
        candidates = [
            (self._l2_distance(topic_vec, list(key[2])), value)
            for key, value in self.memory_store.items()
            if key[0] == user_id
        ]
        candidates.sort(key=lambda x: x[0])
        top_matches = candidates[:top_k]

        if not top_matches:
            return None

        merged_result = [0] * len(top_matches[0][1])
        for _, vec in top_matches:
            merged_result = [m + v for m, v in zip(merged_result, vec)]
        return [v / len(top_matches) for v in merged_result]

# Experimental Setup
if __name__ == "__main__":
    memory_module = WormholeMemoryModule(momentum_alpha=0.3, decay_lambda=0.01)

    # Simulate data for multiple dialogues
    memory_module.store_or_update(user_id=1, conv_id=101, topic_vec=[0.5, -0.3], data_vec=[0.7, 0.2])
    memory_module.store_or_update(user_id=1, conv_id=102, topic_vec=[0.6, -0.2], data_vec=[0.9, -0.1])

    # Retrieve memory
    retrieval_result = memory_module.retrieve(user_id=1, topic_vec=[0.55, -0.25], top_k=2)

    print("Retrieved Memory:", retrieval_result)

    # Debug output
    for key, value in memory_module.memory_store.items():
        print(f"Key: {key}, Value: {value}")
