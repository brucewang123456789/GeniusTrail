"""
Plain Python 3.x SINGLE FILE:
 - Combines the detailed Wormhole Memory Module (A/B/C/D)
   AND a minimal train phase so there's no "ModuleNotFoundError."
 - No external plugins. 
 - Prints out intermediate steps (memory store content, retrieval, 
   naive training logs) to prove it runs.

Usage:
    python wormhole_train_single_file.py
"""

import math

# =====================================================
#  (1) Wormhole Memory Module (Detailed)
# =====================================================
class WormholeMemoryModule:
    """
    Implements a 'Rubik's Cube'-style multi-axis memory with:
      (A) Multi-axis index/cache,
      (B) Momentum & decay update,
      (C) Cross-dialogue retrieval,
      (D) Merging with main hidden states.

    All in plain Python. Provides debug prints to show memory changes.
    """
    def __init__(self, momentum_alpha=0.3, decay_lambda=0.01):
        self.momentum_alpha = momentum_alpha
        self.decay_lambda = decay_lambda
        # memory_store: dict
        #   key = (user_id, conv_id, topic_vec_tuple, time_stamp, etc.)
        #   value = list of floats (the memory vector)
        self.memory_store = {}

    # ------------ (B) momentum/decay logic ------------
    def _l2_norm(self, vec1, vec2):
        dist_sq = 0.0
        length = min(len(vec1), len(vec2))
        for i in range(length):
            diff = vec1[i] - vec2[i]
            dist_sq += diff * diff
        return math.sqrt(dist_sq)

    def _momentum_decay_merge(self, old_vec, new_vec):
        updated = list(old_vec)
        threshold = 1.0
        dist_val = self._l2_norm(old_vec, new_vec)

        if dist_val > threshold:
            alpha = self.momentum_alpha
            for i in range(len(updated)):
                updated[i] = (1 - alpha)*updated[i] + alpha*new_vec[i]

        # apply decay
        for i in range(len(updated)):
            updated[i] *= (1.0 - self.decay_lambda)
        return updated

    # ------------ (A) store/update ------------
    def store_or_update(self, key, data_vector):
        """
        If key doesn't exist, store data_vector directly;
        else do momentum & decay update with the old vector.
        """
        if key not in self.memory_store:
            self.memory_store[key] = list(data_vector)
        else:
            old_vec = self.memory_store[key]
            updated = self._momentum_decay_merge(old_vec, list(data_vector))
            self.memory_store[key] = updated

    # ------------ (C) retrieval ------------
    def retrieve(self, query_key, top_k=2):
        """
        Cross-dialogue retrieval by matching user_id
        and measuring L2 distance on topic vector.
        Then average top_k results.
        """
        if 'user_id' not in query_key or 'topic_vec' not in query_key:
            return None

        q_user = query_key['user_id']
        q_topic = query_key['topic_vec']

        candidates = []
        for k, stored_vec in self.memory_store.items():
            if k[0] == q_user:   # match user_id
                stored_topic = k[2] # a tuple
                dist_val = self._l2_norm(stored_topic, q_topic)
                candidates.append((dist_val, stored_vec))

        if not candidates:
            return None

        candidates.sort(key=lambda x: x[0])
        selected = candidates[:top_k]

        out_len = len(selected[0][1])
        merged = [0.0]*out_len
        for _, memv in selected:
            for i in range(out_len):
                merged[i] += memv[i]
        for i in range(out_len):
            merged[i] /= float(len(selected))

        return merged

    # ------------ (D) merging ------------
    def merge_with_current_hidden(self, current_hidden, retrieved_embed):
        scale = 0.5
        out_len = min(len(current_hidden), len(retrieved_embed))
        out_list = []
        for i in range(out_len):
            out_list.append(current_hidden[i] + scale*retrieved_embed[i])
        return out_list

    # debug
    def debug_print_store(self):
        print("=== WormholeMemory Debug Store ===")
        for k, val in self.memory_store.items():
            print(f" Key={k}, Value(first2)={val[:2]} len={len(val)}")
        print("==================================\n")

# =====================================================
#  (2) Main Model + Train Loop (No Plugins)
# =====================================================
class MainModel:
    """
    A toy model with just a single param 'scale' to illustrate 
    naive forward/backward steps. No external libs used.
    """
    def __init__(self):
        self.scale = 1.0

    def forward(self, input_vector):
        # y[i] = scale * input_vector[i]
        out = []
        for val in input_vector:
            out.append(self.scale * val)
        return out

    def compute_loss(self, pred, target):
        # MSE
        length = min(len(pred), len(target))
        loss_val = 0.0
        for i in range(length):
            diff = pred[i] - target[i]
            loss_val += diff*diff
        return loss_val

    def update_scale(self, grad_list):
        # sum grad, apply LR=0.01
        grad_sum = 0.0
        for g in grad_list:
            grad_sum += g
        lr = 0.01
        self.scale -= lr*grad_sum

def train_phase_demo(epochs=2):
    wmm = WormholeMemoryModule(momentum_alpha=0.3, decay_lambda=0.01)
    model = MainModel()

    # minimal dataset: (query_key, data_vector, target_out)
    dataset = [
        (
         {'user_id':1, 'topic_vec':[1.0, -0.5]}, 
         [0.9, 1.2], 
         [0.8, -1.0]
        ),
        (
         {'user_id':1, 'topic_vec':[2.0, 2.0]},
         [1.0, -0.2],
         [0.5, 0.5]
        ),
        (
         {'user_id':2, 'topic_vec':[0.5, 0.5]},
         [2.0, 2.0],
         [1.5, -0.5]
        ),
    ]

    for ep in range(epochs):
        print(f"\n=== Epoch {ep} ===")
        for step_idx, sample in enumerate(dataset):
            query_k, mem_vec, target = sample
            user_id = query_k['user_id']
            topic_tup = tuple(query_k['topic_vec'])
            # key = (user_id, conv_id=0, topic_tup, time=0)
            key = (user_id, 0, topic_tup, 0)

            # (A/B) store/update
            wmm.store_or_update(key, mem_vec)

            # (C) retrieve
            retrieved = wmm.retrieve(query_k, top_k=1)
            if retrieved is None:
                retrieved = [0.0]*len(mem_vec)

            # (D) merge with current hidden
            current_h = [0.2]*len(retrieved)
            combined_input = wmm.merge_with_current_hidden(current_h, retrieved)

            # forward
            pred = model.forward(combined_input)

            # loss
            loss_val = model.compute_loss(pred, target)

            # grad
            grad_list = []
            # dLoss/dPred = 2*(pred[i] - target[i]) for MSE
            for i in range(len(pred)):
                grad_val = 2.0*(pred[i] - target[i])
                grad_list.append(grad_val)
            model.update_scale(grad_list)

            print(f"[Ep={ep}, Step={step_idx}] Loss={loss_val:.3f}, scale={model.scale:.3f}")

        # optional debug
        wmm.debug_print_store()

if __name__ == "__main__":
    print(">>> Starting single-file training demo...")
    train_phase_demo(epochs=2)
    print(">>> Demo complete.")
