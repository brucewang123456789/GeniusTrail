###############################################################################
# CMoE – Causal-Conditioned Sparse Expert FFN (reference, pure Python) #
###############################################################################

import math, random

# ---------- tiny FFN expert --------------------------------------------------
def make_ffn(dim_in, dim_hidden):
    W1 = [[random.uniform(-0.2, 0.2) for _ in range(dim_in)]
          for _ in range(dim_hidden)]
    b1 = [0.0]*dim_hidden
    W2 = [[random.uniform(-0.2, 0.2) for _ in range(dim_hidden)]
          for _ in range(dim_in)]
    b2 = [0.0]*dim_in
    def gelu(x): return 0.5*x*(1+math.tanh(0.7978846*(x+0.044715*x*x*x)))
    def ffn(x):
        h = [gelu(sum(wi*xi for wi, xi in zip(row, x))+bi)
             for row, bi in zip(W1, b1)]
        return [sum(wi*hi for wi, hi in zip(col, h))+bo
                for col, bo in zip(zip(*W2), b2)]
    return ffn

# ---------- router -----------------------------------------------------------
def router_logits(h, experts):
    """Return a list of logits (here dot with fixed vector per expert)."""
    logits = []
    for e in experts:
        w = e['router_w']
        logits.append(sum(wi*xi for wi, xi in zip(w, h)))
    return logits

# ---------- causal-conditioned MoE ------------------------------------------
def cmoe_forward(Z, A, experts, k=1):
    """
    Z : T×d  attended latents
    A : T×T  causal adjacency (binary upper-triangular)
    experts : list of {'ffn': callable, 'router_w': vector}
    k : top-k experts per token (1 here)
    """
    T, d = len(Z), len(Z[0])
    out = [[0.0]*d for _ in range(T)]
    usage = [0]*len(experts)

    on_vicinity = [set() for _ in range(T)]  # experts used so far in V_t

    for t in range(T):
        # build causal vicinity set indices
        Vt = {j for j in range(T) if A[t][j]==1 or A[j][t]==1}
        allowed = set().union(*(on_vicinity[j] for j in Vt)) if Vt else set(range(len(experts)))

        # compute logits
        logits = router_logits(Z[t], experts)
        # mask
        masked = [logits[i] if i in allowed else -1e9 for i in range(len(experts))]
        # top-k (k=1)
        best = max(range(len(experts)), key=lambda i: masked[i])
        chosen = [best]
        usage[best] += 1
        # record expert used for vicinity propagation
        on_vicinity[t].update(chosen)
        # softmax over chosen (degenerates to 1 when k=1)
        probs = [1.0]
        # compute output
        for p, idx in zip(probs, chosen):
            contrib = experts[idx]['ffn'](Z[t])
            out[t] = [ot + p*ct for ot, ct in zip(out[t], contrib)]
    return out, usage

# ---------- demo -------------------------------------------------------------
if __name__ == "__main__":
    random.seed(0)
    T, d, d_hidden, E = 5, 8, 16, 3
    Z = [[random.uniform(-1,1) for _ in range(d)] for _ in range(T)]
    A = [[1 if j<=i else 0 for j in range(T)] for i in range(T)]
    experts = []
    for _ in range(E):
        experts.append({
            'ffn': make_ffn(d, d_hidden),
            'router_w': [random.uniform(-0.5,0.5) for _ in range(d)]
        })
    H_prime, usage = cmoe_forward(Z, A, experts, k=1)
    print("Output row 0 (rounded):", [round(x,3) for x in H_prime[0]])
    print("Expert usage counts   :", usage)
