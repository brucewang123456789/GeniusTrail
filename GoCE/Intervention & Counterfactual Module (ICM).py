###############################################################################
#  ICM – Interventional KL Consistency (IKLC) demo (pure Python) #
###############################################################################

import math, random

# --- softmax and KL helpers --------------------------------------------------
def softmax(logits, tau=1.0):
    m = max(logits)
    exps = [math.exp((l - m) / tau) for l in logits]
    Z = sum(exps)
    return [e / Z for e in exps]

def kl_div(p, q):
    return sum(pi * math.log(max(pi / max(qi, 1e-12), 1e-12)) for pi, qi in zip(p, q))

# --- tiny backbone (linear classifier) ---------------------------------------
def backbone_forward(H, W):
    """H: T×d  → logits over  vocab_size (here 3) via mean pooling."""
    T, d = len(H), len(H[0])
    pooled = [sum(H[t][j] for t in range(T)) / T for j in range(d)]
    logits = [sum(wj * pooled[j] for j, wj in enumerate(w_row)) for w_row in W]
    return logits  # length = vocab_size

# --- ICM loss ---------------------------------------------------------------
def iklc_loss(H, A, W,   # latents, causal mask (unused here), weights
              tau_cf=0.5,
              lambda_delta=0.1):
    """
    • Sample S (one index) and noise value v.
    • Run baseline forward, run cf forward, compute KL + |Δ| loss.
    """
    # choose a random latent index to clamp
    T, d = len(H), len(H[0])
    idx = random.randrange(T)
    v_noise = [random.uniform(-1, 1) for _ in range(d)]

    # baseline
    logits0 = backbone_forward(H, W)
    p0 = softmax(logits0, tau=1.0)

    # counterfactual: clone H, clamp row idx to v_noise
    H_cf = [row[:] for row in H]
    H_cf[idx] = v_noise
    logits1 = backbone_forward(H_cf, W)
    p1 = softmax(logits1, tau=tau_cf)

    # KL + absolute diff of expected indices
    kl = kl_div(p0, p1)
    exp0 = sum(i * pi for i, pi in enumerate(p0))
    exp1 = sum(i * pi for i, pi in enumerate(p1))
    delta = abs(exp0 - exp1)

    return kl + lambda_delta * delta

# --- demo --------------------------------------------------------------------
if __name__ == "__main__":
    random.seed(0)
    T, d, vocab = 5, 8, 3
    # random latents
    H = [[random.uniform(-1, 1) for _ in range(d)] for _ in range(T)]
    # dummy causal mask (not used directly here but available)
    A = [[1 if j <= i else 0 for j in range(T)] for i in range(T)]
    # classifier weights
    W = [[random.uniform(-0.2, 0.2) for _ in range(d)] for _ in range(vocab)]

    loss = iklc_loss(H, A, W)
    print("ICM loss:", round(loss, 4))
