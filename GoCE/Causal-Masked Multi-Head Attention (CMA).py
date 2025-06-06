###############################################################################
#  CSAIL – Causal-Sparse Attention with Interventional Loss #
###############################################################################

import math, random

# ------------------ helpers --------------------------------------------------
def softmax(vec, tau=1.0):
    m = max(vec)
    exps = [math.exp((v - m) / tau) for v in vec]
    s = sum(exps)
    return [e / s for e in exps]

def matmul(A, B):
    m, k = len(A), len(A[0])
    n = len(B[0])
    out = [[0.0]*n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            out[i][j] = sum(A[i][t]*B[t][j] for t in range(k))
    return out

def attention(Q, K, V, A, tau=1.0):
    T, dk = len(Q), len(Q[0])
    scale = 1.0 / math.sqrt(dk)
    out = [[0.0]*dk for _ in range(T)]
    for i in range(T):
        logits = [
            (scale * sum(Q[i][u]*K[j][u] for u in range(dk))
             if A[i][j] else -1e9)
            for j in range(T)
        ]
        w = softmax(logits, tau)
        for j in range(T):
            for u in range(dk):
                out[i][u] += w[j]*V[j][u]
    return out, w  # return last weights row for KL demo

def l0_norm(mat):
    return sum(1 for row in mat for x in row if abs(x) > 1e-8)

# ------------------ CSAIL core ----------------------------------------------
def csail_forward(H, A, WQ, WK, WV, tau=1.0):
    Q = matmul(H, WQ)
    K = matmul(H, WK)
    V = matmul(H, WV)
    Z, _ = attention(Q, K, V, A, tau)
    return Z

def csail_loss(H, A, WQ, WK, WV,
               gold_logits,
               tau_train=1.0, tau_cf=0.5,
               lam_l0=1e-4):
    # train pass
    Qtr = matmul(H, WQ); Ktr = matmul(H, WK); Vtr = matmul(H, WV)
    Ztr, w_tr = attention(Qtr, Ktr, Vtr, A, tau_train)

    # counter-factual pass (crisper mask)
    Zcf, w_cf = attention(Qtr, Ktr, Vtr, A, tau_cf)

    # toy prediction loss: squared difference to gold logits
    pred_loss = sum((z-g)**2 for row, g_row in zip(Ztr, gold_logits)
                               for z, g in zip(row, g_row))

    # KL between train and cf weights (only use first row for brevity)
    kl = sum(w_tr[j] * math.log(max(w_tr[j]/max(w_cf[j],1e-9), 1e-9))
             for j in range(len(w_tr)))

    # L0 sparsity
    l0 = l0_norm(WQ) + l0_norm(WK)

    return pred_loss + kl + lam_l0*l0

# ------------------ demo -----------------------------------------------------
if __name__ == "__main__":
    random.seed(0)
    T, d, dk = 4, 6, 4
    H   = [[random.uniform(-1,1) for _ in range(d)] for _ in range(T)]
    WQ  = [[random.uniform(-0.3,0.3) for _ in range(dk)] for _ in range(d)]
    WK  = [[random.uniform(-0.3,0.3) for _ in range(dk)] for _ in range(d)]
    WV  = [[random.uniform(-0.3,0.3) for _ in range(dk)] for _ in range(d)]
    A   = [[1 if j<=i else 0 for j in range(T)] for i in range(T)]
    gold= [[0.0]*dk for _ in range(T)]  # dummy target

    print("Loss:", round(csail_loss(H, A, WQ, WK, WV, gold), 4))
