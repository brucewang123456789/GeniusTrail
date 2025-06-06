###############################################################################
#  Graph-of-Causal-Evolution (CoCE) Transformer Demo
#  Pure Python 3.13 IDLE‒ no third-party libraries
#
#  Core Modules (all present):
#    1. Latent-Causal Graph Builder        (LCG)
#    2. Causal-Masked Multi-Head Attention (CMA, 4 heads, rotary)
#    3. Causal-Conditioned Sparse-Expert FFN (C-MoE, 4 experts)
#    4. Intervention & Counterfactual loss (IKLC)
#    5. Self-Evolution Gate               (SEG / CAER loop)
###############################################################################

import math
import random
import copy
random.seed(7)

# ──────────────────────────── Helper Functions ──────────────────────────────
def softmax(vals, tau=1.0):
    m = max(vals)
    exps = [math.exp((v - m) / tau) for v in vals]
    Z = sum(exps)
    return [e / Z for e in exps]

def gelu(x):
    return 0.5 * x * (1 + math.tanh(0.79788456 * (x + 0.044715 * x**3)))

def dot(a, b):
    return sum(x * y for x, y in zip(a, b))

def add(a, b):
    return [x + y for x, y in zip(a, b)]

def layer_norm(v, eps=1e-5):
    mu = sum(v) / len(v)
    var = sum((x - mu) ** 2 for x in v) / len(v)
    inv = 1.0 / math.sqrt(var + eps)
    return [(x - mu) * inv for x in v]

def gumbel():
    u = random.random()
    return -math.log(-math.log(max(u, 1e-9)))

def l0(mat):
    return sum(abs(x) > 1e-8 for row in mat for x in row)

# ───────────────────────── Rotary Positional Encoding ───────────────────────
def _rotate(x, y, theta):
    return x * math.cos(theta) - y * math.sin(theta), x * math.sin(theta) + y * math.cos(theta)

def rotary(q_vec, k_vec, pos_idx, base=10000.0):
    q_out, k_out = q_vec[:], k_vec[:]
    L = len(q_vec)
    for i in range(0, L, 2):
        angle = (pos_idx / base) ** (i / L)
        q_out[i], k_out[i]         = _rotate(q_vec[i], k_vec[i],         angle)
        q_out[i+1], k_out[i+1]     = _rotate(q_vec[i+1], k_vec[i+1],     angle)
    return q_out, k_out

# ───────────────────── 1) Latent-Causal Graph Builder ───────────────────────
class EdgeMLP:
    def __init__(self, d):
        self.W = [[random.uniform(-0.1, 0.1) for _ in range(2 * d)] for _ in range(4)]
        self.b = [0.0] * 4
        self.v = [random.uniform(-0.2, 0.2) for _ in range(4)]

    def __call__(self, x, y):
        h = [gelu(dot(w, add(x, y)) + b) for w, b in zip(self.W, self.b)]
        return dot(self.v, h)

class LCG:
    def __init__(self, d, tau=1.2):
        self.edge = EdgeMLP(d)
        self.tau  = tau

    def _gate(self, logit):
        p = 1 / (1 + math.exp(-(logit + gumbel()) / self.tau))
        return 1.0 if p > 0.5 else 0.0

    def build(self, H):
        T = len(H)
        A = [[1 if j <= i else 0 for j in range(T)] for i in range(T)]
        for i in range(T):
            for j in range(i):
                A[i][j] = self._gate(self.edge(H[i], H[j]))
        return A  # hard DAG mask

# ───────────── 2) Causal-Masked Multi-Head Attention (CMA) ──────────────────
def project_flat(H, W):
    T, d, dk = len(H), len(H[0]), len(W[0])
    flat = []
    for t in range(T):
        h = H[t]
        for k in range(dk):
            acc = 0.0
            for i in range(d):
                if i < len(W) and k < len(W[i]):         # bounds-safe
                    acc += h[i] * W[i][k]
            flat.append(acc)
    return flat  # length = T * dk

class CMA:
    def __init__(self, d_model, heads=4):
        assert d_model % heads == 0
        self.h  = heads
        self.dk = d_model // heads

        def proj():
            return [[random.uniform(-0.3, 0.3) for _ in range(self.dk)]
                    for _ in range(d_model)]

        self.WQ = [proj() for _ in range(self.h)]
        self.WK = [proj() for _ in range(self.h)]
        self.WV = [proj() for _ in range(self.h)]
        self.WO = [[random.uniform(-0.2, 0.2) for _ in range(d_model)]
                   for _ in range(d_model)]

    def head(self, H, A, WQ, WK, WV):
        Q = project_flat(H, WQ)
        K = project_flat(H, WK)
        V = project_flat(H, WV)

        T   = len(H)
        out = [[0.0] * self.dk for _ in range(T)]
        scl = 1.0 / math.sqrt(self.dk)

        # rotary
        for t in range(T):
            q_seg = Q[t*self.dk:(t+1)*self.dk]
            k_seg = K[t*self.dk:(t+1)*self.dk]
            q_rot, k_rot = rotary(q_seg, k_seg, t)
            Q[t*self.dk:(t+1)*self.dk] = q_rot
            K[t*self.dk:(t+1)*self.dk] = k_rot

        # scaled dot-product attention
        for i in range(T):
            qi = Q[i*self.dk:(i+1)*self.dk]
            logits = [
                scl * dot(qi, K[j*self.dk:(j+1)*self.dk]) if A[i][j] else -1e9
                for j in range(T)
            ]
            w = softmax(logits)
            for j in range(T):
                vj = V[j*self.dk:(j+1)*self.dk]
                for u in range(self.dk):
                    out[i][u] += w[j] * vj[u]
        return out  # T × dk

    def __call__(self, H, A):
        heads = [
            self.head(H, A, self.WQ[h], self.WK[h], self.WV[h])
            for h in range(self.h)
        ]
        T = len(H)
        concat = [
            [val for h in range(self.h) for val in heads[h][t]]
            for t in range(T)
        ]
        return [
            [dot(w_row, concat[t]) for w_row in self.WO]
            for t in range(T)
        ]

# ──────── 3) Causal-Conditioned Sparse-Expert Feed-Forward (C-MoE) ──────────
class CMoE:
    def __init__(self, d_model, d_hidden=64, n_exp=4):
        def make_ffn():
            W1 = [[random.uniform(-0.2, 0.2) for _ in range(d_model)]
                  for _ in range(d_hidden)]
            b1  = [0.0] * d_hidden
            W2 = [[random.uniform(-0.2, 0.2) for _ in range(d_hidden)]
                  for _ in range(d_model)]
            b2  = [0.0] * d_model
            def f(x):
                h = [gelu(dot(w, x) + b) for w, b in zip(W1, b1)]
                return [dot(col, h) + bo for col, bo in zip(zip(*W2), b2)]
            return f
        self.experts = [
            {'w': [random.uniform(-0.4, 0.4) for _ in range(d_model)],
             'ffn': make_ffn()} for _ in range(n_exp)
        ]

    def __call__(self, Z, A):
        T, d = len(Z), len(Z[0])
        out  = [[0.0]*d for _ in range(T)]
        vic  = [set() for _ in range(T)]
        for t in range(T):
            # causal vicinity set of token t
            Vt = {j for j in range(T) if A[t][j] or A[j][t]}
            allowed = set().union(*(vic[j] for j in Vt)) or set(range(len(self.experts)))
            scores  = [
                dot(e['w'], Z[t]) if i in allowed else -1e9
                for i, e in enumerate(self.experts)
            ]
            k = max(range(len(scores)), key=scores.__getitem__)
            vic[t].add(k)
            out[t] = self.experts[k]['ffn'](Z[t])
        return out  # T × d_model

# ─── 4) Intervention & Counterfactual Loss (IKLC) ───────────────────────────
def IKLC(H, A, cls_W, n_int=3, tau_cf=0.5, lmb=0.05):
    T, d = len(H), len(H[0])
    def logits(mat):
        pooled = [sum(mat[t][j] for t in range(T)) / T for j in range(d)]
        return [dot(w, pooled) for w in cls_W]
    p0 = softmax(logits(H))
    kl_sum, delta_sum = 0.0, 0.0
    for _ in range(n_int):
        idx   = random.randrange(T)
        noise = [random.uniform(-1, 1) for _ in range(d)]
        Hcf   = [row[:] for row in H]
        Hcf[idx] = noise
        p1    = softmax(logits(Hcf), tau_cf)
        kl    = sum(
            p0[j] * math.log(max(p0[j] / max(p1[j], 1e-9), 1e-9))
            for j in range(len(p0))
        )
        e0 = sum(j * p0[j] for j in range(len(p0)))
        e1 = sum(j * p1[j] for j in range(len(p1)))
        kl_sum    += kl
        delta_sum += abs(e0 - e1)
    return kl_sum / n_int + lmb * (delta_sum / n_int)

# ─────────────── 5) Self-Evolution Gate (CAER loop) ─────────────────────────
def CAER_step(model, mutate_fn, eval_fn,
              temp, eps,
              alpha=0.5, beta=0.1,
              T0=1.0, decay_T=0.92, decay_eps=0.85):
    baseF = eval_fn(model, alpha, beta)
    cand  = mutate_fn(copy.deepcopy(model), eps)
    newF  = eval_fn(cand, alpha, beta)
    dF    = newF - baseF
    accept = (dF < 0) or (random.random() < math.exp(-dF / temp))
    if accept:
        return cand, True, T0, eps
    else:
        return model, False, temp * decay_T, eps * decay_eps

# ──────────────── CoCE Transformer (2 causal blocks) ────────────────────────
class CoCETransformer:
    def __init__(self, vocab=64, seq_len=6, d_model=32):
        self.T  = seq_len
        self.d  = d_model
        self.emb = [[random.uniform(-1, 1) for _ in range(d_model)] for _ in range(vocab)]
        self.block1 = (LCG(d_model), CMA(d_model, 4), CMoE(d_model, 64, 4))
        self.block2 = (LCG(d_model), CMA(d_model, 4), CMoE(d_model, 64, 4))
        self.cls_W  = [[random.uniform(-0.3, 0.3) for _ in range(d_model)] for _ in range(4)]

    def _block(self, H, triple):
        lcg, cma, cmoe = triple
        A = lcg.build(H)
        Z = cma(H, A)
        H1 = [layer_norm(add(H[t], Z[t])) for t in range(self.T)]
        F  = cmoe(H1, A)
        H2 = [layer_norm(add(H1[t], F[t])) for t in range(self.T)]
        return H2, A

    def forward(self, tokens):
        H = [self.emb[t] for t in tokens]
        H, _  = self._block(H, self.block1)
        H, A2 = self._block(H, self.block2)

        pooled = [sum(H[t][j] for t in range(self.T)) / self.T for j in range(self.d)]
        logits = [dot(w, pooled) for w in self.cls_W]
        reward = logits[0]

        Lcf  = IKLC(H, A2, self.cls_W)
        spars = l0(self.block1[1].WO) + l0(self.block2[1].WO)
        return reward, Lcf, spars

# ───────────────────── Fitness & Mutation utilities ─────────────────────────
def fitness(model, alpha, beta):
    tokens = [random.randrange(len(model.emb)) for _ in range(model.T)]
    R, Lcf, sp = model.forward(tokens)
    return -R + alpha * Lcf + beta * sp

def mutate(model, eps):
    target = random.choice([model.block1[1].WO, model.block2[1].WO, model.cls_W])
    i = random.randrange(len(target))
    j = random.randrange(len(target[0]))
    target[i][j] += eps * random.gauss(0, 1)
    return model

# ───────────────────────────── Demo Execution ───────────────────────────────
if __name__ == "__main__":
    model = CoCETransformer()
    Ttemp = 1.0
    eps   = 0.4

    for step in range(10):
        model, ok, Ttemp, eps = CAER_step(model, mutate, fitness, Ttemp, eps)
        R, Lcf, spars = model.forward([1, 2, 3, 4, 5, 6])
        print(f"step {step:02d} {'ACCEPT' if ok else 'reject':>6s} | "
              f"R={R:+.3f}  Lcf={Lcf:.3f}  spars={spars}  T={Ttemp:.2f}  eps={eps:.3f}")
