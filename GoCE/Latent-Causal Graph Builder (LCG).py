###############################################################################
#  Graph-of-Causal Evolution : Latent-Causal Graph Builder (full detail)      #
#  No external packages; Python ≥3.8                                          #
###############################################################################

import random, math, itertools
random.seed(42)

# ────────────────────────────────────────────────────────────────────────────
# Utility functions
# ────────────────────────────────────────────────────────────────────────────
def sigmoid(x: float) -> float:
    """Numerically stable logistic sigmoid."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)

def gelu(x: float) -> float:
    """GELU approximation (tanh form)."""
    return 0.5 * x * (1.0 + math.tanh(0.7978845608 * (x + 0.044715 * x * x * x)))

def gumbel_noise() -> float:
    """Draw Gumbel(0,1) variate."""
    u = random.random()
    return -math.log(-math.log(max(u, 1e-9)))

# ────────────────────────────────────────────────────────────────────────────
# Tiny tensor helpers (lists of floats)
# ────────────────────────────────────────────────────────────────────────────
def vec_add(a, b):           return [x + y for x, y in zip(a, b)]
def vec_dot(a, b):           return sum(x * y for x, y in zip(a, b))
def mat_mul(A, B):           # naive square-mat mul
    n = len(A)
    return [[sum(A[i][k] * B[k][j] for k in range(n)) for j in range(n)]
            for i in range(n)]

def mat_trace(M):            return sum(M[i][i] for i in range(len(M)))

# ────────────────────────────────────────────────────────────────────────────
# Two-layer perceptron (weights as plain lists)
# ────────────────────────────────────────────────────────────────────────────
class TinyMLP:
    def __init__(self, dim_in, dim_hidden):
        self.W1 = [[random.uniform(-0.1, 0.1) for _ in range(dim_in)]
                   for _ in range(dim_hidden)]
        self.b1 = [0.0] * dim_hidden
        self.W2 = [[random.uniform(-0.1, 0.1) for _ in range(dim_hidden)]]
        self.b2 = [0.0]

    def forward(self, vec):
        h = [gelu(vec_dot(w, vec) + b) for w, b in zip(self.W1, self.b1)]
        return vec_dot(self.W2[0], h) + self.b2[0]

# ────────────────────────────────────────────────────────────────────────────
# Latent-Causal Graph Builder
# ────────────────────────────────────────────────────────────────────────────
class LatentCausalGraph:
    """
    • Learns a sparse upper-triangular adjacency
    • Returns both HARD (0/1) and SOFT (0…1) versions
    • No external libraries
    """
    def __init__(self, dim, tau=1.5):
        self.dim = dim
        self.tau = tau
        self.edge_mlp = TinyMLP(dim * 2, dim * 4)

    def hard_concrete(self, logit):
        """Differentiable 0/1 gate (straight-through, temperature self.tau)."""
        noise = gumbel_noise()
        y_soft = sigmoid((logit + noise) / self.tau)
        y_hard = 1.0 if y_soft > 0.5 else 0.0   # straight-through trick
        return y_hard, y_soft

    def build(self, h_list):
        T = len(h_list)
        hard = [[0.0] * T for _ in range(T)]
        soft = [[0.0] * T for _ in range(T)]
        for i, j in itertools.product(range(T), repeat=2):
            if j < i:  # acyclic constraint
                logit = self.edge_mlp.forward(h_list[i] + h_list[j])
                h, s = self.hard_concrete(logit)
                hard[i][j], soft[i][j] = h, s
        return hard, soft

    # ---- penalty terms -----------------------------------------------------
    def cycle_penalty(self, soft_A):
        """trace(exp(A)) using 4-term Taylor (sufficient for tiny soft values)."""
        n = len(soft_A)
        I = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
        A  = soft_A
        A2 = mat_mul(A, A)
        A3 = mat_mul(A2, A)
        series = [[I[i][j] + A[i][j] + 0.5*A2[i][j] + (1/6)*A3[i][j]
                   for j in range(n)] for i in range(n)]
        return mat_trace(series)

    def sparsity_penalty(self, soft_A):
        return sum(sum(row) for row in soft_A)

# ────────────────────────────────────────────────────────────────────────────
# Demo run
# ────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    T, D = 5, 4
    h_vectors = [[random.uniform(-1, 1) for _ in range(D)] for _ in range(T)]
    lcg = LatentCausalGraph(D, tau=1.3)
    A_hard, A_soft = lcg.build(h_vectors)

    print("Hard adjacency:")
    for row in A_hard: print(row)

    print("\nsoft sparsity  :", lcg.sparsity_penalty(A_soft))
    print("cycle penalty  :", lcg.cycle_penalty(A_soft))
