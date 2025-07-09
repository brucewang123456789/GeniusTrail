###############################################################################
# Self-Evolution Gate (SEG) — CAER toy demo (pure Python, no libraries) #
###############################################################################

import math, random

# ---------- dummy metrics ----------------------------------------------------
def task_reward(theta):
    """Higher reward if weights align with +1 vector."""
    return sum(theta)

def intervention_loss(theta):
    """Lower if weights are small (pretend causal consistency)."""
    return sum(abs(w) for w in theta)

def sparsity(theta, thresh=0.1):
    """Count of weights above threshold."""
    return sum(1 for w in theta if abs(w) > thresh)

# ---------- CAER parameters --------------------------------------------------
alpha   = 0.5   # weight on L_cf
beta    = 0.1   # weight on sparsity
T0      = 1.0
gamma_T = 0.95
eps0    = 0.5
gamma_e = 0.9
dim     = 8

def fitness(theta):
    return -task_reward(theta) + alpha*intervention_loss(theta) + beta*sparsity(theta)

# ---------- evolution loop ---------------------------------------------------
random.seed(0)
theta_best = [random.uniform(-1,1) for _ in range(dim)]
T, eps = T0, eps0

for step in range(50):
    # mutate
    theta_prime = [w + eps*random.gauss(0,1) for w in theta_best]
    # compute fitness gap
    dF = fitness(theta_prime) - fitness(theta_best)
    accept = dF < 0 or random.random() < math.exp(-dF / T)
    if accept:
        theta_best = theta_prime
        T, eps = T0, eps0  # reset
        status = "accepted"
    else:
        T  *= gamma_T
        eps *= gamma_e
        status = "rejected"

    if step % 10 == 0 or status == "accepted":
        print(f"step {step:2d}: {status:8s} F={fitness(theta_best):.3f}")

print("Final theta:", [round(w,3) for w in theta_best])
