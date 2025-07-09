# ---------------------------------------------------------------------------------
# Knowledge Graph–Guided GAN
# Python 3.13, no external libraries.
# This code is strictly for illustrative logic, not real production.

import random

# ---------------------------
# 1) TOY KNOWLEDGE GRAPH
# ---------------------------
# We'll store adjacency as a dict of node: [neighbors].
# Each node will also have a "feature_value" to simulate initial attributes.

graph_nodes = {
    "ergonomic": 0.8,  # Some toy feature value
    "optics": 0.5,
    "battery": 0.6,
    "style": 0.9,
}

graph_edges = {
    "ergonomic": ["optics", "battery"],
    "optics": ["ergonomic", "style"],
    "battery": ["ergonomic"],
    "style": ["optics"]
}

# ---------------------------
# 2) SIMPLE GNN EMBEDDING
# ---------------------------
def gnn_embedding(nodes, edges, steps=1):
    # We'll do a naive neighbor-sum approach for demonstration.
    # In each step, we sum the neighbor values + own, then average.
    node_values = dict(nodes)  # copy initial feature values
    for _ in range(steps):
        new_values = {}
        for n in node_values:
            s = node_values[n]
            neighs = edges[n]
            for nx in neighs:
                s += node_values[nx]
            # Average-based update
            new_values[n] = s / (len(neighs) + 1)
        node_values = new_values
    # Combine into a single vector
    # For simplicity, just store as a list [ergonomic_val, optics_val, battery_val, style_val]
    return [node_values["ergonomic"], node_values["optics"],
            node_values["battery"],    node_values["style"]]

# ---------------------------
# 3) CONDITION VECTOR MERGE
# ---------------------------
def build_condition_vector(kg_embed, user_pref):
    # Merges knowledge graph embedding with user preference signals
    # e.g. user_pref can be a list of user-specific parameters
    # We'll just do a naive concatenation
    return kg_embed + user_pref  # list concat

# ---------------------------
# 4) GENERATOR
# ---------------------------
class Generator:
    def __init__(self):
        self.param = 0.5  # toy parameter

    def forward(self, noise, condition):
        # noise: a random float list
        # condition: knowledge + user
        # We'll generate a "design" as a 2D list with random offsets
        # For demonstration only
        design = []
        for i in range(len(condition)):
            val = (condition[i] + noise[i]) * self.param
            design.append(val)
        return design

    def update(self, grad):
        # naive gradient step
        self.param -= 0.01 * grad

# ---------------------------
# 5) DISCRIMINATOR
# ---------------------------
class Discriminator:
    def __init__(self):
        self.weight = 0.3  # toy param

    def forward(self, design, condition):
        # calculates "real/fake" measure
        # We'll do a simple dot-like product
        score = 0.0
        for i in range(len(design)):
            score += design[i] * condition[i] * self.weight
        return score

    def compute_loss(self, real_score, fake_score):
        # a naive "loss"
        # real should be high, fake should be lower
        # We'll do a difference measure
        return abs(1 - real_score) + abs(0 - fake_score)

    def update(self, grad):
        self.weight -= 0.01 * grad

# ---------------------------
# 6) CONSTRAINT CHECK
# ---------------------------
def check_constraints(design):
    # For demonstration, let's say any negative item is a violation
    # or if sum of design is too large. This is purely illustrative.
    if any(x < 0 for x in design):
        return False
    if sum(design) > 8.0:
        return False
    return True

# ---------------------------
# 7) MAIN EXECUTION
# ---------------------------
def main():
    random.seed(42)
    # Build knowledge graph embedding
    kg_embed = gnn_embedding(graph_nodes, graph_edges, steps=2)
    # Suppose user preferences (toy example)
    user_pref = [1.0, 0.2]

    # Construct Condition Vector
    condition_vec = build_condition_vector(kg_embed, user_pref)

    # Initialize G and D
    G = Generator()
    D = Discriminator()

    # Suppose we have "real" design to train D
    # In reality, you'd have multiple real samples from historical designs
    real_design = [2.0, 2.2, 1.5, 1.9, 1.0, 0.3]  # length matches condition length

    # Training loop (toy example)
    for epoch in range(3):
        # 1) Generator forward
        noise = [random.random() for _ in range(len(condition_vec))]
        gen_design = G.forward(noise, condition_vec)

        # 2) Discriminator forward
        real_score = D.forward(real_design, condition_vec)
        fake_score = D.forward(gen_design, condition_vec)

        # 3) Compute loss
        d_loss = D.compute_loss(real_score, fake_score)

        # 4) Backprop signals
        # We'll do a naive approach:
        d_grad = 0.1 if fake_score > real_score else -0.1
        D.update(d_grad)
        # For generator, if fake_score is too low, we push param up, else down
        g_grad = (1.0 - fake_score)
        G.update(g_grad)

        # 5) Print status
        print("Epoch:", epoch, 
              " RealScore:", round(real_score,2), 
              " FakeScore:", round(fake_score,2), 
              " GenDesign:", [round(x,2) for x in gen_design])

    # Check constraints, if fail => revise
    if not check_constraints(gen_design):
        print("Generated design violates constraints; loop back or adjust KG.")
    else:
        print("Generated design passes constraints, proceed to evaluation of 'hyper-personalization'.")

    # End of demonstration

if __name__ == "__main__":
    main()
