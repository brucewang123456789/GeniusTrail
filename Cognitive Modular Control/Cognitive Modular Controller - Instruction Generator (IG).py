# =====================================
# Instruction Generator (IG)
# =====================================

class InstructionGenerator:
    def __init__(self, max_resources):
        """
        Initialize the Instruction Generator with resource constraints.

        Args:
            max_resources (int): Maximum resources available for tasks.
        """
        self.max_resources = max_resources

    # ----------------------------------------
    # Component 1: Instruction Prioritization
    # ----------------------------------------
    def prioritize_features(self, features, weights):
        """
        Prioritize features based on relevance and weights.

        Args:
            features (list): List of semantic features.
            weights (list): List of weights for each feature.

        Returns:
            list: Prioritized scores.
        """
        return [w * f for w, f in zip(weights, features)]

    # ----------------------------------------
    # Component 2: Constraint Analysis
    # ----------------------------------------
    def analyze_constraints(self, task_requirements):
        """
        Analyze task feasibility based on constraints.

        Args:
            task_requirements (list): List of resources required for tasks.

        Returns:
            list: Feasibility scores for each task.
        """
        feasibility_scores = [
            self.max_resources / requirement if requirement > 0 else 0
            for requirement in task_requirements
        ]
        return feasibility_scores

    # ----------------------------------------
    # Component 3: Multi-Instruction Output
    # ----------------------------------------
    def generate_instructions(self, prioritized_scores, feasibility_scores):
        """
        Generate instructions based on priorities and feasibility.

        Args:
            prioritized_scores (list): List of prioritized feature scores.
            feasibility_scores (list): List of feasibility scores.

        Returns:
            list: Final instructions.
        """
        return [
            p * f for p, f in zip(prioritized_scores, feasibility_scores)
        ]

    # ============================================
    # Execution Pipeline
    # ============================================
    def process_features(self, features, weights, task_requirements):
        """
        Full instruction generation pipeline.

        Args:
            features (list): List of semantic features.
            weights (list): List of weights for each feature.
            task_requirements (list): List of task resource requirements.

        Returns:
            list: Final instructions.
        """
        prioritized_scores = self.prioritize_features(features, weights)
        feasibility_scores = self.analyze_constraints(task_requirements)
        instructions = self.generate_instructions(prioritized_scores, feasibility_scores)
        return instructions


# =====================================
# Example Usage
# =====================================

if __name__ == "__main__":
    # Initialize the Instruction Generator
    ig = InstructionGenerator(max_resources=100)

    # Input: Semantic features, weights, and task requirements
    semantic_features = [0.8, 0.6, 0.9]
    weights = [1.2, 1.0, 0.8]
    task_requirements = [50, 30, 20]

    # Generate instructions
    final_instructions = ig.process_features(semantic_features, weights, task_requirements)

    # Output: Final instructions
    print("Final Instructions:", final_instructions)
