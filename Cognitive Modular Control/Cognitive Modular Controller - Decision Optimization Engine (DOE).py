# =====================================
# Decision Optimization Engine (DOE)
# =====================================

class DecisionOptimizationEngine:
    def __init__(self, scenarios, weights):
        """
        Initialize the Decision Optimization Engine.

        Args:
            scenarios (list): Pre-trained decision scenarios.
            weights (list): Weights for scoring actions.
        """
        self.scenarios = scenarios
        self.weights = weights

    # ----------------------------------------
    # Component 1: Decision Evaluation
    # ----------------------------------------
    def evaluate_decision(self, feedback):
        """
        Match feedback to the most similar scenario.

        Args:
            feedback (list): Refined feedback data.

        Returns:
            int: Index of the best-matching scenario.
        """
        similarities = [self.calculate_similarity(feedback, scenario) for scenario in self.scenarios]
        return similarities.index(max(similarities))

    def calculate_similarity(self, feedback, scenario):
        """
        Calculate similarity between feedback and scenario.

        Args:
            feedback (list): Feedback data.
            scenario (list): Scenario data.

        Returns:
            float: Similarity score.
        """
        return sum(f * s for f, s in zip(feedback, scenario))  # Simplified similarity calculation

    # ----------------------------------------
    # Component 2: Strategy Optimization
    # ----------------------------------------
    def optimize_strategy(self, objectives):
        """
        Optimize strategies based on multiple objectives.

        Args:
            objectives (list of lists): List of objective functions.

        Returns:
            list: Optimized decision variables.
        """
        return [min(obj) for obj in zip(*objectives)]  # Simplified multi-objective optimization

    # ----------------------------------------
    # Component 3: Action Selection
    # ----------------------------------------
    def select_action(self, optimized_decisions):
        """
        Rank and select the best action.

        Args:
            optimized_decisions (list of lists): Optimized decision variables.

        Returns:
            int: Index of the selected action.
        """
        scores = [sum(w * d for w, d in zip(self.weights, decision)) for decision in optimized_decisions]
        return scores.index(max(scores))

    # ============================================
    # Execution Pipeline
    # ============================================
    def make_decision(self, feedback, objectives):
        """
        Full decision optimization pipeline.

        Args:
            feedback (list): Refined feedback data.
            objectives (list of lists): List of objective functions.

        Returns:
            int: Index of the final selected action.
        """
        best_scenario = self.evaluate_decision(feedback)
        optimized_decisions = self.optimize_strategy(objectives)
        final_action = self.select_action([optimized_decisions])  # Fix: Wrap as list of decisions
        return final_action


# =====================================
# Example Usage
# =====================================

if __name__ == "__main__":
    # Initialize the Decision Optimization Engine
    doe = DecisionOptimizationEngine(scenarios=[[1, 2, 3], [4, 5, 6]], weights=[0.5, 0.3, 0.2])

    # Input: Refined feedback and objectives
    refined_feedback = [1.2, 0.8, 1.5]
    objectives = [[0.5, 1.0, 0.8], [0.7, 0.9, 1.1]]

    # Make a decision
    selected_action = doe.make_decision(refined_feedback, objectives)

    # Output: Final selected action
    print("Final Selected Action Index:", selected_action)
