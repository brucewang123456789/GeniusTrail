# =====================================
# Execution Validation and Learning (EVL)
# =====================================

class ExecutionValidationAndLearning:
    def __init__(self, learning_rate):
        """
        Initialize the Execution Validation and Learning module.

        Args:
            learning_rate (float): Rate at which policies are updated.
        """
        self.learning_rate = learning_rate

    # ----------------------------------------
    # Component 1: Execution Validation
    # ----------------------------------------
    def validate_execution(self, expected_outcomes, actual_outcomes):
        """
        Validate execution by comparing expected and actual outcomes.

        Args:
            expected_outcomes (list): Expected outcomes of actions.
            actual_outcomes (list): Actual outcomes of actions.

        Returns:
            list: List of errors for each action.
        """
        errors = [expected - actual for expected, actual in zip(expected_outcomes, actual_outcomes)]
        return errors

    # ----------------------------------------
    # Component 2: Performance Scoring
    # ----------------------------------------
    def score_performance(self, task_completed, task_target):
        """
        Score performance based on task completion.

        Args:
            task_completed (float): Amount of task completed.
            task_target (float): Target amount for the task.

        Returns:
            float: Performance score as a percentage.
        """
        return (task_completed / task_target) * 100

    # ----------------------------------------
    # Component 3: Learning Update
    # ----------------------------------------
    def update_policy(self, policy, errors):
        """
        Update the policy based on observed errors.

        Args:
            policy (dict): Original decision policy.
            errors (list): List of errors for each action.

        Returns:
            dict: Updated policy.
        """
        updated_policy = {
            action: value - self.learning_rate * error
            for (action, value), error in zip(policy.items(), errors)
        }
        return updated_policy

    # ============================================
    # Execution Pipeline
    # ============================================
    def execute_learning_pipeline(self, expected_outcomes, actual_outcomes, task_completed, task_target, policy):
        """
        Full validation and learning pipeline.

        Args:
            expected_outcomes (list): Expected outcomes of actions.
            actual_outcomes (list): Actual outcomes of actions.
            task_completed (float): Amount of task completed.
            task_target (float): Target amount for the task.
            policy (dict): Original decision policy.

        Returns:
            dict: Updated decision policy.
        """
        errors = self.validate_execution(expected_outcomes, actual_outcomes)
        performance_score = self.score_performance(task_completed, task_target)
        print(f"Performance Score: {performance_score}%")
        updated_policy = self.update_policy(policy, errors)
        return updated_policy


# =====================================
# Example Usage
# =====================================

if __name__ == "__main__":
    # Initialize the Execution Validation and Learning module
    evl = ExecutionValidationAndLearning(learning_rate=0.1)

    # Input: Expected and actual outcomes, task progress, and policy
    expected_outcomes = [1.0, 0.8, 1.2]
    actual_outcomes = [0.9, 0.7, 1.3]
    task_completed = 8
    task_target = 10
    policy = {"action_1": 0.5, "action_2": 0.7, "action_3": 0.6}

    # Execute the pipeline
    updated_policy = evl.execute_learning_pipeline(
        expected_outcomes, actual_outcomes, task_completed, task_target, policy
    )

    # Output: Updated policy
    print("Updated Policy:", updated_policy)
