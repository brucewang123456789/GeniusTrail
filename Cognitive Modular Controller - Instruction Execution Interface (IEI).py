class ActionOutputGateway:
    def __init__(self, weight_matrix, min_limits, max_limits):
        """
        Initializes the ActionOutputGateway.

        Args:
            weight_matrix (list of list of float): Maps validated decisions to robot-specific commands.
            min_limits (list of float): Minimum allowable values for each command.
            max_limits (list of float): Maximum allowable values for each command.
        """
        self.weight_matrix = weight_matrix
        self.min_limits = min_limits
        self.max_limits = max_limits

    def map_decisions_to_commands(self, validated_decisions):
        """
        Maps validated decisions to initial action commands.

        Args:
            validated_decisions (list of float): Final decisions from the EVL module.

        Returns:
            list of float: Initial action commands.
        """
        commands = [sum(w * d for w, d in zip(row, validated_decisions)) for row in self.weight_matrix]
        return commands

    def refine_commands_with_feedback(self, initial_commands, feedback):
        """
        Refines commands based on feedback.

        Args:
            initial_commands (list of float): Initial commands mapped from decisions.
            feedback (list of float): Environmental feedback for refinement.

        Returns:
            list of float: Refined commands.
        """
        refined_commands = [cmd + fb for cmd, fb in zip(initial_commands, feedback)]
        return refined_commands

    def clip_commands(self, commands):
        """
        Clips commands to ensure feasibility.

        Args:
            commands (list of float): Commands to be clipped.

        Returns:
            list of float: Feasible commands.
        """
        return [
            max(min(cmd, max_lim), min_lim)
            for cmd, min_lim, max_lim in zip(commands, self.min_limits, self.max_limits)
        ]

    def generate_action_commands(self, validated_decisions, feedback):
        """
        Generates actionable commands for the modular robots.

        Args:
            validated_decisions (list of float): Final decisions from the EVL module.
            feedback (list of float): Environmental feedback.

        Returns:
            list of float: Final actionable commands.
        """
        initial_commands = self.map_decisions_to_commands(validated_decisions)
        refined_commands = self.refine_commands_with_feedback(initial_commands, feedback)
        final_commands = self.clip_commands(refined_commands)
        return final_commands


# Example usage:
if __name__ == "__main__":
    # Example configuration
    weight_matrix = [
        [0.8, 0.1, 0.1],
        [0.2, 0.7, 0.1],
        [0.3, 0.3, 0.4],
    ]
    min_limits = [0.0, 0.0, 0.0]
    max_limits = [1.0, 1.0, 1.0]

    gateway = ActionOutputGateway(weight_matrix, min_limits, max_limits)

    validated_decisions = [0.9, 0.6, 0.8]  # From EVL module
    environmental_feedback = [0.05, -0.1, 0.02]  # Example feedback

    final_commands = gateway.generate_action_commands(validated_decisions, environmental_feedback)
    print("Final Actionable Commands:", final_commands)
