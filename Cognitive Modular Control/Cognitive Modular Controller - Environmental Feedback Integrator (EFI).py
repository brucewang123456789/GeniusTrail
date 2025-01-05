# =====================================
# Environmental Feedback Integrator (EFI)
# =====================================

class EnvironmentalFeedbackIntegrator:
    def __init__(self, sensor_weights, noise_threshold):
        """
        Initialize the Environmental Feedback Integrator.

        Args:
            sensor_weights (list): List of weights for each sensor.
            noise_threshold (float): Threshold for noise filtering.
        """
        self.sensor_weights = sensor_weights
        self.noise_threshold = noise_threshold

    # ----------------------------------------
    # Component 1: Feedback Aggregation
    # ----------------------------------------
    def aggregate_feedback(self, sensor_data):
        """
        Aggregate feedback from multiple sensors.

        Args:
            sensor_data (list): List of sensor readings.

        Returns:
            float: Aggregated feedback value.
        """
        aggregated_feedback = sum(w * s for w, s in zip(self.sensor_weights, sensor_data))
        return aggregated_feedback

    # ----------------------------------------
    # Component 2: Environmental Analysis
    # ----------------------------------------
    def detect_environmental_events(self, current_feedback, previous_feedback):
        """
        Detect significant environmental changes.

        Args:
            current_feedback (float): Current aggregated feedback.
            previous_feedback (float): Previous aggregated feedback.

        Returns:
            bool: True if significant change detected, otherwise False.
        """
        return abs(current_feedback - previous_feedback) > self.noise_threshold

    # ----------------------------------------
    # Component 3: Feedback Refinement
    # ----------------------------------------
    def refine_feedback(self, aggregated_feedback, noise_estimate):
        """
        Refine feedback by filtering noise.

        Args:
            aggregated_feedback (float): Aggregated feedback value.
            noise_estimate (float): Estimated noise level.

        Returns:
            float: Refined feedback value.
        """
        refined_feedback = aggregated_feedback - noise_estimate
        return max(refined_feedback, 0)  # Ensure non-negative values

    def score_feedback(self, feedback, impact, noise):
        """
        Score feedback based on relevance and noise level.

        Args:
            feedback (float): Feedback signal.
            impact (float): Measured impact of feedback.
            noise (float): Measured noise level.

        Returns:
            float: Feedback relevance score.
        """
        return impact / (noise + 1)

    # ============================================
    # Execution Pipeline
    # ============================================
    def integrate_feedback(self, sensor_data, previous_feedback, noise_estimate, impact_scores):
        """
        Full feedback integration pipeline.

        Args:
            sensor_data (list): List of sensor readings.
            previous_feedback (float): Previous aggregated feedback.
            noise_estimate (float): Estimated noise level.
            impact_scores (list): List of impact scores for each sensor.

        Returns:
            float: Final refined feedback score.
        """
        aggregated_feedback = self.aggregate_feedback(sensor_data)
        event_detected = self.detect_environmental_events(aggregated_feedback, previous_feedback)
        if event_detected:
            refined_feedback = self.refine_feedback(aggregated_feedback, noise_estimate)
            final_score = sum(self.score_feedback(refined_feedback, impact, noise_estimate) for impact in impact_scores)
            return final_score
        return 0  # No significant feedback to process


# =====================================
# Example Usage
# =====================================

if __name__ == "__main__":
    # Initialize the Environmental Feedback Integrator
    efi = EnvironmentalFeedbackIntegrator(sensor_weights=[0.5, 0.3, 0.2], noise_threshold=5.0)

    # Input: Sensor data and previous feedback
    sensor_data = [20, 30, 10]
    previous_feedback = 50
    noise_estimate = 3
    impact_scores = [1.0, 0.8, 0.6]

    # Integrate feedback
    final_feedback_score = efi.integrate_feedback(sensor_data, previous_feedback, noise_estimate, impact_scores)

    # Output: Final feedback score
    print("Final Feedback Score:", final_feedback_score)
