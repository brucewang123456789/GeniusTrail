"""
TDR Framework - Module 2: Dynamic Causal Reasoner
----------------------------------------------------
Order of components (from bottom to top):
  1) CausalInferenceEngine
  2) HierarchicalDecisionPlanner
  3) PredictiveErrorMonitor
     - Subcomponent: FeedbackLoopIntegrator

This module demonstrates how the final annotated data from Module 1
is transformed into a state representation, used to decide high-level
actions, and then refined through predictive error monitoring. The real-time
updates here address key shortcomings of Adaptive Domain Randomization (ADR),
namely the inability to handle causal dependencies or perform online error-driven
adjustments.

Code Requirements:
  - Written for Python 3.13 without any external libraries.
  - Must seamlessly connect with the output of Module 1 (ContextualAnnotator).
  - Must reflect advanced engineering practice and reference academic logic,
    rather than random domain perturbations as in ADR.
"""

import random


# ====================================================
# (1) CausalInferenceEngine (BOTTOM of the second module)
# ====================================================
class CausalInferenceEngine:
    """
    Responsible for constructing or updating the internal state representation
    based on annotated data from Module 1. This state is intended to capture
    cause-and-effect relationships (e.g., environment tag influences certain
    physical dynamics).

    Attributes:
        causal_params: A dictionary or numeric variable representing an internal
                       model of how states transition. Demonstrates how the engine
                       goes beyond random domain modifications in ADR.
    """

    def __init__(self):
        # An example parameter capturing how strongly environment tags influence the state
        self.causal_params = {
            "env_influence": 0.5  # dummy default
        }

    def infer_state(self, annotated_data):
        """
        annotated_data might include:
          - 'pyramid_vector': the fused multi-scale features
          - 'importance_measure': numeric summary
          - 'environment_tag': e.g. 'indoor', 'outdoor', 'low_light', etc.

        Returns a dictionary representing the newly inferred 'state'.
        """
        pyramid_vector = annotated_data.get("pyramid_vector", [])
        importance = annotated_data.get("importance_measure", 0.0)
        env_tag = annotated_data.get("environment_tag", "unknown")

        # A trivial approach: incorporate environment influence
        # to highlight how the system is factoring in domain context.
        # In real systems, you might have learned or formula-based expansions.
        if env_tag == "indoor":
            env_factor = self.causal_params["env_influence"]
        elif env_tag == "outdoor":
            env_factor = self.causal_params["env_influence"] * 2.0
        elif env_tag == "low_light":
            env_factor = self.causal_params["env_influence"] * 1.5
        else:
            # normal, foggy, or unknown
            env_factor = self.causal_params["env_influence"] * 0.8

        # Create a combined state representation
        # For demonstration: average over pyramid_vector plus a scaled environment factor
        vector_sum = 0.0
        for val in pyramid_vector:
            vector_sum += val
        vector_mean = 0.0
        if len(pyramid_vector) > 0:
            vector_mean = vector_sum / float(len(pyramid_vector))

        # The 'state_value' might represent an internal numeric state
        state_value = vector_mean + (importance * 0.1) + env_factor
        new_state = {
            "state_value": state_value,
            "env_tag": env_tag,
            "vector_mean": vector_mean
        }

        # Return the new or updated state structure
        return new_state


# ==============================================================
# (2) HierarchicalDecisionPlanner (MIDDLE of the second module)
# ==============================================================
class HierarchicalDecisionPlanner:
    """
    Leverages the state from the CausalInferenceEngine to pick a high-level action
    or sub-goal. In advanced RL or planning systems, this might be a multi-layer
    neural net or hierarchical policy. Here, we demonstrate the structure and logic.

    By focusing on the meaningful relationships in 'state', we surpass ADR’s random
    parameter sweeps, which cannot deeply capture environment-driven nuances.
    """

    def __init__(self):
        # Example parameter that influences how 'state_value' translates into an action
        self.planner_threshold = 1.0

    def plan(self, inferred_state):
        """
        Depending on 'state_value', choose a discrete action ID or a structured plan.
        """
        state_val = inferred_state.get("state_value", 0.0)
        env_tag = inferred_state.get("env_tag", "unknown")

        # Example simplistic logic:
        # If state_value is high => we consider the environment easy and plan a "forward" action
        # Otherwise => we consider it more uncertain and plan a "cautious" action
        if state_val >= self.planner_threshold:
            action_id = "ACTION_FORWARD"
        else:
            action_id = "ACTION_CAUTIOUS_TURN"

        # We might also embed environment tag considerations
        # e.g., if env_tag is 'low_light', add 'light_compensation_routine'
        sub_goal = None
        if env_tag == "low_light":
            sub_goal = "ENABLE_NIGHT_VISION"
        elif env_tag == "foggy":
            sub_goal = "SLOW_SCAN_MODE"

        # Return a plan or high-level action dictionary
        return {
            "action_id": action_id,
            "sub_goal": sub_goal
        }


# ===========================================================
# (3) PredictiveErrorMonitor (TOP of the second module)
#       Subcomponent: FeedbackLoopIntegrator
# ===========================================================
class FeedbackLoopIntegrator:
    """
    Nested subcomponent that updates the causal model parameters in real time
    based on observed discrepancies. This is how we integrate the error-driven
    adaptation principle to surpass the purely random domain shifts in ADR.
    """

    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update_causal_params(self, causal_params, error_value):
        """
        Example gradient-based parameter tweak:
          new_param = old_param - (lr * error_value * partial_derivative)

        For illustration, we assume a direct link between 'env_influence' and error_value.
        """
        if "env_influence" in causal_params:
            # A naive gradient approach: decreasing the parameter if error is large
            old_val = causal_params["env_influence"]
            gradient_estimate = error_value  # placeholder
            new_val = old_val - (self.learning_rate * gradient_estimate)
            # clamp to avoid negative or overly large
            if new_val < 0.0:
                new_val = 0.0
            elif new_val > 5.0:
                new_val = 5.0
            causal_params["env_influence"] = new_val

class PredictiveErrorMonitor:
    """
    Monitors the difference between predicted vs. actual outcomes,
    then delegates to the FeedbackLoopIntegrator to keep refining
    causal parameters, ensuring the system continuously adapts
    and outperforms static randomizations.
    """

    def __init__(self, integrator_learning_rate=0.01):
        self.integrator = FeedbackLoopIntegrator(learning_rate=integrator_learning_rate)
        # We track a simple 'predicted_outcome' for demonstration
        self.predicted_outcome = 0.0

    def generate_prediction(self, state_value):
        """
        Simulates a predicted outcome given the current state.
        In a real system, you'd have a learned model f(state) -> predicted outcome.
        We'll do a trivial transform for demonstration.
        """
        # E.g., predicted_outcome = state_value * some constant
        self.predicted_outcome = state_value * 1.2
        return self.predicted_outcome

    def observe_and_update(self, causal_inference_engine, inferred_state):
        """
        1) Generate a predicted outcome from 'inferred_state'.
        2) Compare with an 'observed_outcome' (simulated).
        3) Compute error, update the CausalInferenceEngine's parameters through the integrator.
        4) Return the updated state or relevant data.
        """
        current_state_value = inferred_state.get("state_value", 0.0)
        predicted = self.generate_prediction(current_state_value)

        # We'll simulate an actual observation with some random factor
        observed_outcome = predicted + (random.random() - 0.5) * 0.5  # offset between -0.25 and +0.25

        # Compute the error
        error = observed_outcome - predicted
        if error < 0.0:
            error = -error  # absolute value

        # Update the causal parameters
        self.integrator.update_causal_params(causal_inference_engine.causal_params, error)

        # Possibly produce a refined or re-inferred state
        # (In more advanced systems, we'd re-run the CausalInferenceEngine with updated params)
        updated_state = causal_inference_engine.infer_state(inferred_state)
        return {
            "predicted_outcome": predicted,
            "observed_outcome": observed_outcome,
            "error": error,
            "updated_state": updated_state
        }


# =====================================================================
# DEMONSTRATION / TEST ROUTINE FOR MODULE 2 (DYNAMIC CAUSAL REASONER)
# =====================================================================
def demo_dynamic_causal_reasoner_module_2(annotated_output_from_module_1):
    """
    Demonstrates how data flows from Module 1's final annotated output
    through the second module's bottom-to-top components:
      - CausalInferenceEngine
      - HierarchicalDecisionPlanner
      - PredictiveErrorMonitor (with FeedbackLoopIntegrator subcomponent)

    :param annotated_output_from_module_1: the dictionary from the first module's
                                           ContextualAnnotator. Example structure:
                                           {
                                             "pyramid_vector": [...],
                                             "importance_measure": 1.234,
                                             "environment_tag": "indoor"
                                           }
    """

    print("====== Dynamic Causal Reasoner (Module 2) ======")

    # (1) BOTTOM: CausalInferenceEngine
    causal_engine = CausalInferenceEngine()
    inferred_state = causal_engine.infer_state(annotated_output_from_module_1)
    print("[CausalInferenceEngine] Inferred State:", inferred_state)

    # (2) MIDDLE: HierarchicalDecisionPlanner
    planner = HierarchicalDecisionPlanner()
    high_level_action = planner.plan(inferred_state)
    print("[HierarchicalDecisionPlanner] High-Level Action:", high_level_action)

    # (3) TOP: PredictiveErrorMonitor (with FeedbackLoopIntegrator inside)
    error_monitor = PredictiveErrorMonitor(integrator_learning_rate=0.02)
    monitor_result = error_monitor.observe_and_update(causal_engine, inferred_state)

    print("[PredictiveErrorMonitor] Prediction vs. Observation => Error:",
          monitor_result["error"])
    print("[PredictiveErrorMonitor] Updated State after Feedback Loop:",
          monitor_result["updated_state"])

    print("===== End of Module 2 Execution =====\n")
    return {
        "inferred_state": inferred_state,
        "high_level_action": high_level_action,
        "monitor_result": monitor_result
    }


# ==================================================
# OPTIONAL MAIN DEMO (SHOWING CHAIN FROM MODULE 1)
# ==================================================
def main_demo_chain_module_1_and_2():
    """
    This optional function shows how to pass the final output from Module 1
    into Module 2 seamlessly. We'll assume that code from the first module
    (Sensory Perception System) is available in the same project and
    can be imported or directly used here.
    """

    # Simulate or directly call the function from Module 1
    # For demonstration, we rebuild a minimal snippet of code from Module 1 here
    # or you can import it if it's in a separate file.

    # This mock function returns an annotated output similar to Module 1's last step:
    def mock_module_1_output():
        return {
            "pyramid_vector": [0.1, 0.2, 0.3],
            "importance_measure": 2.5,
            "environment_tag": "low_light"
        }

    annotated_data = mock_module_1_output()
    # Now feed it into Module 2
    result_module_2 = demo_dynamic_causal_reasoner_module_2(annotated_data)
    # Here you'd proceed or combine with subsequent modules (Adaptive Strategy Refinement, etc.)


if __name__ == "__main__":
    # If you wish to test just the second module, you can simulate a typical annotated output from Module 1
    sample_annotated_output = {
        "pyramid_vector": [0.5, 0.7, 0.6, 0.4],
        "importance_measure": 3.2,
        "environment_tag": "foggy"
    }
    demo_dynamic_causal_reasoner_module_2(sample_annotated_output)
