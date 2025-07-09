"""
TDR Framework - Module 3: Adaptive Policy Refinement
---------------------------------------------------------
Order of components from top to bottom:
  1) Dynamic Action Selector
  2) Policy Adjustment Engine (subcomponent: Contextual Parameter Tuner)
  3) Feedback Optimization Unit

This module takes the output of Module 2 (e.g., 'high_level_action', 'monitor_result',
and 'inferred_state') and refines the robot's action plans in a more context-aware way
than ADR, which relies on random domain perturbations. The subcomponent, Contextual
Parameter Tuner, further adjusts parameters in real time, while the Feedback Optimization
Unit applies final corrections. The structure anticipates sending episodic data into
Module 4 for knowledge integration and meta-learning.

Code Requirements:
  - Pure Python 3.13 (no external libraries such as NumPy).
  - Must seamlessly interface with the final outputs of Module 2.
"""

import random


# ========================================================
# (1) DYNAMIC ACTION SELECTOR (TOP of Module 3)
# ========================================================
class DynamicActionSelector:
    """
    Refines the high-level directive from Module 2 into specific motion or control outputs.
    For example, 'ACTION_FORWARD' might become a velocity vector, while 'ACTION_CAUTIOUS_TURN'
    might become a reduced forward speed with a turning rate.

    In surpassing ADR, we focus on context-driven adjustments rather than randomizing
    domain parameters.
    """

    def __init__(self):
        # Could store default control specs here.
        self.default_speed = 1.0  # Example base speed
        self.cautious_speed_factor = 0.5

    def select_action(self, module2_output):
        """
        module2_output is expected to be the dictionary returned
        by the second module, containing e.g.:
          "high_level_action": { "action_id": "ACTION_CAUTIOUS_TURN", "sub_goal": ... }
          "inferred_state": ...
          "monitor_result": ...
        
        Returns a dictionary with a more concrete 'action_command'.
        """
        high_level_action = module2_output.get("high_level_action", {})
        action_id = high_level_action.get("action_id", "ACTION_UNKNOWN")

        if action_id == "ACTION_FORWARD":
            # Straight line motion, normal speed
            velocity = self.default_speed
            turn_rate = 0.0
        elif action_id == "ACTION_CAUTIOUS_TURN":
            # Slower speed, some turning
            velocity = self.default_speed * self.cautious_speed_factor
            turn_rate = 0.3  # arbitrary turn rate
        else:
            velocity = 0.0
            turn_rate = 0.0

        # We may embed sub_goals or environment tags in the final command
        sub_goal = high_level_action.get("sub_goal", None)

        action_command = {
            "velocity": velocity,
            "turn_rate": turn_rate,
            "sub_goal": sub_goal
        }
        return action_command


# =====================================================================
# (2) POLICY ADJUSTMENT ENGINE (MIDDLE of Module 3)
#        Subcomponent: Contextual Parameter Tuner
# =====================================================================
class ContextualParameterTuner:
    """
    Nested subcomponent that adjusts certain policy or control parameters
    based on immediate environmental or performance signals. This allows
    a more granular adaptation than ADR's broad domain randomization.
    """

    def __init__(self, param_gain=0.05):
        self.param_gain = param_gain
        # Example parameter that might be tuned
        self.context_factor = 1.0

    def tune_parameters(self, environment_tag, performance_feedback):
        """
        Adjusts 'context_factor' or other internal variables depending on environment
        or immediate performance feedback. Real systems might handle friction, lighting,
        or concurrency. Here, we do a trivial demonstration.

        :param environment_tag: A string describing the environment (e.g., 'indoor', 'foggy').
        :param performance_feedback: A numeric or symbolic feedback from recent outcomes.
        """
        # Example logic: environment_tag influences direction of tuning
        if environment_tag == "low_light":
            self.context_factor += (self.param_gain * 0.5)  # adapt less aggressively
        elif environment_tag == "foggy":
            self.context_factor += (self.param_gain * 1.0)  # normal adaptation
        else:
            self.context_factor += (self.param_gain * 0.2)  # minimal adaptation

        # performance_feedback might accelerate or decelerate this factor
        # if feedback > 0 => good performance => keep adjusting upward
        # if feedback < 0 => poor performance => reduce context factor
        self.context_factor += (performance_feedback * 0.01)
        # Ensure it doesn't go negative or explode
        if self.context_factor < 0.1:
            self.context_factor = 0.1
        elif self.context_factor > 5.0:
            self.context_factor = 5.0

class PolicyAdjustmentEngine:
    """
    Incorporates context signals (e.g., environment tags, immediate feedback) to
    refine the action command. Surpasses ADR by selectively tuning policy parameters
    instead of sampling random domain variations.
    """

    def __init__(self, base_learning_rate=0.01):
        self.base_learning_rate = base_learning_rate
        self.context_tuner = ContextualParameterTuner(param_gain=0.05)

    def adjust_policy(self, action_command, environment_tag, performance_feedback):
        """
        Merges real-time environment info, performance signals, and the base action command
        from the DynamicActionSelector. Possibly modifies velocity, turning rate, or other
        parameters.

        :param action_command: dictionary like { "velocity": 1.0, "turn_rate": 0.3, ... }
        :param environment_tag: e.g. 'indoor', 'foggy', or user-defined tags
        :param performance_feedback: numeric measure of how well recent actions fared
        """
        # 1) Update context parameters
        self.context_tuner.tune_parameters(environment_tag, performance_feedback)

        # 2) Modify the action command
        # Example logic: scale velocity by context_factor
        current_factor = self.context_tuner.context_factor
        adjusted_velocity = action_command["velocity"] * current_factor
        adjusted_turn_rate = action_command["turn_rate"]

        # Suppose we also reduce turn_rate if performance feedback is negative
        if performance_feedback < 0:
            adjusted_turn_rate *= 0.8

        # Return updated action command
        adjusted_command = {
            "velocity": adjusted_velocity,
            "turn_rate": adjusted_turn_rate,
            "sub_goal": action_command["sub_goal"]
        }
        return adjusted_command


# =========================================================
# (3) FEEDBACK OPTIMIZATION UNIT (BOTTOM of Module 3)
# =========================================================
class FeedbackOptimizationUnit:
    """
    Observes the real outcome (e.g., measured displacement, orientation),
    compares it to the intended command, and applies final corrections if
    the mismatch is beyond a threshold. This real-time feedback approach
    is more precise than ADR’s random domain coverage.

    Could also store episodic data for the next module (Knowledge Integration).
    """

    def __init__(self, mismatch_threshold=0.2, correction_gain=0.05):
        self.mismatch_threshold = mismatch_threshold
        self.correction_gain = correction_gain

    def optimize(self, final_command, actual_outcome):
        """
        Compares final_command with actual_outcome. If mismatch is large,
        returns a correction factor for further adjustments or logging.

        :param final_command: e.g. { "velocity": x, "turn_rate": y }
        :param actual_outcome: measured outcome after executing the command
                               on hardware or a simulator
        :return: (corrected_command, feedback_info) 
        """
        # For simplicity, actual_outcome might be a dictionary with "actual_velocity", "actual_turn"
        intended_velocity = final_command["velocity"]
        intended_turn = final_command["turn_rate"]

        measured_velocity = actual_outcome.get("actual_velocity", 0.0)
        measured_turn = actual_outcome.get("actual_turn", 0.0)

        # Evaluate mismatch as Euclidean distance or a simple difference
        velocity_diff = abs(intended_velocity - measured_velocity)
        turn_diff = abs(intended_turn - measured_turn)

        # If mismatch is large, create a correction
        corrected_velocity = intended_velocity
        corrected_turn = intended_turn
        mismatch_flag = False

        if velocity_diff > self.mismatch_threshold:
            mismatch_flag = True
            # Move velocity slightly towards measured value
            correction_v = self.correction_gain * (measured_velocity - intended_velocity)
            corrected_velocity += correction_v

        if turn_diff > self.mismatch_threshold:
            mismatch_flag = True
            # Move turn rate slightly towards measured value
            correction_t = self.correction_gain * (measured_turn - intended_turn)
            corrected_turn += correction_t

        # Build final command post-correction
        corrected_command = {
            "velocity": corrected_velocity,
            "turn_rate": corrected_turn,
            "sub_goal": final_command.get("sub_goal")
        }

        # feedback_info might store data for the Episodic Memory Vault (Module 4).
        feedback_info = {
            "mismatch_occurred": mismatch_flag,
            "velocity_diff": velocity_diff,
            "turn_diff": turn_diff,
            "corrected_velocity": corrected_velocity,
            "corrected_turn": corrected_turn
        }
        return (corrected_command, feedback_info)


# =======================================================================
# DEMONSTRATION / TEST ROUTINE FOR MODULE 3 (ADAPTIVE POLICY REFINEMENT)
# =======================================================================
def demo_adaptive_policy_refinement_module_3(module2_results):
    """
    Demonstrates how data flows from Module 2 into the third module's top-to-bottom components:
      1) Dynamic Action Selector
      2) Policy Adjustment Engine (with subcomponent Contextual Parameter Tuner)
      3) Feedback Optimization Unit

    :param module2_results: dictionary from Module 2's final output, for example:
        {
          "inferred_state": { ... },
          "high_level_action": { "action_id": "ACTION_FORWARD", "sub_goal": ... },
          "monitor_result": { ... }
        }
    """

    print("====== Adaptive Policyy Refinement (Module 3) ======")

    # Top: Dynamic Action Selector
    action_selector = DynamicActionSelector()
    action_command = action_selector.select_action(module2_results)
    print("[Dynamic Action Selector] Action Command:", action_command)

    # Middle: Policy Adjustment Engine
    policy_engine = PolicyAdjustmentEngine(base_learning_rate=0.01)
    # We'll fetch environment tag or performance feedback from the second module
    environment_tag = module2_results.get("inferred_state", {}).get("env_tag", "normal")
    # Fake a performance metric: e.g., from monitor_result's error
    error_val = module2_results.get("monitor_result", {}).get("error", 0.0)
    # Suppose we interpret a small error as positive feedback, large error negative
    performance_feedback = -1.0 if error_val > 0.5 else 1.0

    adjusted_command = policy_engine.adjust_policy(action_command,
                                                   environment_tag,
                                                   performance_feedback)
    print("[Policy Adjustment Engine] Adjusted Command:", adjusted_command)

    # Bottom: Feedback Optimization Unit
    feedback_opt_unit = FeedbackOptimizationUnit(mismatch_threshold=0.2, correction_gain=0.05)
    # We simulate an actual outcome. In a real system, we'd measure actual movement.
    actual_outcome = {
        "actual_velocity": adjusted_command["velocity"] + (random.random() - 0.5) * 0.4,  # +/- 0.2 offset
        "actual_turn": adjusted_command["turn_rate"] + (random.random() - 0.5) * 0.2
    }
    corrected_command, feedback_info = feedback_opt_unit.optimize(adjusted_command, actual_outcome)
    print("[Feedback Optimization Unit] Corrected Command:", corrected_command)
    print("[Feedback Optimization Unit] Feedback Info for Episodic Memory Vault (Module 4):", feedback_info)

    # Demonstrate how we might store data for Module 4's "Episodic Memory Vault"
    # E.g., collect the final command, environment tag, and performance outcomes
    episodic_record = {
        "final_command": corrected_command,
        "environment_tag": environment_tag,
        "performance_feedback": performance_feedback,
        "feedback_info": feedback_info
    }
    print("[Module 3] Episodic record ready for Module 4:", episodic_record)

    print("===== End of Module 3 Execution =====\n")
    return {
        "action_command": action_command,
        "adjusted_command": adjusted_command,
        "corrected_command": corrected_command,
        "feedback_info": feedback_info,
        "episodic_record": episodic_record
    }


# ==============================================
# OPTIONAL MAIN DEMO (CHAIN MODULE 2 -> MODULE 3)
# ==============================================
def main_demo_chain_module_2_and_3():
    """
    Example showing how we connect the second module to the third.
    In a real project, you'd likely import the second module's code or use
    a direct function call. For demonstration, we mock minimal data from Module 2.
    """

    # A mock result from Module 2
    module2_mock_output = {
        "inferred_state": {
            "state_value": 1.3,
            "env_tag": "low_light",
            "vector_mean": 0.6
        },
        "high_level_action": {
            "action_id": "ACTION_FORWARD",
            "sub_goal": None
        },
        "monitor_result": {
            "predicted_outcome": 1.56,
            "observed_outcome": 1.3,
            "error": 0.26,
            "updated_state": { "state_value": 1.3, "env_tag": "low_light" }
        }
    }

    # Now feed it into the third module
    module3_results = demo_adaptive_policy_refinement_module_3(module2_mock_output)
    return module3_results


if __name__ == "__main__":
    # For testing just Module 3 in isolation, run the chain demo or feed real data from Module 2
    main_demo_chain_module_2_and_3()
