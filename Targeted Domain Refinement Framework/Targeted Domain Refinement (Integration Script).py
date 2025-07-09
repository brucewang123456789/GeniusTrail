"""
Integrated TDR Framework
---------------------------
This script unifies all six TDR modules into one pipeline, designed to run
in Python 3.13 IDLE with no plug-ins. It can handle synthetic domain mismatch
data (friction, slope, lighting, obstacles) to demonstrate how TDR surpasses
ADR's random domain approach.

Modules & Flow:
  1) Sensory Perception System
  2) Dynamic Causal Reasoner
  3) Adaptive Policy Refinement
  4) Knowledge Integration & Meta-Learning
  5) Predictive Representation & Internal Simulation
  6) Cross-Domain Validation & Deployment

You can feed in synthetic domain parameters (mechanical factors like friction/slope,
environmental factors like lighting/obstacles) to see TDR adapt and log results.

Usage:
  python tdr_integrated.py

Note:
  - The code is conceptual and simplified for demonstration.
  - Adjust domain ranges, tasks, or data logging as needed.
  - No external libraries (e.g., numpy) are used.
"""

import random


# =========================================================
# =========== MODULE 1: SENSORY PERCEPTION SYSTEM =========
# =========================================================
class SensorArray:
    """
    Simulates or receives raw sensor inputs (multi-modal).
    For demonstration, we store environment data as raw arrays.
    """

    def capture_data(self, domain_params):
        """
        domain_params might include friction, slope, lighting, obstacles.
        We'll just pack them into a synthetic sensor reading.
        Returns a dictionary that later modules can parse.
        """
        # Example sensor representation:
        sensor_readings = {
            "mechanical_factors": [
                domain_params.get("friction", 1.0),
                domain_params.get("slope", 0.0)
            ],
            "environmental_factors": [
                1.0 if domain_params.get("lighting", "normal") == "bright" else 0.5,
                len(domain_params.get("obstacles", []))
            ]
        }
        return sensor_readings


class SignalProcessor:
    """
    Denoises / normalizes sensor arrays. We do trivial normalization here.
    """

    def process(self, sensor_data):
        processed = {}
        for key, values in sensor_data.items():
            max_val = 0.0
            for v in values:
                if abs(v) > max_val:
                    max_val = abs(v)
            # Avoid division by zero
            if max_val == 0.0:
                processed[key] = values
            else:
                processed[key] = [val / max_val for val in values]
        return processed


class SparseAttention:
    """
    Emphasizes sensor streams with higher average magnitude
    (after normalization). We do a simplistic weighting approach.
    """

    def apply_attention(self, processed_data):
        # Compute average magnitude in each list
        attention_scores = {}
        for k, vals in processed_data.items():
            if len(vals) == 0:
                score = 0.0
            else:
                score = sum(vals) / len(vals)
            attention_scores[k] = score

        # Sum scores for normalization
        total_score = sum(attention_scores.values())
        if total_score == 0.0:
            # fallback uniform
            normalized = {}
            for k in attention_scores:
                normalized[k] = 1.0
        else:
            normalized = {}
            for k, sc in attention_scores.items():
                normalized[k] = sc / total_score

        # Weighted sum (flattening the categories)
        merged_output = []
        for k, vals in processed_data.items():
            weight = normalized[k]
            for v in vals:
                merged_output.append(v * weight)

        return merged_output


class FeaturePyramidExtractor:
    """
    Produces a multi-scale representation of the merged sensor data.
    Example: scale1=all, scale2=economical, etc.
    """

    def build_feature_pyramid(self, merged_output):
        scale1 = merged_output[:]
        scale2 = merged_output[::2]
        scale3 = merged_output[::3]
        # Concatenate
        pyramid_vector = scale1 + scale2 + scale3
        return pyramid_vector


class SensorFusionEngine:
    """
    Fuses the feature pyramid into one representation.
    Here, we just store a sum and the vector.
    """

    def fuse_features(self, pyramid_vector):
        total_sum = 0.0
        for val in pyramid_vector:
            total_sum += val
        fused_output = {
            "pyramid_vector": pyramid_vector,
            "importance_measure": total_sum
        }
        return fused_output


class ContextualAnnotator:
    """
    Annotates with environment tags (like 'lighting' or 'obstacles').
    We'll do a simple classification logic for demonstration.
    """

    def annotate(self, fused_output, domain_params):
        # E.g., read lighting from domain_params
        env_tag = domain_params.get("lighting", "normal")
        obstacle_count = len(domain_params.get("obstacles", []))

        annotated = {
            "pyramid_vector": fused_output["pyramid_vector"],
            "importance_measure": fused_output["importance_measure"],
            "env_tag": env_tag,
            "obstacle_count": obstacle_count
        }
        return annotated


# =========================================================
# ========= MODULE 2: DYNAMIC CAUSAL REASONER =============
# =========================================================
class CausalInferenceEngine:
    """
    Infers state_value from annotated data. Could incorporate domain knowledge,
    but we do a simplified approach here.
    """

    def __init__(self):
        self.base_influence = 0.5

    def infer_state(self, annotated_data):
        # For demonstration, sum up importance + obstacle_count
        # plus an environment factor.
        vector_mean = 0.0
        vec = annotated_data["pyramid_vector"]
        if len(vec) > 0:
            vector_mean = sum(vec) / float(len(vec))

        env_tag = annotated_data["env_tag"]
        env_factor = 1.0
        if env_tag == "bright":
            env_factor = 1.2
        elif env_tag == "dim":
            env_factor = 0.8
        elif env_tag == "normal":
            env_factor = 1.0

        obstacle_count = annotated_data["obstacle_count"]

        state_value = (vector_mean + annotated_data["importance_measure"] * 0.1
                       + self.base_influence * env_factor
                       - 0.1 * obstacle_count)

        inferred_state = {
            "state_value": state_value,
            "env_tag": env_tag,
            "obstacle_count": obstacle_count
        }
        return inferred_state


class PredictiveErrorMonitor:
    """
    Predicts an outcome from the current state_value, compares with 
    a simulated or observed outcome, and updates model parameters.
    """

    def __init__(self, learning_rate=0.02):
        self.learning_rate = learning_rate
        self.base_param = 1.0

    def predict_and_update(self, causal_engine, inferred_state):
        # mock predicted outcome
        pred_outcome = inferred_state["state_value"] * self.base_param
        # mock observation
        observed = pred_outcome + (random.random() - 0.5) * 0.5

        error = abs(pred_outcome - observed)
        # update base_param
        new_val = self.base_param - self.learning_rate * error
        if new_val < 0.1:
            new_val = 0.1
        self.base_param = new_val

        # Re-run inference with updated base_influence if desired.
        # We'll just do a minor tweak:
        old_influence = causal_engine.base_influence
        new_influence = old_influence - (self.learning_rate * 0.1 * error)
        if new_influence < 0.1:
            new_influence = 0.1
        causal_engine.base_influence = new_influence

        updated_state = {
            "pred_outcome": pred_outcome,
            "observed_outcome": observed,
            "error": error,
            "causal_param": causal_engine.base_influence,
            "outcome_param": self.base_param
        }
        return updated_state


class HierarchicalDecisionPlanner:
    """
    Decides a high-level action or sub-goal based on updated state.
    For demonstration, we output a simple 'action_id' or numeric plan.
    """

    def plan_action(self, updated_state):
        # If updated_state["error"] is large => "cautious" action
        if updated_state["error"] > 0.25:
            action_id = "ACTION_CAUTIOUS"
        else:
            action_id = "ACTION_FORWARD"

        # We'll just return a dictionary
        return {
            "action_id": action_id,
            "debug_info": updated_state
        }


# =========================================================
# ===== MODULE 3: ADAPTIVE POLICY REFINEMENT ============
# =========================================================
class DynamicActionSelector:
    """
    Refines the high-level action into a more concrete command 
    (e.g., velocity, turn_rate).
    """

    def __init__(self):
        self.default_speed = 1.0
        self.cautious_factor = 0.6

    def choose_action(self, planner_output):
        act_id = planner_output["action_id"]
        if act_id == "ACTION_CAUTIOUS":
            velocity = self.default_speed * self.cautious_factor
            turn_rate = 0.2
        else:
            velocity = self.default_speed
            turn_rate = 0.0

        chosen_action = {
            "velocity": velocity,
            "turn_rate": turn_rate
        }
        return chosen_action


class PolicyAdjustmentEngine:
    """
    Adjusts policy parameters based on environment feedback.
    Could incorporate a 'Contextual Parameter Tuner.'
    """

    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.context_factor = 1.0

    def update_policy(self, chosen_action, domain_params, performance_feedback):
        """
        For demonstration, if friction is low or slope is high, we 
        tweak context_factor. performance_feedback might be +1 or -1 
        from a success/fail check.
        """
        friction = domain_params.get("friction", 1.0)
        slope = domain_params.get("slope", 0.0)

        # If slope is large or friction is small => we increase context factor
        # so the policy might lower velocity, etc.
        condition_score = (1.0 - friction) + (slope / 10.0)
        self.context_factor += (condition_score * self.learning_rate * performance_feedback)

        if self.context_factor < 0.5:
            self.context_factor = 0.5
        elif self.context_factor > 2.0:
            self.context_factor = 2.0


class FeedbackOptimizationUnit:
    """
    Observes final outcome, applies a small correction if mismatch is large.
    """

    def __init__(self, correction_gain=0.05):
        self.correction_gain = correction_gain

    def optimize(self, chosen_action, real_outcome):
        """
        Example: Compare chosen velocity vs. actual velocity we see, 
        do a minor correction.
        """
        intended_v = chosen_action["velocity"]
        measured_v = real_outcome.get("measured_velocity", intended_v)

        diff = measured_v - intended_v
        correction = diff * self.correction_gain
        final_vel = intended_v + correction

        optimized_command = {
            "velocity": final_vel,
            "turn_rate": chosen_action["turn_rate"]
        }
        return optimized_command


# =========================================================
# == MODULE 4: KNOWLEDGE INTEGRATION & META-LEARNING ======
# =========================================================
class EpisodicMemoryVault:
    """
    Stores records from each TDR run for meta-learning or analysis.
    """

    def __init__(self):
        self.memory_storage = []

    def store_episode(self, record):
        self.memory_storage.append(record)

    def retrieve_all(self):
        return self.memory_storage


class SemanticKnowledgeGraph:
    """
    Captures relationships (e.g., env_tag => success rate, friction => difficulty).
    """

    def __init__(self):
        self.knowledge_relations = {}

    def update_graph(self, episodes):
        for ep in episodes:
            env_tag = ep.get("env_tag", "unknown")
            friction = ep.get("friction", 1.0)
            success = ep.get("success", False)

            key = "env:"+env_tag
            if key not in self.knowledge_relations:
                self.knowledge_relations[key] = {
                    "count": 0,
                    "success_count": 0,
                    "frictions": []
                }
            self.knowledge_relations[key]["count"] += 1
            if success:
                self.knowledge_relations[key]["success_count"] += 1
            self.knowledge_relations[key]["frictions"].append(friction)

    def summarize(self):
        summary = {}
        for k, val in self.knowledge_relations.items():
            c = val["count"]
            s = val["success_count"]
            if c > 0:
                sr = float(s) / c
            else:
                sr = 0.0
            avg_f = 0.0
            if len(val["frictions"]) > 0:
                avg_f = sum(val["frictions"]) / len(val["frictions"])
            summary[k] = {
                "success_rate": sr,
                "avg_friction": avg_f
            }
        return summary


class MetaLearningScheduler:
    """
    Uses the memory to adapt meta-parameters if certain tasks/environments 
    are repeatedly problematic.
    """

    def __init__(self):
        self.global_adaptation_rate = 0.1

    def schedule_learning(self, episodes):
        # Example: if success_rate < 0.7 in some environment => raise global adaptation
        total_episodes = len(episodes)
        fail_count = sum(1 for e in episodes if not e.get("success", False))
        fail_ratio = (fail_count / float(total_episodes)) if total_episodes > 0 else 0.0

        # If fail_ratio is high, we bump adaptation rate
        if fail_ratio > 0.3:
            self.global_adaptation_rate += 0.02
            if self.global_adaptation_rate > 5.0:
                self.global_adaptation_rate = 5.0

        return {
            "global_adaptation_rate": self.global_adaptation_rate,
            "fail_ratio": fail_ratio
        }


class TransferCatalysisUnit:
    """
    Aligns or adapts knowledge gleaned from multiple tasks and environments
    for cross-domain usage.
    """

    def __init__(self):
        self.alignment_strength = 0.5

    def catalyze_transfer(self, knowledge_summary, meta_params):
        # For demonstration, produce a dictionary of 'transfer_factor' per env.
        aligned_knowledge = {}
        for env_key, info in knowledge_summary.items():
            # if success_rate < 0.8 => higher transfer factor
            if info["success_rate"] < 0.8:
                factor = (1.0 - info["success_rate"]) * self.alignment_strength * meta_params["global_adaptation_rate"]
            else:
                factor = 0.1 * meta_params["global_adaptation_rate"]
            aligned_knowledge[env_key] = {
                "transfer_factor": factor,
                "avg_friction": info["avg_friction"]
            }
        return aligned_knowledge


# =========================================================
# = MODULE 5: PREDICTIVE REPRESENTATION & INTERNAL SIM.  =
# =========================================================
class GenerativeWorldModel:
    """
    Builds a simplified model of environment dynamics for internal simulation.
    """

    def __init__(self):
        self.base_dynamics_factor = 1.0

    def integrate_knowledge(self, aligned_knowledge, meta_params):
        # If there's high 'transfer_factor' in many envs, we might raise or lower base_dynamics_factor
        total_transfer = 0.0
        for k, info in aligned_knowledge.items():
            total_transfer += info["transfer_factor"]
        adapt_rate = meta_params.get("global_adaptation_rate", 0.1)
        self.base_dynamics_factor += (total_transfer * adapt_rate * 0.001)
        if self.base_dynamics_factor < 0.1:
            self.base_dynamics_factor = 0.1
        elif self.base_dynamics_factor > 10.0:
            self.base_dynamics_factor = 10.0

    def simulate_next_state(self, current_state, action):
        """
        current_state might be {"x":..., "y":..., ...}, we do minimal logic
        """
        x = current_state.get("x", 0.0)
        y = current_state.get("y", 0.0)

        velocity = action.get("velocity", 0.0)
        turn_rate = action.get("turn_rate", 0.0)

        new_x = x + (velocity * self.base_dynamics_factor)
        new_y = y + (turn_rate * 0.1 * self.base_dynamics_factor)

        return {
            "x": new_x,
            "y": new_y
        }


class CounterfactualScenarioEngine:
    """
    Creates hypothetical (counterfactual) variations of the action or environment 
    to refine the generative model or plan safer strategies.
    """

    def __init__(self, gen_model):
        self.gen_model = gen_model
        self.scenario_count = 3

    def explore_counterfactuals(self, current_state, base_action):
        scenarios = []
        for _ in range(self.scenario_count):
            v_offset = (random.random() - 0.5) * 0.4
            t_offset = (random.random() - 0.5) * 0.2
            alt_action = {
                "velocity": base_action["velocity"] + v_offset,
                "turn_rate": base_action["turn_rate"] + t_offset
            }
            sim_next = self.gen_model.simulate_next_state(current_state, alt_action)
            scenarios.append({
                "variant_action": alt_action,
                "sim_next_state": sim_next
            })
        return scenarios


class InterventionManager:
    """
    Decides if real-world trials or further simulation is needed 
    based on scenario risk or mismatch.
    """

    def __init__(self, risk_threshold=0.3):
        self.risk_threshold = risk_threshold

    def decide_intervention(self, scenarios):
        max_risk = 0.0
        for sc in scenarios:
            # trivial risk measure: velocity + 0.1 * turn
            v = sc["variant_action"]["velocity"]
            t = sc["variant_action"]["turn_rate"]
            scenario_risk = v * 0.1 + t * 0.05
            if scenario_risk > max_risk:
                max_risk = scenario_risk

        proceed_real = (max_risk < self.risk_threshold)
        return {
            "proceed_real_test": proceed_real,
            "max_risk": max_risk
        }


class ModelDivergenceAnalyzer:
    """
    Monitors how far predictions deviate, adjusting or logging potential drift.
    """

    def __init__(self):
        self.cumulative_divergence = 0.0

    def analyze_divergence(self, scenarios):
        # sum up positions for demonstration
        total_pos = 0.0
        for sc in scenarios:
            sx = sc["sim_next_state"]["x"]
            sy = sc["sim_next_state"]["y"]
            total_pos += (sx*sx + sy*sy)**0.5
        avg_pos = 0.0
        if len(scenarios) > 0:
            avg_pos = total_pos / len(scenarios)

        self.cumulative_divergence += avg_pos * 0.001
        return {
            "cumulative_divergence": self.cumulative_divergence,
            "average_scenario_pos": avg_pos
        }


# =========================================================
# == MODULE 6: CROSS-DOMAIN VALIDATION & DEPLOYMENT ======
# =========================================================
class PolicyValidationSuite:
    """
    Tests TDR's final policy or outputs under varied domain conditions.
    """

    def validate_policy(self, logs):
        # logs might store success/fail info from actual episodes
        total_episodes = len(logs)
        if total_episodes == 0:
            return {"success_rate": 0.0, "episodes": 0}
        success_count = sum(1 for r in logs if r.get("success", False))
        return {
            "success_rate": float(success_count) / total_episodes,
            "episodes": total_episodes
        }


class RealTimeSafetyAuditor:
    """
    Checks if TDR operation has encountered anomalies. If so, 
    we might restrict deployment.
    """

    def audit(self, validation_result):
        sr = validation_result["success_rate"]
        # If success_rate < 0.7 => raise caution
        safe = (sr >= 0.7)
        return {
            "safe_to_deploy": safe,
            "success_rate": sr
        }


class HierarchicalDeploymentManager:
    """
    Rolls out the final policy in phases if safe.
    """

    def deploy_policy(self, safety_report):
        if not safety_report["safe_to_deploy"]:
            return [{"phase": "aborted", "reason": "low success_rate"}]

        phases = ["lab_test", "restricted_field", "full_deployment"]
        record = []
        for ph in phases:
            # random pass/fail chance for demonstration
            chance = random.random()
            if chance < 0.1:
                record.append({"phase": ph, "status": "failed"})
                break
            else:
                record.append({"phase": ph, "status": "success"})
        return record


class ComparativeBenchmarkingEngine:
    """
    (Optional) Compares TDR results to an ADR baseline if you feed them in.
    """

    def run_benchmark(self, tdr_success, adr_success):
        # trivial comparison
        better = (tdr_success > adr_success)
        return {
            "tdr_success": tdr_success,
            "adr_success": adr_success,
            "tdr_better": better
        }


# =========================================================
# =============== TDR INTEGRATED SIMULATION ==============
# =========================================================

class TDRIntegratedSimulator:
    """
    Orchestrates a single-run approach: 
      1) Takes synthetic domain parameters,
      2) Runs TDR pipeline (Modules 1..6),
      3) Returns success/fail plus logs.
    """

    def __init__(self):
        # Instantiate module components in a logical chain
        # (You can store them in the class so we can reuse.)
        # 1. Sensory Perception
        self.sensor_array = SensorArray()
        self.signal_processor = SignalProcessor()
        self.sparse_attention = SparseAttention()
        self.feature_pyramid = FeaturePyramidExtractor()
        self.fusion_engine = SensorFusionEngine()
        self.annotator = ContextualAnnotator()

        # 2. Dynamic Causal Reasoner
        self.causal_engine = CausalInferenceEngine()
        self.pred_monitor = PredictiveErrorMonitor(learning_rate=0.02)
        self.decision_planner = HierarchicalDecisionPlanner()

        # 3. Adaptive Policy Refinement
        self.action_selector = DynamicActionSelector()
        self.policy_engine = PolicyAdjustmentEngine(learning_rate=0.01)
        self.feedback_optimizer = FeedbackOptimizationUnit(correction_gain=0.05)

        # 4. Knowledge Integration
        self.memory_vault = EpisodicMemoryVault()
        self.knowledge_graph = SemanticKnowledgeGraph()
        self.meta_scheduler = MetaLearningScheduler()
        self.transfer_unit = TransferCatalysisUnit()

        # 5. Predictive Representation
        self.gen_world_model = GenerativeWorldModel()
        self.cf_scenario_engine = CounterfactualScenarioEngine(self.gen_world_model)
        self.intervention_mgr = InterventionManager(risk_threshold=0.3)
        self.divergence_analyzer = ModelDivergenceAnalyzer()

        # 6. Cross-Domain Validation
        self.validation_suite = PolicyValidationSuite()
        self.safety_auditor = RealTimeSafetyAuditor()
        self.deployment_mgr = HierarchicalDeploymentManager()
        self.benchmark_engine = ComparativeBenchmarkingEngine()

        # Logging
        self.run_logs = []

    def run_episode(self, domain_params):
        """
        Simulate one TDR 'episode' with the given domain parameters.
        We'll define success/fail in a simplified manner, just like ADR baseline.
        """
        # ---------- MODULE 1: SENSORY PERCEPTION ----------
        raw_data = self.sensor_array.capture_data(domain_params)
        processed = self.signal_processor.process(raw_data)
        merged = self.sparse_attention.apply_attention(processed)
        pyramid = self.feature_pyramid.build_feature_pyramid(merged)
        fused = self.fusion_engine.fuse_features(pyramid)
        annotated = self.annotator.annotate(fused, domain_params)

        # ---------- MODULE 2: DYNAMIC CAUSAL REASONER ----------
        inferred_state = self.causal_engine.infer_state(annotated)
        updated_pred = self.pred_monitor.predict_and_update(self.causal_engine, inferred_state)
        planner_output = self.decision_planner.plan_action(updated_pred)

        # ---------- MODULE 3: ADAPTIVE POLICY REFINEMENT ----------
        chosen_action = self.action_selector.choose_action(planner_output)
        # We'll define a trivial performance_feedback: 
        # If friction is low or slope is high => more likely fail => performance_feedback = -1 or +1
        friction = domain_params.get("friction", 1.0)
        slope = domain_params.get("slope", 0.0)
        # success if velocity < threshold that depends on friction/slope
        # replicate logic similar to ADR baseline
        threshold = friction + (1.0 - (slope / 10.0))
        success = (chosen_action["velocity"] <= threshold)
        perf_feedback = 1 if success else -1

        self.policy_engine.update_policy(chosen_action, domain_params, perf_feedback)

        # Real outcome (mock)
        real_outcome = {
            "measured_velocity": chosen_action["velocity"] + (random.random() - 0.5) * 0.2
        }
        optimized_cmd = self.feedback_optimizer.optimize(chosen_action, real_outcome)

        # ---------- MODULE 4: KNOWLEDGE INTEGRATION & META-LEARNING ----------
        # For demonstration, store an episodic record
        ep_record = {
            "domain_params": domain_params,
            "success": success,
            "env_tag": domain_params.get("lighting", "normal"),
            "friction": friction
        }
        self.memory_vault.store_episode(ep_record)

        # ---------- MODULE 5: PREDICTIVE REPRESENTATION & INTERNAL SIM ----------
        # Suppose we define current_state for the agent as x=0,y=0
        current_state = {"x": 0.0, "y": 0.0}
        # The generative model might adapt based on knowledge
        # but first we do batch updates after many episodes, see run_all_episodes
        # We'll just do a single scenario exploration here
        cf_scenarios = self.cf_scenario_engine.explore_counterfactuals(current_state, optimized_cmd)
        intervention_decision = self.intervention_mgr.decide_intervention(cf_scenarios)
        divergence_summary = self.divergence_analyzer.analyze_divergence(cf_scenarios)

        # ---------- Summarize partial result for this episode ----------
        ep_data = {
            "domain_params": domain_params,
            "success": success,
            "action_chosen": chosen_action,
            "optimized_action": optimized_cmd,
            "intervention_decision": intervention_decision,
            "divergence_summary": divergence_summary
        }
        self.run_logs.append(ep_data)
        return success

    def run_all_episodes(self, domain_param_list):
        """
        Runs multiple episodes, each with a distinct or randomly generated domain_params.
        Then applies knowledge integration (Module 4) 
        and final validations (Module 6).
        """
        success_count = 0
        total_eps = len(domain_param_list)

        # Execute each episode
        for dp in domain_param_list:
            s = self.run_episode(dp)
            if s:
                success_count += 1

        # ---------- MODULE 4 Continued: Summaries & Meta-Learning ----------
        episodes_data = self.memory_vault.retrieve_all()
        self.knowledge_graph.update_graph(episodes_data)
        knowledge_summary = self.knowledge_graph.summarize()
        meta_params = self.meta_scheduler.schedule_learning(episodes_data)
        aligned_knowledge = self.transfer_unit.catalyze_transfer(knowledge_summary, meta_params)

        # ---------- MODULE 5 Continued: Integrate Knowledge into Gen Model ----------
        self.gen_world_model.integrate_knowledge(aligned_knowledge, meta_params)
        # We don't re-run the episodes here, but you could do another pass if needed.

        # ---------- MODULE 6: Validation & Deployment ----------
        validation_result = self.validation_suite.validate_policy(self.run_logs)
        safety_report = self.safety_auditor.audit(validation_result)
        deployment_record = self.deployment_mgr.deploy_policy(safety_report)

        # Return final summary
        tdr_final_out = {
            "success_rate": float(success_count) / total_eps if total_eps > 0 else 0.0,
            "validation_result": validation_result,
            "safety_report": safety_report,
            "deployment_record": deployment_record,
            "meta_params": meta_params,
            "aligned_knowledge": aligned_knowledge
        }
        return tdr_final_out


# =========================================================
# =========== EXAMPLE MAIN DEMO / ENTRY POINT =============
# =========================================================
def run_TDR_experiment():
    """
    Demonstrates how to create synthetic data for domain mismatch
    (mechanical + environmental factors) and feed it into the TDR pipeline.
    """

    # 1) Create a list of domain parameters for each episode
    # We'll do 20 episodes, mixing friction, slope, lighting, obstacles

    domain_list = []
    possible_lightings = ["normal", "bright", "dim"]
    for _ in range(20):
        friction_val = random.uniform(0.5, 1.0)
        slope_val = random.uniform(0.0, 10.0)
        lighting_opt = random.choice(possible_lightings)
        # randomly assign obstacles
        obstacle_count = random.randint(0, 3)
        obstacles = []
        for i in range(obstacle_count):
            obstacles.append({"id": i, "size": random.uniform(0.2, 1.0)})

        dp = {
            "friction": friction_val,
            "slope": slope_val,
            "lighting": lighting_opt,
            "obstacles": obstacles
        }
        domain_list.append(dp)

    # 2) Initialize TDR integrated simulator
    tdr_system = TDRIntegratedSimulator()

    # 3) Run all episodes
    tdr_result = tdr_system.run_all_episodes(domain_list)

    # 4) Print final TDR results
    print("=== TDR Experiment Complete ===")
    print("Total Episodes:", len(domain_list))
    print("TDR Success Rate: {:.2f}".format(tdr_result["success_rate"]))
    print("Validation:", tdr_result["validation_result"])
    print("Safety Report:", tdr_result["safety_report"])
    print("Deployment Record:", tdr_result["deployment_record"])
    print("Meta Params:", tdr_result["meta_params"])
    print("Aligned Knowledge:", tdr_result["aligned_knowledge"])


if __name__ == "__main__":
    run_TDR_experiment()
