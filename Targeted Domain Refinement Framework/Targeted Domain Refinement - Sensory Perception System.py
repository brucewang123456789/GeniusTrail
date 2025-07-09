"""
TDR Framework - Module 1: Sensory Perception System
------------------------------------------------------
This module implements the following sequence of components:
  (1) SensorArray
  (2) SignalProcessor
  (3) SparseAttention
  (4) FeaturePyramidExtractor
  (5) SensorFusionEngine
  (6) ContextualAnnotator

The goal is to demonstrate how each subcomponent
addresses the known limitations of Adaptive Domain Randomization (ADR)
by applying context-aware data processing rather than random perturbations.

Note: This code relies solely on built-in Python 3.13 modules.
No external libraries (e.g., NumPy) are used.
"""

import random


# ---------------------------------------------------------------------
# 1. SENSOR ARRAY
# ---------------------------------------------------------------------
class SensorArray:
    """
    The SensorArray simulates raw sensor inputs.
    In a real system, this might read from actual devices or a simulator.
    
    Attributes:
        num_sensors: How many different sensor modalities exist (e.g., camera, LiDAR).
                     For demonstration, we generate lists of floating-point values.
    """

    def __init__(self, num_sensors=3):
        self.num_sensors = num_sensors

    def capture_data(self):
        """
        Simulates capturing raw sensor data for each modality.
        Each sensor's data is stored as a list of floating-point values.
        """
        sensor_data = []
        for _ in range(self.num_sensors):
            # Generate a random list length between 5 and 15 for illustration
            data_length = random.randint(5, 15)
            # Fill with random floats in [0,1)
            data_values = [random.random() for _ in range(data_length)]
            sensor_data.append(data_values)
        return sensor_data


# ---------------------------------------------------------------------
# 2. SIGNAL PROCESSOR
# ---------------------------------------------------------------------
class SignalProcessor:
    """
    Cleans, normalizes, or otherwise preprocesses raw sensor readings.
    This step helps avoid noise or range mismatches that ADR might randomly inject.
    """

    def remove_outliers(self, data_list):
        """
        Example function to clamp extremely large or negative values.
        Since we already have [0,1) from random, we show the structure anyway.
        """
        clamped = []
        for val in data_list:
            if val < 0.0:
                clamped.append(0.0)
            elif val > 1.0:
                clamped.append(1.0)
            else:
                clamped.append(val)
        return clamped

    def normalize_data(self, data_list):
        """
        Scales the sensor values into a [0,1] range according to the max element.
        """
        maximum = 0.0
        for val in data_list:
            if val > maximum:
                maximum = val
        # Avoid division by zero
        if maximum == 0.0:
            return data_list
        return [val / maximum for val in data_list]

    def process(self, raw_data):
        """
        Applies a sequence of steps: outlier removal, then normalization.
        raw_data is a list of lists, one per sensor.
        """
        processed_data = []
        for sensor_values in raw_data:
            step1 = self.remove_outliers(sensor_values)
            step2 = self.normalize_data(step1)
            processed_data.append(step2)
        return processed_data


# ---------------------------------------------------------------------
# 3. SPARSE ATTENTION
# ---------------------------------------------------------------------
class SparseAttention:
    """
    Assigns an attention weight to each sensor channel, emphasizing
    particularly strong or unique signals.
    
    This counters ADR’s blind randomization by focusing computational effort
    on the most important sensor streams. In more advanced systems, a
    learned network could replace or augment these heuristics.
    """

    def compute_attention_weights(self, processed_data):
        """
        A simplistic approach: the average magnitude of each sensor's data
        forms the basis for attention weight. Then we normalize across sensors.
        """
        # Calculate average magnitude for each sensor
        magnitudes = []
        for sensor_values in processed_data:
            # Just compute mean
            if len(sensor_values) == 0:
                magnitudes.append(0.0)
            else:
                avg_val = sum(sensor_values) / float(len(sensor_values))
                magnitudes.append(avg_val)

        # Sum the magnitudes to create a normalizing factor
        total_magnitude = sum(magnitudes)

        # Compute normalized attention weights
        weights = []
        if total_magnitude > 0.0:
            for mag in magnitudes:
                weights.append(mag / total_magnitude)
        else:
            # If total_magnitude == 0, set uniform weights
            num_sensors = len(magnitudes)
            if num_sensors > 0:
                uniform_weight = 1.0 / num_sensors
                weights = [uniform_weight for _ in range(num_sensors)]
            else:
                weights = []

        return weights

    def apply_attention(self, processed_data):
        """
        Returns a single merged list that is the sum of each sensor’s data
        multiplied by its attention weight. Real systems may adopt more advanced
        vector or convolutional fusion, but this demonstrates the principle.
        """
        weights = self.compute_attention_weights(processed_data)
        # Determine the largest length among sensor lists
        max_length = 0
        for sensor_values in processed_data:
            if len(sensor_values) > max_length:
                max_length = len(sensor_values)

        # Build a weighted sum across sensors
        merged_output = [0.0] * max_length
        for i, sensor_values in enumerate(processed_data):
            w = weights[i]
            for idx, val in enumerate(sensor_values):
                merged_output[idx] += val * w

        return merged_output


# ---------------------------------------------------------------------
# 4. FEATURE PYRAMID EXTRACTOR
# ---------------------------------------------------------------------
class FeaturePyramidExtractor:
    """
    Produces a multi-scale representation of the merged sensor data.
    This addresses the variety of scales or resolutions that ADR might randomly manipulate.
    """

    def extract_scales(self, merged_output):
        """
        Splits or resamples the merged data at multiple 'scales'.
        Example: 
          scale1 = original data
          scale2 = every other element
          scale3 = every fourth element
        A real system might apply sliding window filters or CNN layers.
        """
        scale1 = merged_output[:]  # copy entire list
        scale2 = merged_output[::2]
        scale3 = merged_output[::4]
        return (scale1, scale2, scale3)

    def build_feature_pyramid(self, merged_output):
        """
        Concatenates multiple scales into one final list, 
        simulating a coarse multi-scale feature vector.
        """
        scale1, scale2, scale3 = self.extract_scales(merged_output)
        # Flatten them into one list
        final_pyramid = scale1 + scale2 + scale3
        return final_pyramid


# ---------------------------------------------------------------------
# 5. SENSOR FUSION ENGINE
# ---------------------------------------------------------------------
class SensorFusionEngine:
    """
    Fuses the multi-scale feature vectors into a single representation.
    In advanced setups, this might be a learned module that encodes/decodes
    feature pyramids. Here, it's a simplified approach focusing on core logic.
    """

    def fuse_features(self, pyramid_vector):
        """
        For illustration, we do a minimal transform. A real system might
        apply gating, learned embeddings, or other forms of fusion.
        """
        # Example: we simply keep the vector as is, or we might do a
        # basic statistic like computing the length or some summary.
        # We'll do a small 'dense' transformation to show the idea.
        # Since we have no external libraries, we do it in a minimal way.
        
        # We'll produce a single scalar that might represent
        # an 'importance measure', plus the original vector.
        total_sum = 0.0
        for val in pyramid_vector:
            total_sum += val
        # We output both for demonstration
        fused_output = {
            "pyramid_vector": pyramid_vector,
            "importance_measure": total_sum
        }
        return fused_output


# ---------------------------------------------------------------------
# 6. CONTEXTUAL ANNOTATOR
# ---------------------------------------------------------------------
class ContextualAnnotator:
    """
    Annotates the final fused features with contextual metadata.
    This is the final step of Module 1, handing off to the next module.
    """

    def annotate(self, fused_info):
        """
        We attach contextual tags (like environment classification, time-of-day, etc.)
        to the fused representation. For demonstration, we just pick pseudo-random
        tags that might reflect an environment's condition or some domain attribute.
        """
        # The fused_info includes a "pyramid_vector" and an "importance_measure".
        environment_types = ["indoor", "outdoor", "low_light", "foggy", "normal"]
        chosen_env = environment_types[random.randint(0, len(environment_types) - 1)]

        annotated = {
            "pyramid_vector": fused_info["pyramid_vector"],
            "importance_measure": fused_info["importance_measure"],
            "environment_tag": chosen_env
        }
        return annotated


# ---------------------------------------------------------------------
# DEMONSTRATION / TEST ROUTINE FOR MODULE 1
# ---------------------------------------------------------------------
def demo_sensory_perception_system():
    """
    Shows how raw data moves through each component of Module 1,
    culminating in a final annotated output that a subsequent module
    could use for causal inference or policy decisions.
    """

    # 1) Sensor Array
    sensor_array = SensorArray(num_sensors=3)
    raw_sensor_data = sensor_array.capture_data()

    # 2) Signal Processor
    processor = SignalProcessor()
    processed_data = processor.process(raw_sensor_data)

    # 3) Sparse Attention
    attention_module = SparseAttention()
    attention_output = attention_module.apply_attention(processed_data)

    # 4) Feature Pyramid Extractor
    feature_pyramid = FeaturePyramidExtractor()
    pyramid_vector = feature_pyramid.build_feature_pyramid(attention_output)

    # 5) Sensor Fusion Engine
    fusion_engine = SensorFusionEngine()
    fused_output = fusion_engine.fuse_features(pyramid_vector)

    # 6) Contextual Annotator
    annotator = ContextualAnnotator()
    final_annotated_output = annotator.annotate(fused_output)

    # Display the final annotated data
    print("===== Sensory Perception System: Module 1 Complete =====")
    print("Raw Sensor Data:", raw_sensor_data)
    print("Processed Data:", processed_data)
    print("Attention Output:", attention_output)
    print("Feature Pyramid Vector:", pyramid_vector)
    print("Fused Output (Importance + Vector):", fused_output)
    print("Final Annotated Output:", final_annotated_output)


if __name__ == "__main__":
    demo_sensory_perception_system()
