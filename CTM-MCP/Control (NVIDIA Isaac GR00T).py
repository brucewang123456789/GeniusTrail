# ==============================================================================
# NVIDIA Isaac GR00T N1 “thinking→behavior” control stub
# pure-Python 3.13 IDLE style (no external libs)
# ==============================================================================

import json
import random
from abc import ABC, abstractmethod
from pathlib import Path

# ==============================================================================
# BasePolicy ABC
# ==============================================================================
class BasePolicy(ABC):
    @abstractmethod
    def get_action(self, observations):
        """
        Given a dict of raw observations, return a dict of raw actions.
        """
        raise NotImplementedError

    @abstractmethod
    def get_modality_config(self):
        """
        Return the modality_config mapping (stub).
        """
        raise NotImplementedError


# ==============================================================================
# Gr00tControlStub
# ==============================================================================

class Gr00tControlStub(BasePolicy):
    """
    Stubbed GR00T_N1→action policy: all the subcomponent hooks are here,
    but we do *not* load any real model—just generate random actions.
    """

    def __init__(self,
                 model_path,
                 embodiment_tag,
                 modality_config,
                 modality_transform,
                 denoising_steps=None,
                 device=None):
        # preserve API fields
        self.model_path = Path(model_path)
        self.embodiment_tag = embodiment_tag
        self._modality_config = modality_config
        self._modality_transform = modality_transform
        self.device = device or "cpu"
        self.denoising_steps = denoising_steps or 0

        # run through the same init pipeline
        self._load_model()
        self._load_metadata(self.model_path / "experiment_cfg")
        self._load_horizons()

    # --------------------------------------------------------------------------
    # Subcomponent: model loader
    # --------------------------------------------------------------------------
    def _load_model(self):
        # no real model—just note that we'd have loaded here
        print(f"[Stub] would load model from {self.model_path}")

    # --------------------------------------------------------------------------
    # Subcomponent: metadata loader
    # --------------------------------------------------------------------------
    def _load_metadata(self, cfg_dir):
        # stub: pretend to read metadata.json
        meta_file = cfg_dir / "metadata.json"
        if cfg_dir.exists():
            print(f"[Stub] would load metadata from {meta_file}")
        else:
            print(f"[Stub] metadata directory {cfg_dir} not found, skipping")

    # --------------------------------------------------------------------------
    # Subcomponent: compute horizons
    # --------------------------------------------------------------------------
    def _load_horizons(self):
        # stub delta‐indices
        # video: use last 3 frames every 2 steps → [-4, -2, 0]
        self.video_delta = [-4, -2, 0]
        self.video_horizon = len(self.video_delta)
        # state: last 2 timesteps → [-1, 0]
        self.state_delta = [-1, 0]
        self.state_horizon = len(self.state_delta)
        print(f"[Stub] horizons set: video {self.video_delta}, state {self.state_delta}")

    # --------------------------------------------------------------------------
    # Transforms (no‐ops in stub)
    # --------------------------------------------------------------------------
    def apply_transforms(self, obs):
        # identity
        return obs

    def unapply_transforms(self, action):
        # identity
        return action

    # --------------------------------------------------------------------------
    # Public API
    # --------------------------------------------------------------------------
    def get_modality_config(self):
        return self._modality_config

    def get_action(self, observations):
        """
        obs → action: preserve batching semantics, then random‐sample.
        """
        batched = self._is_batched(observations)
        if not batched:
            observations = _unsqueeze_dict(observations)

        # apply stub transforms
        norm_in = self.apply_transforms(observations)

        # stub “inference”
        norm_act = self._forward_stub(norm_in)

        # unapply stub transforms + restore unbatched if needed
        raw_act = self.unapply_transforms({"action": norm_act})
        if not batched:
            raw_act = _squeeze_dict(raw_act)
        return raw_act

    # --------------------------------------------------------------------------
    # Stub forward: random action generator
    # --------------------------------------------------------------------------
    def _forward_stub(self, norm_in):
        """
        Build a random tensor‐like structure matching expected shape:
        - if batched: assume B×A dims
        - here: return a list of floats
        """
        # detect batch size from one modality
        first = next(iter(norm_in.values()))
        if isinstance(first, list) and isinstance(first[0], list):
            # batched: first dim = len(first)
            B = len(first)
        else:
            B = 1

        # for each batch produce a fixed‐size action vector, e.g. length=16
        out = []
        for _ in range(B):
            # each action is e.g. 16 floats in [0,1)
            vec = [random.random() for __ in range(16)]
            out.append(vec)
        return out

    # --------------------------------------------------------------------------
    # Helper: batched check
    # --------------------------------------------------------------------------
    def _is_batched(self, obs):
        # if any value is a list-of-lists, assume batched
        for v in obs.values():
            if isinstance(v, list) and v and isinstance(v[0], list):
                return True
        return False


# ==============================================================================
# Helper functions for batching (pure-Python)
# ==============================================================================

def _unsqueeze_dict(data):
    """
    Wrap every leaf in a singleton list for batch dimension.
    """
    out = {}
    for k, v in data.items():
        if isinstance(v, list):
            out[k] = [v]            # list-of-lists
        else:
            out[k] = [v]            # wrap scalars
    return out

def _squeeze_dict(data):
    """
    Remove the batch list‐wrapper.
    """
    out = {}
    for k, v in data.items():
        if isinstance(v, list) and len(v) == 1:
            out[k] = v[0]
        else:
            out[k] = v
    return out

# ==============================================================================
# End of control‐group stub
# ==============================================================================
