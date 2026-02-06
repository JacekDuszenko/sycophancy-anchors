"""Configuration modules for multi-model experiments."""

from .model_registry import MODEL_CONFIGS, get_model_config
from .experiment_config import ExperimentConfig

__all__ = ["MODEL_CONFIGS", "get_model_config", "ExperimentConfig"]
