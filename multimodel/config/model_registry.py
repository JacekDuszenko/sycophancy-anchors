"""Model configurations for multi-model sycophancy experiments."""

from typing import Optional
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for a single model."""
    name: str
    huggingface_path: str
    cot_end_token: Optional[str]
    quantization: Optional[str]
    max_new_tokens: int = 4096
    max_tokens: int = 4096  # Alias for compatibility
    temperature: float = 0.6
    top_p: float = 0.95
    trust_remote_code: bool = True
    max_model_len: int = 8192  # Context length for vLLM
    # Channel-based CoT format support (for models like GPT-OSS-20B)
    cot_format: str = "think"  # "think" or "channel"
    cot_start_token: Optional[str] = None  # Start token for CoT (channel format)
    answer_start_token: Optional[str] = None  # Token marking answer start (channel format)


MODEL_CONFIGS = {
    "deepseek-llama-8b": ModelConfig(
        name="deepseek-llama-8b",
        huggingface_path="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        cot_end_token="</think>",
        quantization=None,
    ),
    "deepseek-qwen-1.5b": ModelConfig(
        name="deepseek-qwen-1.5b",
        huggingface_path="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        cot_end_token="</think>",
        quantization=None,
    ),
    "deepseek-qwen-7b": ModelConfig(
        name="deepseek-qwen-7b",
        huggingface_path="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        cot_end_token="</think>",
        quantization=None,
    ),
    "olmo-7b-think": ModelConfig(
        name="olmo-7b-think",
        huggingface_path="allenai/OLMo-3-7B-Think",
        cot_end_token="</think>",  # Common reasoning token
        quantization=None,
    ),
    "falcon-h1r-7b": ModelConfig(
        name="falcon-h1r-7b",
        huggingface_path="tiiuae/Falcon-H1R-7B",
        cot_end_token="</think>",
        quantization=None,
    ),
    # GPT-OSS-20B: MoE model with MXFP4 quantization
    # Requires: vLLM >= 0.10.2 (or special build for A100)
    # H100: Use --async-scheduling for best performance
    # A100: May need vllm==0.10.1+gptoss from https://wheels.vllm.ai/gpt-oss/
    "gpt-oss-20b": ModelConfig(
        name="gpt-oss-20b",
        huggingface_path="openai/gpt-oss-20b",
        cot_end_token=None,
        quantization="mxfp4",
        cot_format="channel",
        cot_start_token="<|channel|>analysis<|message|>",
        answer_start_token="<|end|><|start|>assistant<|channel|>final<|message|>",
        temperature=1.2,
        top_p=0.99,
    ),
}

# Order of models to run (deepseek-llama-8b excluded - already have results)
MODEL_ORDER = [
    "deepseek-qwen-1.5b",
    "deepseek-qwen-7b",
    "olmo-7b-think",
    "falcon-h1r-7b",
]


def get_model_config(model_name: str) -> ModelConfig:
    """Get model configuration by name."""
    if model_name not in MODEL_CONFIGS:
        available = ", ".join(MODEL_CONFIGS.keys())
        raise ValueError(f"Unknown model: {model_name}. Available: {available}")
    return MODEL_CONFIGS[model_name]


def update_cot_token(model_name: str, cot_token: str) -> None:
    """Update the CoT end token for a model after discovery."""
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}")
    # Create a new config with the updated token
    old_config = MODEL_CONFIGS[model_name]
    MODEL_CONFIGS[model_name] = ModelConfig(
        name=old_config.name,
        huggingface_path=old_config.huggingface_path,
        cot_end_token=cot_token,
        quantization=old_config.quantization,
        max_new_tokens=old_config.max_new_tokens,
        temperature=old_config.temperature,
        top_p=old_config.top_p,
        trust_remote_code=old_config.trust_remote_code,
        cot_format=old_config.cot_format,
        cot_start_token=old_config.cot_start_token,
        answer_start_token=old_config.answer_start_token,
    )
