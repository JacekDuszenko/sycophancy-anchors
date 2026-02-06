"""Model utilities for activation extraction with transformers."""

from typing import Dict, Any, List, Tuple, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from ..config.model_registry import ModelConfig

FALCON_MODELS = {"falcon-h1r-7b"}


def get_model_config_info(model_config: ModelConfig) -> Dict[str, Any]:
    """Get model configuration info without loading the full model.

    Args:
        model_config: Model configuration

    Returns:
        Dictionary with model info
    """
    config = AutoConfig.from_pretrained(
        model_config.huggingface_path,
        trust_remote_code=model_config.trust_remote_code,
    )

    return {
        "num_hidden_layers": getattr(config, "num_hidden_layers", None),
        "hidden_size": getattr(config, "hidden_size", None),
        "vocab_size": getattr(config, "vocab_size", None),
        "model_type": getattr(config, "model_type", None),
    }


def get_target_layers(
    num_layers: int,
    percentages: Optional[List[float]] = None,
) -> List[int]:
    """Calculate target layer indices based on percentages.

    Args:
        num_layers: Total number of layers in model
        percentages: List of percentages (0-1) for layer selection

    Returns:
        List of layer indices
    """
    if percentages is None:
        percentages = [0.35, 0.5, 0.65, 0.80]

    return [int(p * (num_layers - 1)) for p in percentages]


def load_transformers_model(
    model_config: ModelConfig,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load model and tokenizer using transformers for activation extraction.

    Args:
        model_config: Model configuration
        device: Device to load model on
        dtype: Data type for model weights

    Returns:
        Tuple of (model, tokenizer)
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.huggingface_path,
        trust_remote_code=model_config.trust_remote_code,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    effective_dtype = dtype
    if model_config.name in FALCON_MODELS:
        effective_dtype = torch.bfloat16
        print(f"Using bfloat16 for {model_config.name} (hybrid Mamba architecture)")

    load_kwargs = {
        "trust_remote_code": model_config.trust_remote_code,
        "torch_dtype": effective_dtype,
        "device_map": "auto",
    }

    # For mxfp4 quantized models, we may need special handling
    if model_config.quantization == "mxfp4":
        # Load with bitsandbytes 4-bit if available
        try:
            from transformers import BitsAndBytesConfig

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
            )
            load_kwargs["quantization_config"] = quantization_config
        except ImportError:
            print("Warning: bitsandbytes not available, loading without quantization")

    model = AutoModelForCausalLM.from_pretrained(
        model_config.huggingface_path,
        **load_kwargs,
    )

    model.eval()

    return model, tokenizer


def get_hidden_states_hook(
    layer_idx: int,
    storage: Dict[int, torch.Tensor],
):
    """Create a hook function to capture hidden states at a specific layer.

    Args:
        layer_idx: Layer index to capture
        storage: Dictionary to store captured states

    Returns:
        Hook function
    """
    def hook(module, input, output):
        # Output is usually a tuple, first element is hidden states
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output
        storage[layer_idx] = hidden.detach().cpu()

    return hook


def find_layer_modules(model: AutoModelForCausalLM) -> List[torch.nn.Module]:
    """Find the transformer layer modules in a model.

    Args:
        model: The loaded model

    Returns:
        List of layer modules
    """
    # Common attribute names for transformer layers
    layer_attrs = [
        "model.layers",
        "transformer.h",
        "gpt_neox.layers",
        "model.decoder.layers",
    ]

    for attr_path in layer_attrs:
        try:
            obj = model
            for attr in attr_path.split("."):
                obj = getattr(obj, attr)
            return list(obj)
        except AttributeError:
            continue

    # Fallback: try to find any ModuleList that looks like layers
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.ModuleList) and len(module) > 10:
            return list(module)

    raise ValueError("Could not find transformer layers in model")
