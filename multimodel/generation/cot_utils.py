"""Unified CoT (chain-of-thought) utilities for different model formats.

Supports two formats:
1. "think" format: <think>...[reasoning]...</think>[answer]
2. "channel" format: <|channel|>analysis<|message|>[reasoning]<|end|><|start|>assistant<|channel|>final<|message|>[answer]
"""

import re
from typing import Tuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..config.model_registry import ModelConfig


def extract_cot_and_answer(
    response: str, model_config: "ModelConfig"
) -> Tuple[str, str, bool]:
    """Extract CoT and final answer from response.

    Args:
        response: The model's response text
        model_config: Model configuration with format info

    Returns:
        Tuple of (cot_text, final_answer, has_cot_marker)
    """
    if model_config.cot_format == "channel":
        return _extract_channel_format(response, model_config)
    else:
        return _extract_think_format(response, model_config)


def _extract_channel_format(
    response: str, config: "ModelConfig"
) -> Tuple[str, str, bool]:
    """Extract CoT from channel-based format (e.g., GPT-OSS-20B).

    Format: <|channel|>analysis<|message|>[cot]<|end|><|start|>assistant<|channel|>final<|message|>[answer]
    
    Note: vLLM may decode special tokens as plain text, so we also handle:
    - "analysis..." at start (decoded from <|channel|>analysis<|message|>)
    - "assistantfinal" or "assistant" + "final" as answer separator
    """
    answer_marker = config.answer_start_token
    cot_start = config.cot_start_token

    if answer_marker and answer_marker in response:
        parts = response.split(answer_marker, 1)
        cot = parts[0]

        if cot_start and cot.startswith(cot_start):
            cot = cot[len(cot_start):]

        answer = parts[1].strip() if len(parts) > 1 else ""

        for eos in ["<|return|>", "<|end|>", "<|endoftext|>"]:
            if answer.endswith(eos):
                answer = answer[:-len(eos)].strip()

        return cot.strip(), answer.strip(), True

    if cot_start and cot_start in response:
        parts = response.split(cot_start, 1)
        cot = parts[1] if len(parts) > 1 else response
        return cot.strip(), "", True

    decoded_separators = [
        "assistantfinal",
        "assistant<|channel|>final<|message|>",
        "<|end|><|start|>assistant",
    ]
    
    for sep in decoded_separators:
        if sep.lower() in response.lower():
            idx = response.lower().find(sep.lower())
            cot = response[:idx]
            answer = response[idx + len(sep):]
            
            if cot.lower().startswith("analysis"):
                cot = cot[8:].lstrip()
            
            for eos in ["<|return|>", "<|end|>", "<|endoftext|>"]:
                if answer.endswith(eos):
                    answer = answer[:-len(eos)]
            
            return cot.strip(), answer.strip(), True

    if response.lower().startswith("analysis"):
        cot = response[8:].lstrip()
        
        answer_patterns = [
            r"\*\*([A-D])\.\s",
            r"(?:The\s+)?(?:correct\s+)?(?:answer|choice)\s+is[:\s]+\*?\*?([A-D])",
            r"(?:So|Thus),?\s+(?:the\s+)?(?:correct\s+)?(?:answer|choice)\s+is[:\s]+\*?\*?([A-D])",
        ]
        
        for pattern in answer_patterns:
            matches = list(re.finditer(pattern, response, re.IGNORECASE))
            if matches:
                last_match = matches[-1]
                answer = response[last_match.start():].strip()
                return cot, answer, True
        
        return cot, "", True

    return response, "", False


def _extract_think_format(
    response: str, config: "ModelConfig"
) -> Tuple[str, str, bool]:
    """Extract CoT from think tag format (e.g., DeepSeek, OLMo).

    Format: <think>[cot]</think>[answer]
    """
    # Check for hardcoded </think> first (highest priority)
    if "</think>" in response:
        parts = response.split("</think>", 1)
        cot = parts[0].replace("<think>", "").strip()
        answer = parts[1].strip() if len(parts) > 1 else ""
        return cot, answer, True

    # Fall back to config's cot_end_token
    cot_token = config.cot_end_token
    if cot_token and cot_token in response:
        parts = response.split(cot_token, 1)
        cot = parts[0].strip()
        answer = parts[1].strip() if len(parts) > 1 else ""
        return cot, answer, True

    # Final fallback: try common patterns
    patterns = [
        r"(?:Therefore|Thus|So),?\s+(?:the\s+)?answer\s+is",
        r"(?:My\s+)?(?:final\s+)?answer:?\s*",
        r"The\s+correct\s+answer\s+is",
    ]
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            cot = response[:match.start()].strip()
            answer = response[match.start():].strip()
            return cot, answer, False

    return response, "", False


def get_cot_prefix(model_config: "ModelConfig") -> str:
    """Get the prefix to prepend to CoT content.

    Args:
        model_config: Model configuration

    Returns:
        The CoT prefix string (e.g., "<think>\n" or "<|channel|>analysis<|message|>")
    """
    if model_config.cot_format == "channel":
        return model_config.cot_start_token or ""
    else:
        return "<think>\n"


def get_answer_probe_suffix(model_config: "ModelConfig") -> str:
    """Get the suffix for probing the model to produce an answer.

    Args:
        model_config: Model configuration

    Returns:
        The probe suffix string
    """
    if model_config.cot_format == "channel":
        answer_start = model_config.answer_start_token or ""
        return f"{answer_start}The correct answer is"
    else:
        cot_end = model_config.cot_end_token or "</think>"
        return f"{cot_end}\n\nThe correct answer is"


def build_probe_prompt(
    base_prompt: str, cot_prefix: str, model_config: "ModelConfig"
) -> str:
    """Build a complete probe prompt with partial CoT.

    Args:
        base_prompt: The base prompt before CoT
        cot_prefix: The partial CoT content to include
        model_config: Model configuration

    Returns:
        Complete probe prompt
    """
    cot_start = get_cot_prefix(model_config)
    suffix = get_answer_probe_suffix(model_config)
    return base_prompt + cot_start + cot_prefix + suffix


def build_full_text_for_extraction(
    prompt: str, response: str, model_config: "ModelConfig"
) -> str:
    """Build the full text for activation extraction.

    Args:
        prompt: The original prompt
        response: The model's response (CoT portion)
        model_config: Model configuration

    Returns:
        Full text for tokenization and extraction
    """
    cot_prefix = get_cot_prefix(model_config)
    return prompt + cot_prefix + response


def get_cot_prefix_length(model_config: "ModelConfig") -> int:
    """Get the character length of the CoT prefix.

    Args:
        model_config: Model configuration

    Returns:
        Length of the CoT prefix in characters
    """
    return len(get_cot_prefix(model_config))
