"""Discover chain-of-thought end tokens for models."""

import re
from typing import Optional, List
from collections import Counter

from vllm import LLM, SamplingParams

from ..config.model_registry import ModelConfig, update_cot_token
from .vllm_utils import create_vllm_engine


# Common CoT end tokens to search for
COT_END_PATTERNS = [
    r"</think>",
    r"<\|end_thinking\|>",
    r"</reasoning>",
    r"</thought>",
    r"<\|endofthought\|>",
    r"\[/THINK\]",
    r"---\s*Answer:",
    r"\*\*Answer:\*\*",
    r"Therefore,\s+the\s+answer\s+is",
]

# Test prompts for discovery
TEST_PROMPTS = [
    "What is 15 + 27? Think step by step.",
    "If a train travels 60 miles in 1 hour, how far will it travel in 2.5 hours?",
    "What comes next in the sequence: 2, 4, 8, 16, ?",
    "A farmer has 3 chickens and 2 cows. How many legs are there in total?",
    "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
]


class CoTTokenDiscovery:
    """Discover chain-of-thought end tokens for a model."""

    def __init__(
        self,
        model_config: ModelConfig,
        n_samples: int = 5,
    ):
        """Initialize CoT token discovery.

        Args:
            model_config: Model configuration
            n_samples: Number of test samples to generate
        """
        self.model_config = model_config
        self.n_samples = min(n_samples, len(TEST_PROMPTS))
        self.llm: Optional[LLM] = None

    def _load_model(self) -> LLM:
        """Load the model for discovery."""
        if self.llm is None:
            self.llm = create_vllm_engine(
                self.model_config,
                gpu_memory_utilization=0.95,
                max_num_seqs=256,
            )
        return self.llm

    def _generate_samples(self) -> List[str]:
        """Generate sample outputs for analysis.

        Returns:
            List of generated text samples
        """
        llm = self._load_model()

        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.95,
            max_tokens=10_000,
        )

        prompts = TEST_PROMPTS[: self.n_samples]
        outputs = llm.generate(prompts, sampling_params)

        return [output.outputs[0].text for output in outputs]

    def _detect_pattern(self, texts: List[str]) -> Optional[str]:
        """Detect CoT end pattern in generated texts.

        Args:
            texts: Generated text samples

        Returns:
            Detected pattern or None
        """
        pattern_counts: Counter = Counter()

        for text in texts:
            for pattern in COT_END_PATTERNS:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    # Use the actual matched string
                    pattern_counts[matches[0]] += 1

        if not pattern_counts:
            return None

        # Return the most common pattern if it appears in majority of samples
        most_common, count = pattern_counts.most_common(1)[0]
        if count >= len(texts) / 2:
            return most_common

        return None

    def discover(self) -> Optional[str]:
        """Run discovery to find CoT end token.

        Returns:
            Discovered token or None if not found
        """
        print(f"Discovering CoT token for {self.model_config.name}...")

        samples = self._generate_samples()

        # Print all generations for manual inspection
        print("\n" + "=" * 80)
        print("GENERATED SAMPLES FOR MANUAL INSPECTION")
        print("=" * 80)
        prompts = TEST_PROMPTS[: self.n_samples]
        for i, (prompt, sample) in enumerate(zip(prompts, samples)):
            print(f"\n{'─' * 80}")
            print(f"SAMPLE {i + 1}")
            print(f"{'─' * 80}")
            print(f"PROMPT: {prompt}")
            print(f"{'─' * 40}")
            print("GENERATION:")
            print(sample)
            print(f"{'─' * 80}")
        print("\n" + "=" * 80)

        token = self._detect_pattern(samples)

        if token:
            print(f"Auto-detected CoT end token: {token}")
            update_cot_token(self.model_config.name, token)
        else:
            print("No consistent CoT end token auto-detected. Please inspect samples above.")

        return token

    def cleanup(self) -> None:
        """Clean up model resources."""
        if self.llm is not None:
            del self.llm
            self.llm = None


def discover_cot_token(
    model_config: ModelConfig,
    n_samples: int = 5,
) -> Optional[str]:
    """Convenience function to discover CoT token.

    Args:
        model_config: Model configuration
        n_samples: Number of test samples

    Returns:
        Discovered token or None
    """
    discovery = CoTTokenDiscovery(model_config, n_samples)
    try:
        return discovery.discover()
    finally:
        discovery.cleanup()
