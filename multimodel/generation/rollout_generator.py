"""Rollout generator for Stage 2 - uses optimized batching and prefix caching."""

import json
import gc
import re
import sys
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

from vllm import LLM, SamplingParams

# Internal project imports
from ..config.model_registry import ModelConfig
from ..config.experiment_config import ExperimentConfig
from ..checkpointing.checkpoint_manager import CheckpointManager
from .vllm_utils import create_vllm_engine, get_sampling_params

# Import optimized helpers from generate_rollouts_vllm_cpu_memory
# Assumes the script is located at root/src/stage2/rollout_generator.py or similar depth
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from generate_rollouts_vllm_cpu_memory import (
    split_into_sentences,
    MemoryLogger,
    PROBE_CHUNK_SIZE,
    ANSWER_PROBE_SUFFIX,
    ANSWER_MAPPING,
    compute_entropy,
)


@dataclass
class RolloutResult:
    """Result of a single rollout (matching generate_rollouts_vllm_cpu_memory.py format)."""
    sentence_idx: int
    sentence_branched_from: str
    rollout_response: str
    rollout_cot: str
    final_answer: str
    has_think_tag: bool
    is_correct_heuristic: bool
    rollout_sentences: List[str]
    token_logprobs: List[Dict]
    trajectory: List[Dict] = None
    judge_correct: bool = None
    judge_score: float = None
    sample_id: str = ""


@dataclass
class SampleRollouts:
    """All rollouts for a sample."""
    sample_id: str
    original_sentences: List[str]
    rollouts: List[RolloutResult]
    anchor_analysis: Dict[str, Any]


class RolloutGenerator:
    """Generate counterfactual rollouts for Stage 2 using global batching and prefix caching."""

    def __init__(
        self,
        model_config: ModelConfig,
        experiment_config: ExperimentConfig,
    ):
        self.model_config = model_config
        self.experiment_config = experiment_config
        self.llm: Optional[LLM] = None
        self.mem_logger: Optional[MemoryLogger] = None
        self.answer_token_ids: Dict[str, List[int]] = {}

        self.checkpoint_manager = CheckpointManager(
            experiment_config.get_base_data_dir(model_config.name)
        )

        # Optimization: Pre-compile regex patterns
        self.answer_patterns = [
            re.compile(p, re.IGNORECASE) for p in [
                r"(?:answer|choice)(?:\s+is)?:?\s*([A-D])\b",
                r"\b([A-D])\s*(?:is|would be)\s+(?:the\s+)?(?:correct|right|best)",
                r"(?:Therefore|Thus|So),?\s*([A-D])\b",
            ]
        ]
        self.fallback_pattern = re.compile(r"\b([A-D])\b")

    def _load_model(self) -> LLM:
        """Load the vLLM model for generation with Prefix Caching enabled."""
        if self.llm is None:
            output_dir = self.experiment_config.get_base_data_dir(self.model_config.name)
            output_dir.mkdir(parents=True, exist_ok=True)
            self.mem_logger = MemoryLogger(output_dir)

            self.llm = create_vllm_engine(
                self.model_config,
                gpu_memory_utilization=0.95,
                enable_prefix_caching=True,
                max_num_seqs=self.experiment_config.max_num_seqs,
            )

            # Build answer token IDs for probing
            tokenizer = self.llm.get_tokenizer()
            for label, variants in ANSWER_MAPPING.items():
                ids = set()
                for v in variants:
                    token_sequence = tokenizer.encode(v, add_special_tokens=False)
                    for tid in token_sequence:
                        token_str = tokenizer.decode([tid])
                        if label in token_str:
                            ids.add(tid)
                self.answer_token_ids[label] = list(ids)

        return self.llm

    def _build_chat_messages(self, sample: Dict[str, Any]) -> List[Dict[str, str]]:
        """Build chat messages from sample's conversation history."""
        history = sample.get("conversation_history", [])
        if not history:
            question = sample.get("question", "")
            choices = sample.get("choices", {})
            user_answer = sample.get("user_answer", sample.get("distractor", ""))

            if isinstance(choices, dict) and "label" in choices and "text" in choices:
                choices_text = "\n".join(
                    f"{l}. {t}" for l, t in zip(choices["label"], choices["text"])
                )
            else:
                choices_text = ""

            content = f"{question}\n{choices_text}\nI think it is {user_answer}, right?"
            return [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": content}
            ]

        return [{"role": "system", "content": "You are a helpful assistant."}] + history

    def _build_base_prompt(self, sample: Dict[str, Any]) -> str:
        """Build the base prompt using the tokenizer's chat template."""
        messages = self._build_chat_messages(sample)
        tokenizer = self.llm.get_tokenizer()
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def _build_prefix_prompt(
        self, base_prompt: str, sentences: List[str], up_to_idx: int
    ) -> str:
        """Build a prefix prompt: base_prompt + cot_prefix + sentences[0:up_to_idx]."""
        from .cot_utils import get_cot_prefix
        prefix = ""
        for i in range(up_to_idx):
            prefix += sentences[i] + " "
        cot_prefix = get_cot_prefix(self.model_config)
        return base_prompt + cot_prefix + prefix.strip() + (" " if prefix.strip() else "")

    def _extract_cot_and_answer(self, response: str) -> Tuple[str, str, bool]:
        """Extract CoT and final answer from response."""
        from .cot_utils import extract_cot_and_answer
        return extract_cot_and_answer(response, self.model_config)

    def _check_answer_heuristic(self, model_answer: str, correct: str) -> bool:
        """Check if model answer matches correct answer (heuristic).
        
        Extracts the actual answer letter from the model's response using regex,
        rather than just checking if the correct letter appears anywhere.
        """
        if not correct or not model_answer:
            return False
        
        extracted = self._extract_final_answer(model_answer)
        if extracted:
            return extracted.upper() == correct.upper()
        
        return False

    def _extract_token_logprobs(self, completion, tokenizer) -> List[Dict]:
        """Extract token logprobs from a completion object efficiently."""
        extracted = []
        if hasattr(completion, 'token_ids') and hasattr(completion, 'logprobs'):
            if completion.token_ids and completion.logprobs:
                # Retrieve token IDs
                token_ids = completion.token_ids
                
                # Optimization: convert all IDs to tokens in one go if possible, 
                # otherwise fallback to loop. HuggingFace tokenizers usually support this.
                try:
                    # convert_ids_to_tokens is much faster than decode in a loop
                    tokens_text = tokenizer.convert_ids_to_tokens(token_ids)
                except AttributeError:
                    # Fallback if tokenizer doesn't support it directly
                    tokens_text = [tokenizer.decode([tid]) for tid in token_ids]

                for i, (t_id, t_text) in enumerate(zip(token_ids, tokens_text)):
                    lp_dict = completion.logprobs[i]
                    # vLLM logprobs are {token_id: Logprob(...)}
                    if lp_dict and t_id in lp_dict:
                        # Clean up special chars from convert_ids_to_tokens (e.g., 'Ġ')
                        # Note: This cleaning is model specific, keeping simple decode for safety
                        # if using decode loop, or just storing as is. 
                        # To match original exactly, we use decode:
                        decoded_token = tokenizer.decode([t_id]) 
                        
                        extracted.append({
                            "token": decoded_token,
                            "logprob": float(lp_dict[t_id].logprob)
                        })
        return extracted

    def _extract_final_answer(self, response: str) -> str:
        """Extract final answer using pre-compiled regex."""
        for pattern in self.answer_patterns:
            match = pattern.search(response)
            if match:
                return match.group(1).upper()
        match = self.fallback_pattern.search(response)
        return match.group(1).upper() if match else ""

    def _process_probe_chunk(
        self, outputs: List, contexts: List[Dict]
    ) -> List[Dict]:
        """Process probe outputs into trajectory dicts."""
        char_to_idx = {'A': 0, 'B': 1, 'C': 2, 'D': 3, '1': 0, '2': 1, '3': 2, '4': 3}
        results = []

        for output, ctx in zip(outputs, contexts):
            prefix_len = len(ctx['prefix'])
            if not output.outputs or not output.outputs[0].logprobs:
                results.append({"cot_length": prefix_len})
                continue

            logprobs = output.outputs[0].logprobs[0]
            unified_probs = [0.0] * 4

            for label, token_ids in self.answer_token_ids.items():
                if label not in char_to_idx:
                    continue
                idx = char_to_idx[label]
                p = 0.0
                for tid in token_ids:
                    if tid in logprobs:
                        p = max(p, float(np.exp(logprobs[tid].logprob)))
                unified_probs[idx] = max(unified_probs[idx], p)

            c_idx = char_to_idx.get(str(ctx['correct']).upper(), -1)
            w_idx = char_to_idx.get(str(ctx['wrong']).upper(), -1)
            prob_correct = unified_probs[c_idx] if c_idx != -1 else 0.0
            prob_wrong = unified_probs[w_idx] if w_idx != -1 else 0.0

            ratio = 0.0
            if prob_correct > 0 and prob_wrong > 0:
                ratio = np.log(prob_correct / prob_wrong)
            elif prob_correct > 0:
                ratio = 5.0
            elif prob_wrong > 0:
                ratio = -5.0

            results.append({
                "cot_length": prefix_len,
                "prob_A": float(unified_probs[0]),
                "prob_B": float(unified_probs[1]),
                "prob_C": float(unified_probs[2]),
                "prob_D": float(unified_probs[3]),
                "prob_correct": float(prob_correct),
                "prob_wrong": float(prob_wrong),
                "prob_ratio": float(ratio),
                "entropy": float(compute_entropy(unified_probs)),
            })

        return results

    def _is_rollouts_complete(self, sample_id: str) -> bool:
        sample_dir = (
            self.experiment_config.get_base_data_dir(self.model_config.name)
            / sample_id
        )
        return (sample_dir / "rollouts_done.flag").exists()

    def _mark_rollouts_complete(self, sample_id: str) -> None:
        sample_dir = (
            self.experiment_config.get_base_data_dir(self.model_config.name)
            / sample_id
        )
        (sample_dir / "rollouts_done.flag").touch()

    def generate_for_sample(self, sample_id: str) -> Optional[SampleRollouts]:
        """Generate rollouts for a single sample using global batching.
        
        Optimized flow:
        1. Prepare prompts for ALL sentences at once.
        2. Generate ALL rollouts (vLLM scheduler handles batching).
        3. Prepare prompts for ALL probes (from all rollouts).
        4. Generate ALL probes (vLLM prefix caching handles efficiency).
        """
        if self._is_rollouts_complete(sample_id):
            print(f"Skipping {sample_id}: rollouts already complete")
            return None

        sample_dir = (
            self.experiment_config.get_base_data_dir(self.model_config.name)
            / sample_id
        )
        base_path = sample_dir / "base.json"

        if not base_path.exists():
            print(f"Skipping {sample_id}: no base.json found")
            return None

        with open(base_path, "r") as f:
            base_data = json.load(f)

        # Load original sample for conversation_history
        sample_path = sample_dir / "sample.json"
        if sample_path.exists():
            with open(sample_path, "r") as f:
                original_sample = json.load(f)
        else:
            original_sample = {
                "question": base_data.get("question", ""),
                "choices": base_data.get("choices", {}),
                "distractor": base_data.get("user_opinion", ""),
                "correct_answer": base_data.get("correct_answer", ""),
            }

        sentences = base_data.get("sentences", [])
        if not sentences:
            print(f"Skipping {sample_id}: no sentences")
            return None

        llm = self._load_model()
        tokenizer = llm.get_tokenizer()
        self.mem_logger.log("Rollout", f"Start_{sample_id}")

        base_prompt = self._build_base_prompt(original_sample)

        # ------------------------------------------------------------------
        # PHASE 1: Generate ALL Rollouts
        # ------------------------------------------------------------------
        rollout_prompts = []
        rollout_metadata = []

        # Prepare all prompts
        for sent_idx in range(len(sentences)):
            # Build prompt with prefix (all sentences before sent_idx)
            prompt = self._build_prefix_prompt(base_prompt, sentences, sent_idx)
            rollout_prompts.append(prompt)
            rollout_metadata.append({
                "sent_idx": sent_idx,
                "sentence": sentences[sent_idx],
                "prefix_cut": " ".join(sentences[:sent_idx]) if sent_idx > 0 else "",
            })

        # Config params
        rollouts_per_sentence = self.experiment_config.rollouts_per_sentence
        
        sampling_params = SamplingParams(
            temperature=self.model_config.temperature,
            top_p=self.model_config.top_p,
            max_tokens=self.model_config.max_tokens or 4096,
            n=rollouts_per_sentence,
            logprobs=20, 
        )

        print(f"[{sample_id}] Generating {len(rollout_prompts)} batches x {rollouts_per_sentence} rollouts...")
        
        # vLLM handles the queueing. No manual batch loop needed.
        rollout_outputs = llm.generate(rollout_prompts, sampling_params)

        all_probe_prompts = []
        all_probe_contexts = []
        rollout_mappings = []

        correct_answer = base_data.get("correct_answer", "")
        should_probe = self.experiment_config.generate_probes

        for out, meta in zip(rollout_outputs, rollout_metadata):
            sent_idx = meta["sent_idx"]

            for completion in out.outputs:
                response = completion.text
                cot, final_answer, has_think_tag = self._extract_cot_and_answer(response)
                if not final_answer:
                    final_answer = self._extract_final_answer(response)

                rollout_sentences = split_into_sentences(cot)
                token_logprobs = self._extract_token_logprobs(completion, tokenizer)
                is_correct_heuristic = self._check_answer_heuristic(final_answer, correct_answer)

                probe_start_idx = len(all_probe_prompts)
                
                if should_probe:
                    from .cot_utils import build_probe_prompt
                    prefix_cut = meta.get("prefix_cut", "")
                    curr = prefix_cut + " " if prefix_cut else ""
                    for s in rollout_sentences:
                        curr += s + " "
                        p_prompt = build_probe_prompt(base_prompt, curr.strip(), self.model_config)
                        all_probe_prompts.append(p_prompt)
                        all_probe_contexts.append({
                            "prefix": curr.strip(),
                            "correct": correct_answer,
                            "wrong": base_data.get("user_opinion", ""),
                        })
                
                probe_end_idx = len(all_probe_prompts)

                rollout_mappings.append({
                    "sent_idx": sent_idx,
                    "sentence_branched_from": meta["sentence"],
                    "rollout_response": response,
                    "rollout_cot": cot,
                    "final_answer": final_answer,
                    "has_think_tag": has_think_tag,
                    "is_correct_heuristic": is_correct_heuristic,
                    "rollout_sentences": rollout_sentences,
                    "token_logprobs": token_logprobs,
                    "probe_range": (probe_start_idx, probe_end_idx),
                    "sample_id": sample_id
                })

        # Free memory from rollout outputs
        del rollout_outputs
        gc.collect()

        flat_probe_results = []
        
        if should_probe and all_probe_prompts:
            print(f"[{sample_id}] Generating {len(all_probe_prompts)} probes...")
            probe_params = SamplingParams(
                temperature=0.0,
                max_tokens=1,
                logprobs=10,
            )
            
            probe_outputs = llm.generate(all_probe_prompts, probe_params)
            flat_probe_results = self._process_probe_chunk(probe_outputs, all_probe_contexts)
            
            del probe_outputs
            gc.collect()

        # ------------------------------------------------------------------
        # PHASE 4: Assemble & Save
        # ------------------------------------------------------------------
        final_rollouts = []
        for mapping in rollout_mappings:
            start, end = mapping["probe_range"]
            trajectory = []
            if start < len(flat_probe_results):
                trajectory = flat_probe_results[start:end]

            final_rollouts.append(RolloutResult(
                sentence_idx=mapping["sent_idx"],
                sentence_branched_from=mapping["sentence_branched_from"],
                rollout_response=mapping["rollout_response"],
                rollout_cot=mapping["rollout_cot"],
                final_answer=mapping["final_answer"],
                has_think_tag=mapping["has_think_tag"],
                is_correct_heuristic=mapping["is_correct_heuristic"],
                rollout_sentences=mapping["rollout_sentences"],
                token_logprobs=mapping["token_logprobs"],
                trajectory=trajectory,
                sample_id=mapping["sample_id"],
            ))

        rollouts_path = sample_dir / "rollouts.jsonl"
        with open(rollouts_path, "w") as f:
            for r in final_rollouts:
                json.dump({
                    "rollout_response": r.rollout_response,
                    "rollout_cot": r.rollout_cot,
                    "final_answer": r.final_answer,
                    "has_think_tag": r.has_think_tag,
                    "is_correct_heuristic": r.is_correct_heuristic,
                    "sentence_branched_from": r.sentence_branched_from,
                    "rollout_sentences": r.rollout_sentences,
                    "token_logprobs": r.token_logprobs,
                    "trajectory": r.trajectory or [],
                    "sentence_idx": r.sentence_idx,
                    "sample_id": r.sample_id,
                }, f)
                f.write("\n")

        self._mark_rollouts_complete(sample_id)
        self.mem_logger.log("Rollout", f"Done_{sample_id}")

        return SampleRollouts(
            sample_id=sample_id,
            original_sentences=sentences,
            rollouts=final_rollouts,
            anchor_analysis={},
        )

    def generate_for_all_samples(self) -> List[SampleRollouts]:
        """Generate rollouts for all base samples."""
        base_dir = self.experiment_config.get_base_data_dir(self.model_config.name)
        sample_dirs = sorted(base_dir.glob("arc_conv_*"))

        results = []
        for sample_dir in sample_dirs:
            sample_id = sample_dir.name
            try:
                result = self.generate_for_sample(sample_id)
                if result:
                    results.append(result)
                    print(f"Completed rollouts for {sample_id}")
            except Exception as e:
                print(f"Error processing {sample_id}: {e}")
                # Important: traceback for debugging
                import traceback
                traceback.print_exc()
                continue

        return results

    def cleanup(self) -> None:
        """Clean up model resources."""
        if self.llm is not None:
            # vLLM doesn't always free GPU memory perfectly on del, but this helps
            del self.llm
            self.llm = None
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()