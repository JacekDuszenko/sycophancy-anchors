"""Extract window activations for experiments."""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from ..config.model_registry import ModelConfig
from ..config.experiment_config import ExperimentConfig
from .model_utils import (
    load_transformers_model,
    get_target_layers,
    get_model_config_info,
    find_layer_modules,
    get_hidden_states_hook,
)

FALCON_MODELS = {"falcon-h1r-7b"}


@dataclass
class AnchorInfo:
    sample_id: str
    anchor_idx: int
    sentence_text: str
    accuracy_drop: float
    is_sycophantic: bool
    anchor_end_position: int


@dataclass
class SentenceInfo:
    sample_id: str
    sentence_idx: int
    sentence_text: str
    prob_ratio: float
    prob_correct: float
    prob_wrong: float
    is_sycophantic: bool
    sentence_end_position: int


class ActivationExtractor:

    def __init__(
        self,
        model_config: ModelConfig,
        experiment_config: ExperimentConfig,
    ):
        self.model_config = model_config
        self.experiment_config = experiment_config
        self.model: Optional[AutoModelForCausalLM] = None
        self.tokenizer: Optional[AutoTokenizer] = None

        model_info = get_model_config_info(model_config)
        self.num_layers = model_info["num_hidden_layers"]
        self.hidden_size = model_info["hidden_size"]

        self.target_layers = get_target_layers(
            self.num_layers,
            experiment_config.layer_percentages,
        )
        
        self._is_falcon = model_config.name in FALCON_MODELS

    def _load_model(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        if self.model is None:
            self.model, self.tokenizer = load_transformers_model(self.model_config)
        return self.model, self.tokenizer

    def _find_sentence_end_position(
        self,
        tokenizer: AutoTokenizer,
        full_text: str,
        sentence: str,
        cot_start_offset: int,
    ) -> Optional[int]:
        sentence_start = full_text.find(sentence, cot_start_offset)
        if sentence_start == -1:
            return None

        sentence_end = sentence_start + len(sentence)
        prefix_with_sentence = full_text[:sentence_end]
        tokens = tokenizer.encode(prefix_with_sentence, add_special_tokens=True)

        return len(tokens) - 1

    def _extract_hidden_states_native(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        text: str,
    ) -> Dict[int, torch.Tensor]:
        inputs = tokenizer(text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.device)

        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)

        result = {}
        for layer_idx in self.target_layers:
            hs = outputs.hidden_states[layer_idx + 1]
            result[layer_idx] = hs[0].cpu().to(torch.float16)

        return result

    def _extract_hidden_states_hooks(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        text: str,
    ) -> Dict[int, torch.Tensor]:
        inputs = tokenizer(text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.device)

        storage = {}
        hooks = []
        layer_modules = find_layer_modules(model)

        for layer_idx in self.target_layers:
            hook = layer_modules[layer_idx].register_forward_hook(
                get_hidden_states_hook(layer_idx, storage)
            )
            hooks.append(hook)

        try:
            with torch.no_grad():
                model(input_ids, output_hidden_states=False)

            result = {}
            for layer_idx, hidden in storage.items():
                result[layer_idx] = hidden[0].cpu().to(torch.float16)

            return result

        finally:
            for hook in hooks:
                hook.remove()

    def _extract_all_hidden_states(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        text: str,
    ) -> Dict[int, torch.Tensor]:
        if self._is_falcon:
            return self._extract_hidden_states_native(model, tokenizer, text)
        return self._extract_hidden_states_hooks(model, tokenizer, text)

    def _collect_anchors_from_sample(
        self,
        sample_id: str,
        base_data: Dict[str, Any],
    ) -> List[AnchorInfo]:
        anchor_analysis = base_data.get("anchor_analysis", [])
        sentences = base_data.get("sentences", [])
        is_sycophantic = base_data.get("is_sycophantic", False)

        anchors = []
        for idx, entry in enumerate(anchor_analysis):
            sentence_text = entry.get("sentence_text", "")
            accuracy_drop = entry.get("accuracy_drop", 0.0)

            if not sentence_text:
                continue

            anchors.append(AnchorInfo(
                sample_id=sample_id,
                anchor_idx=idx,
                sentence_text=sentence_text,
                accuracy_drop=accuracy_drop,
                is_sycophantic=is_sycophantic,
                anchor_end_position=-1,
            ))

        return anchors

    def _collect_trajectory_from_sample(
        self,
        sample_id: str,
        base_data: Dict[str, Any],
    ) -> List[SentenceInfo]:
        """Collect sentence-level trajectory info for prob_ratio regression."""
        sentences = base_data.get("sentences", [])
        trajectory = base_data.get("trajectory", [])
        is_sycophantic = base_data.get("is_sycophantic", False)

        if len(sentences) != len(trajectory):
            return []

        sentence_infos = []
        for idx, (sentence, traj) in enumerate(zip(sentences, trajectory)):
            if not sentence:
                continue

            sentence_infos.append(SentenceInfo(
                sample_id=sample_id,
                sentence_idx=idx,
                sentence_text=sentence,
                prob_ratio=traj.get("prob_ratio", 0.0),
                prob_correct=traj.get("prob_correct", 0.0),
                prob_wrong=traj.get("prob_wrong", 0.0),
                is_sycophantic=is_sycophantic,
                sentence_end_position=-1,
            ))

        return sentence_infos

    def extract_for_sample(
        self,
        sample_id: str,
    ) -> Optional[Dict[str, Any]]:
        sample_dir = (
            self.experiment_config.get_base_data_dir(self.model_config.name)
            / sample_id
        )
        base_path = sample_dir / "base.json"

        if not base_path.exists():
            return None

        with open(base_path, "r") as f:
            base_data = json.load(f)

        anchors = self._collect_anchors_from_sample(sample_id, base_data)
        if not anchors:
            return None

        model, tokenizer = self._load_model()

        prompt = base_data.get("prompt", "")
        response = base_data.get("response", "")

        # Use model-specific CoT prefix
        from ..generation.cot_utils import get_cot_prefix, get_cot_prefix_length
        cot_prefix = get_cot_prefix(self.model_config)
        full_text = prompt + cot_prefix + response

        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
        cot_prefix_tokens = tokenizer.encode(cot_prefix, add_special_tokens=False)
        cot_start_offset = len(prompt) + get_cot_prefix_length(self.model_config)

        hidden_states = self._extract_all_hidden_states(model, tokenizer, full_text)
        seq_len = next(iter(hidden_states.values())).shape[0]

        window_size = self.experiment_config.window_size
        n_layers = len(self.target_layers)

        experiment_acts = []
        emergence_acts = []
        valid_anchors = []

        for anchor in anchors:
            anchor_end_pos = self._find_sentence_end_position(
                tokenizer, full_text, anchor.sentence_text, cot_start_offset
            )

            if anchor_end_pos is None or anchor_end_pos >= seq_len:
                continue

            anchor.anchor_end_position = anchor_end_pos

            exp_tensor = torch.zeros(n_layers, 1, self.hidden_size, dtype=torch.float16)
            emerg_tensor = torch.zeros(n_layers, window_size, self.hidden_size, dtype=torch.float16)

            for i, layer_idx in enumerate(sorted(self.target_layers)):
                layer_hidden = hidden_states[layer_idx]

                exp_tensor[i, 0, :] = layer_hidden[anchor_end_pos, :]

                start = max(0, anchor_end_pos - window_size + 1)
                end = anchor_end_pos + 1
                actual_window = layer_hidden[start:end, :]

                pad_left = window_size - actual_window.shape[0]
                if pad_left > 0:
                    emerg_tensor[i, pad_left:, :] = actual_window
                else:
                    emerg_tensor[i, :, :] = actual_window

            experiment_acts.append(exp_tensor)
            emergence_acts.append(emerg_tensor)
            valid_anchors.append(anchor)

        if not valid_anchors:
            return None

        return {
            "sample_id": sample_id,
            "anchors": valid_anchors,
            "experiment_activations": experiment_acts,
            "emergence_activations": emergence_acts,
        }

    def extract_all(self) -> Dict[str, Any]:
        base_dir = self.experiment_config.get_base_data_dir(self.model_config.name)
        output_dir = self.experiment_config.get_activations_dir(self.model_config.name)

        sample_dirs = sorted(base_dir.glob("arc_conv_*"))

        all_experiment_acts = []
        all_emergence_acts = []
        all_anchor_info = []

        for sample_dir in tqdm(sample_dirs, desc="Extracting activations"):
            sample_id = sample_dir.name

            if not (sample_dir / "rollouts_done.flag").exists():
                continue

            result = self.extract_for_sample(sample_id)
            if result is None:
                continue

            for i, anchor in enumerate(result["anchors"]):
                all_anchor_info.append({
                    "sample_id": anchor.sample_id,
                    "anchor_idx": anchor.anchor_idx,
                    "sentence_text": anchor.sentence_text,
                    "accuracy_drop": anchor.accuracy_drop,
                    "is_sycophantic": anchor.is_sycophantic,
                    "anchor_end_position": anchor.anchor_end_position,
                })
                all_experiment_acts.append(result["experiment_activations"][i])
                all_emergence_acts.append(result["emergence_activations"][i])

        if not all_experiment_acts:
            raise ValueError("No activations extracted!")

        experiment_tensor = torch.stack(all_experiment_acts)
        emergence_tensor = torch.stack(all_emergence_acts)

        is_sycophantic = torch.tensor(
            [a["is_sycophantic"] for a in all_anchor_info], 
            dtype=torch.bool
        )
        accuracy_drops = torch.tensor(
            [a["accuracy_drop"] for a in all_anchor_info],
            dtype=torch.float32
        )

        exp_path = output_dir / "experiments.safetensors"
        save_file({
            "activations": experiment_tensor,
            "is_sycophantic": is_sycophantic,
            "accuracy_drop": accuracy_drops,
        }, str(exp_path))

        emerg_path = output_dir / "emergence.safetensors"
        save_file({
            "activations": emergence_tensor,
            "is_sycophantic": is_sycophantic,
            "accuracy_drop": accuracy_drops,
        }, str(emerg_path))

        metadata = {
            "model": self.model_config.name,
            "num_anchors": len(all_anchor_info),
            "num_samples": len(set(a["sample_id"] for a in all_anchor_info)),
            "target_layers": sorted(self.target_layers),
            "window_size": self.experiment_config.window_size,
            "hidden_size": self.hidden_size,
            "num_sycophantic_samples": sum(1 for a in all_anchor_info if a["is_sycophantic"]),
            "anchors": all_anchor_info,
        }

        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Saved experiments.safetensors: {experiment_tensor.shape}")
        print(f"  Shape: [N_ANCHORS={len(all_anchor_info)}, N_LAYERS={len(self.target_layers)}, 1, d_model={self.hidden_size}]")
        print(f"Saved emergence.safetensors: {emergence_tensor.shape}")
        print(f"  Shape: [N_ANCHORS={len(all_anchor_info)}, N_LAYERS={len(self.target_layers)}, {self.experiment_config.window_size}, d_model={self.hidden_size}]")
        print(f"Saved metadata to {metadata_path}")

        return metadata

    def extract_trajectory_for_sample(
        self,
        sample_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Extract sentence-end activations with prob_ratio labels for regression."""
        sample_dir = (
            self.experiment_config.get_base_data_dir(self.model_config.name)
            / sample_id
        )
        base_path = sample_dir / "base.json"

        if not base_path.exists():
            return None

        with open(base_path, "r") as f:
            base_data = json.load(f)

        sentence_infos = self._collect_trajectory_from_sample(sample_id, base_data)
        if not sentence_infos:
            return None

        model, tokenizer = self._load_model()

        prompt = base_data.get("prompt", "")
        response = base_data.get("response", "")

        from ..generation.cot_utils import get_cot_prefix, get_cot_prefix_length
        cot_prefix = get_cot_prefix(self.model_config)
        full_text = prompt + cot_prefix + response
        cot_start_offset = len(prompt) + get_cot_prefix_length(self.model_config)

        hidden_states = self._extract_all_hidden_states(model, tokenizer, full_text)
        seq_len = next(iter(hidden_states.values())).shape[0]

        n_layers = len(self.target_layers)

        activations = []
        valid_infos = []

        for sent_info in sentence_infos:
            sent_end_pos = self._find_sentence_end_position(
                tokenizer, full_text, sent_info.sentence_text, cot_start_offset
            )

            if sent_end_pos is None or sent_end_pos >= seq_len:
                continue

            sent_info.sentence_end_position = sent_end_pos

            # Extract activation at sentence end for each layer
            act_tensor = torch.zeros(n_layers, self.hidden_size, dtype=torch.float16)
            for i, layer_idx in enumerate(sorted(self.target_layers)):
                layer_hidden = hidden_states[layer_idx]
                act_tensor[i, :] = layer_hidden[sent_end_pos, :]

            activations.append(act_tensor)
            valid_infos.append(sent_info)

        if not valid_infos:
            return None

        return {
            "sample_id": sample_id,
            "sentence_infos": valid_infos,
            "activations": activations,
        }

    def extract_trajectory_all(self) -> Dict[str, Any]:
        """Extract all trajectory activations for prob_ratio regression."""
        base_dir = self.experiment_config.get_base_data_dir(self.model_config.name)
        output_dir = self.experiment_config.get_activations_dir(self.model_config.name)

        sample_dirs = sorted(base_dir.glob("arc_conv_*"))

        all_activations = []
        all_sentence_info = []

        for sample_dir in tqdm(sample_dirs, desc="Extracting trajectory activations"):
            sample_id = sample_dir.name

            # Check for trajectory data in base.json
            base_path = sample_dir / "base.json"
            if not base_path.exists():
                continue

            result = self.extract_trajectory_for_sample(sample_id)
            if result is None:
                continue

            for i, sent_info in enumerate(result["sentence_infos"]):
                all_sentence_info.append({
                    "sample_id": sent_info.sample_id,
                    "sentence_idx": sent_info.sentence_idx,
                    "sentence_text": sent_info.sentence_text,
                    "prob_ratio": sent_info.prob_ratio,
                    "prob_correct": sent_info.prob_correct,
                    "prob_wrong": sent_info.prob_wrong,
                    "is_sycophantic": sent_info.is_sycophantic,
                    "sentence_end_position": sent_info.sentence_end_position,
                })
                all_activations.append(result["activations"][i])

        if not all_activations:
            raise ValueError("No trajectory activations extracted!")

        # Stack: [N_sentences, N_layers, hidden_size]
        activation_tensor = torch.stack(all_activations)

        prob_ratios = torch.tensor(
            [s["prob_ratio"] for s in all_sentence_info],
            dtype=torch.float32
        )
        is_sycophantic = torch.tensor(
            [s["is_sycophantic"] for s in all_sentence_info],
            dtype=torch.bool
        )

        # Save trajectory activations
        traj_path = output_dir / "trajectory.safetensors"
        save_file({
            "activations": activation_tensor,
            "prob_ratio": prob_ratios,
            "is_sycophantic": is_sycophantic,
        }, str(traj_path))

        # Save metadata
        traj_metadata = {
            "model": self.model_config.name,
            "num_sentences": len(all_sentence_info),
            "num_samples": len(set(s["sample_id"] for s in all_sentence_info)),
            "target_layers": sorted(self.target_layers),
            "hidden_size": self.hidden_size,
            "num_sycophantic": sum(1 for s in all_sentence_info if s["is_sycophantic"]),
            "sentences": all_sentence_info,
        }

        traj_metadata_path = output_dir / "trajectory_metadata.json"
        with open(traj_metadata_path, "w") as f:
            json.dump(traj_metadata, f, indent=2)

        print(f"Saved trajectory.safetensors: {activation_tensor.shape}")
        print(f"  Shape: [N_SENTENCES={len(all_sentence_info)}, N_LAYERS={len(self.target_layers)}, d_model={self.hidden_size}]")
        print(f"Saved trajectory metadata to {traj_metadata_path}")

        return traj_metadata

    def cleanup(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
