# Sycophantic Anchors

Code for the paper: **"Sycophantic Anchors: Detecting When Reasoning Models Lock Into User Agreement"**

We introduce *sycophantic anchors*—sentences in reasoning traces where models commit to agreeing with incorrect user suggestions. Using counterfactual rollout analysis, we identify these anchors and show they can be detected mid-inference with linear probes (84.6% balanced accuracy).

## Dataset

Dataset available at: https://huggingface.co/datasets/jacekduszenko/sycophantic-anchors

The dataset contains 1,101 adversarial multi-turn conversations:
- **101 sycophantic samples** (model agrees with wrong user suggestion)
- **1,000 correct reasoning samples** (model resists user suggestion)

Each sample includes the full reasoning trace, sentence segmentation, and probability trajectories.

## Repository Structure

```
├── generate_rollouts_vllm_spacy_fix.py   # Counterfactual rollout generation
├── generate_sycophantic_base.py          # Base response generation for sycophancy analysis
├── train_sycophancy_detector.py          # Linear probe training for anchor detection
├── experiment_pairwise_discriminability.py # Pairwise classification experiments
├── train_binary_time_probe.py            # Sycophancy emergence analysis
├── train_causal_impact_regressor.py      # Strength regression (R² = 0.74)
├── train_layer_probes.py                 # Layer-wise probe analysis
├── train_three_class_probe.py            # Three-class anchor classification
├── generate_appendix_figure.py           # Figure generation for paper
├── detector_utils.py                     # Utility functions for detection
├── utils.py                              # General utilities
└── prompts.py                            # Prompt templates
```

## Reproducing Results

### Prerequisites

```bash
pip install torch transformers vllm spacy scikit-learn matplotlib
python -m spacy download en_core_web_sm
```

### 1. Generate Base Responses

Generate initial model responses to adversarial conversations:

```bash
python generate_sycophantic_base.py \
    --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --temperature 0.6 \
    --top_p 0.95
```

### 2. Generate Counterfactual Rollouts

For each sentence position, generate 20 counterfactual completions:

```bash
python generate_rollouts_vllm_spacy_fix.py \
    --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --num_rollouts 20 \
    --temperature 0.6 \
    --top_p 0.95
```

### 3. Train Pairwise Classification Probes (Figure 2)

Train linear probes to distinguish sycophantic anchors from correct anchors and neutral sentences:

```bash
python experiment_pairwise_discriminability.py \
    --layer 28 \
    --n_seeds 10
```

Expected results:
- Sycophantic vs Correct anchors: **84.6%** balanced accuracy
- Sycophantic vs Neutral: **77.5%** balanced accuracy
- Correct vs Neutral: **64.0%** balanced accuracy

### 4. Analyze Sycophancy Emergence (Figure 3)

Train probes at each token position before anchor to study when sycophancy emerges:

```bash
python train_binary_time_probe.py \
    --window_size 30 \
    --layer 28
```

Expected results:
- Prompt final token: **55.1%** (near chance)
- At anchor: **72.9%** (17.8 pp increase during reasoning)

### 5. Train Strength Regressor (Figure 4)

Train MLP regressor to predict probability ratio from activations:

```bash
python train_causal_impact_regressor.py \
    --layer 28 \
    --hidden_dim 256
```

Expected results:
- R² = **0.742** (74% variance explained)
- RMSE = **1.90**

### 6. Layer-wise Analysis

Analyze probe accuracy across all model layers:

```bash
python train_layer_probes.py
```

## Key Findings

1. **Asymmetric Detectability**: Sycophantic anchors are highly distinguishable (84.6%) while correct reasoning anchors are nearly indistinguishable from neutral text (64.0%)

2. **Gradual Emergence**: Sycophancy builds during reasoning (55% → 73% detectability), not pre-determined by prompt

3. **Quantifiable Strength**: Activations encode not just presence but magnitude of sycophantic tendency (R² = 0.74)

## Model Details

- **Model**: DeepSeek-R1-Distill-Llama-8B
- **Dataset**: ARC (AI2 Reasoning Challenge)
- **Anchor threshold**: δ = 0.50 (50% accuracy shift)
- **Probe layer**: 28 (of 32)
- **Statistical methodology**: 10 random seeds, 80/20 train/test splits

## Citation

```bibtex
@article{duszenko2026sycophantic,
  title={Sycophantic Anchors: Detecting When Reasoning Models Lock Into User Agreement},
  author={Duszenko, Jacek},
  journal={ICLR 2026},
  year={2026}
}
```

## Acknowledgments

This work builds on the [Thought Anchors](https://arxiv.org/abs/2506.19143) framework by Bogdan et al.
