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
├── multimodel/                  # Core pipeline package
│   ├── config/                  #   Model registry & experiment config
│   ├── data/                    #   Dataset loading (HuggingFace)
│   ├── generation/              #   Base, rollout, and judge generation (vLLM)
│   ├── extraction/              #   Activation extraction
│   ├── experiments/             #   Probes, regressors, emergence analysis
│   ├── checkpointing/           #   Checkpoint & progress tracking
│   ├── output/                  #   CSV export
│   └── scripts/                 #   Entry points (run_pipeline.py, shell wrappers)
├── scripts/                     # Analysis & post-processing
│   ├── aggregate_multimodel_results.py
│   ├── prepare_llama_multimodel.py
│   ├── text_baseline_comparison.py
│   └── threshold_sweep_analysis.py
├── latex/                       # Paper source (ICLR 2026 / KDD 2026)
└── requirements.txt
```

## Reproducing Results

### Prerequisites

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Run the full pipeline for a single model

The pipeline handles base generation, counterfactual rollouts, judging, activation extraction, and all experiments in one command:

```bash
VLLM_WORKER_MULTIPROC_METHOD=spawn python multimodel/scripts/run_pipeline.py \
    --model deepseek-qwen-1.5b \
    --n-samples 101 \
    --rollouts-per-sentence 20
```

See `multimodel/config/model_registry.py` for the list of supported model keys.

### Run across all models

```bash
bash multimodel/scripts/run_prod_loop.sh
```

### Aggregate cross-model results

```bash
python scripts/aggregate_multimodel_results.py --results-dir multimodel_prod/
```

## Key Findings

1. **Asymmetric Detectability**: Sycophantic anchors are highly distinguishable (84.6%) while correct reasoning anchors are nearly indistinguishable from neutral text (64.0%)

2. **Gradual Emergence**: Sycophancy builds during reasoning (55% → 73% detectability), not pre-determined by prompt

3. **Quantifiable Strength**: Activations encode not just presence but magnitude of sycophantic tendency (R² = 0.74)

## Model Details

- **Models**: Multi-model study across DeepSeek-R1 distillations (Qwen 1.5B/7B/14B/32B, Llama 8B/70B)
- **Dataset**: ARC (AI2 Reasoning Challenge)
- **Anchor threshold**: δ = 0.50 (50% accuracy shift)
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
