#!/usr/bin/env python3
"""Standalone script for discovering CoT end tokens for all models."""

import argparse
import json
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from multimodel.config.model_registry import (
    MODEL_CONFIGS,
    MODEL_ORDER,
    get_model_config,
)
from multimodel.generation.cot_token_discovery import CoTTokenDiscovery


def discover_for_model(model_name: str, n_samples: int = 5) -> dict:
    """Discover CoT token for a single model.

    Args:
        model_name: Model identifier
        n_samples: Number of test samples to generate

    Returns:
        Discovery result dictionary
    """
    print(f"\n{'='*50}")
    print(f"Discovering CoT token for: {model_name}")
    print("=" * 50)

    model_config = get_model_config(model_name)

    # Skip if already has token
    if model_config.cot_end_token is not None:
        print(f"Already has CoT token: {model_config.cot_end_token}")
        return {
            "model": model_name,
            "cot_token": model_config.cot_end_token,
            "source": "predefined",
        }

    discovery = CoTTokenDiscovery(model_config, n_samples=n_samples)

    try:
        token = discovery.discover()
        return {
            "model": model_name,
            "cot_token": token,
            "source": "discovered" if token else "not_found",
        }
    finally:
        discovery.cleanup()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Discover CoT end tokens for reasoning models"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=list(MODEL_CONFIGS.keys()),
        help="Specific model to discover for",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Discover for all models",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=5,
        help="Number of test samples per model",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="cot_tokens.json",
        help="Output file for discovered tokens",
    )

    args = parser.parse_args()

    if not args.model and not args.all:
        parser.error("Must specify --model or --all")

    models = MODEL_ORDER if args.all else [args.model]
    results = []

    for model_name in models:
        try:
            result = discover_for_model(model_name, args.n_samples)
            results.append(result)
            print(f"Result: {result}")
        except Exception as e:
            print(f"Error discovering for {model_name}: {e}")
            results.append({
                "model": model_name,
                "cot_token": None,
                "source": "error",
                "error": str(e),
            })

    # Save results
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*50}")
    print(f"Results saved to: {output_path}")
    print("=" * 50)

    # Print summary
    print("\nSummary:")
    for r in results:
        status = "✓" if r.get("cot_token") else "✗"
        token = r.get("cot_token") or "None"
        print(f"  {status} {r['model']}: {token} ({r['source']})")


if __name__ == "__main__":
    main()
