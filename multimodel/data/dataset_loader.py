"""Dataset loader for ARC conversation data from HuggingFace."""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from huggingface_hub import hf_hub_download


class DatasetLoader:
    """Load and manage the ARC conversation dataset."""

    REPO_ID = "jacekduszenko/sycophantic-anchors"
    FILENAME = "arc_conversation_2k.json"

    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize dataset loader.

        Args:
            cache_dir: Directory to cache downloaded files
        """
        self.cache_dir = cache_dir or Path(".cache/multimodel")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._data: Optional[List[Dict[str, Any]]] = None

    def download(self) -> Path:
        """Download the dataset from HuggingFace.

        Returns:
            Path to downloaded file
        """
        local_path = hf_hub_download(
            repo_id=self.REPO_ID,
            filename=self.FILENAME,
            repo_type="dataset",
            cache_dir=str(self.cache_dir),
        )
        return Path(local_path)

    def load(self) -> List[Dict[str, Any]]:
        """Load the dataset, downloading if necessary.

        Returns:
            List of conversation samples
        """
        if self._data is not None:
            return self._data

        filepath = self.download()
        with open(filepath, "r", encoding="utf-8") as f:
            self._data = json.load(f)

        return self._data

    def get_sample(self, idx: int) -> Dict[str, Any]:
        """Get a single sample by index.

        Args:
            idx: Sample index

        Returns:
            Sample dictionary
        """
        data = self.load()
        return data[idx]

    def __len__(self) -> int:
        """Get number of samples in dataset."""
        return len(self.load())

    def __iter__(self):
        """Iterate over samples."""
        return iter(self.load())


def load_arc_dataset(cache_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Convenience function to load the ARC dataset.

    Args:
        cache_dir: Optional cache directory

    Returns:
        List of conversation samples
    """
    loader = DatasetLoader(cache_dir=cache_dir)
    return loader.load()
