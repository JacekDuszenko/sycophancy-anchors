"""Progress tracker for monitoring syco/nonsyco sample counts."""

import json
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class Progress:
    """Progress state."""
    syco_count: int = 0
    nonsyco_count: int = 0
    total_processed: int = 0
    failed_count: int = 0


class ProgressTracker:
    """Track progress toward target sample counts."""

    def __init__(
        self,
        target_syco: int = 101,
        target_nonsyco: int = 410,
    ):
        """Initialize progress tracker.

        Args:
            target_syco: Target sycophantic sample count
            target_nonsyco: Target non-sycophantic sample count
        """
        self.target_syco = target_syco
        self.target_nonsyco = target_nonsyco
        self.progress = Progress()

    def update(self, is_sycophantic: bool) -> None:
        """Update progress with a new sample.

        Args:
            is_sycophantic: Whether the sample is sycophantic
        """
        self.progress.total_processed += 1
        if is_sycophantic:
            self.progress.syco_count += 1
        else:
            self.progress.nonsyco_count += 1

    def mark_failed(self) -> None:
        """Mark a sample as failed."""
        self.progress.failed_count += 1

    def is_complete(self) -> bool:
        """Check if target counts have been reached.

        Returns:
            True if both targets met
        """
        return (
            self.progress.syco_count >= self.target_syco
            and self.progress.nonsyco_count >= self.target_nonsyco
        )

    def needs_syco(self) -> bool:
        """Check if more sycophantic samples needed.

        Returns:
            True if more syco samples needed
        """
        return self.progress.syco_count < self.target_syco

    def needs_nonsyco(self) -> bool:
        """Check if more non-sycophantic samples needed.

        Returns:
            True if more nonsyco samples needed
        """
        return self.progress.nonsyco_count < self.target_nonsyco

    def get_status(self) -> str:
        """Get human-readable status string.

        Returns:
            Status string
        """
        return (
            f"syco: {self.progress.syco_count}/{self.target_syco}, "
            f"nonsyco: {self.progress.nonsyco_count}/{self.target_nonsyco}, "
            f"total: {self.progress.total_processed}, "
            f"failed: {self.progress.failed_count}"
        )

    def save(self, path: Path) -> None:
        """Save progress to file.

        Args:
            path: Path to save file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "progress": asdict(self.progress),
            "target_syco": self.target_syco,
            "target_nonsyco": self.target_nonsyco,
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: Path) -> bool:
        """Load progress from file.

        Args:
            path: Path to load from

        Returns:
            True if loaded successfully
        """
        path = Path(path)
        if not path.exists():
            return False

        try:
            with open(path, "r") as f:
                data = json.load(f)

            prog = data.get("progress", {})
            self.progress = Progress(
                syco_count=prog.get("syco_count", 0),
                nonsyco_count=prog.get("nonsyco_count", 0),
                total_processed=prog.get("total_processed", 0),
                failed_count=prog.get("failed_count", 0),
            )

            return True

        except (json.JSONDecodeError, KeyError):
            return False

    def reset(self) -> None:
        """Reset progress to initial state."""
        self.progress = Progress()
