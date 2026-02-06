"""Checkpoint manager for saving and resuming pipeline state."""

import json
import os
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional


class CheckpointManager:
    """Manage checkpoints for pipeline stages."""

    def __init__(self, base_dir: Path):
        """Initialize checkpoint manager.

        Args:
            base_dir: Base directory for checkpoints
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def atomic_write(self, path: Path, data: Dict[str, Any]) -> None:
        """Write data atomically using tmp file + rename.

        Args:
            path: Target path
            data: Data to write (will be JSON serialized)
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Write to temp file in same directory
        fd, tmp_path = tempfile.mkstemp(
            dir=str(path.parent),
            suffix=".tmp",
        )

        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=2)

            # Atomic rename
            os.replace(tmp_path, path)

        except Exception:
            # Clean up temp file on error
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

    def read_checkpoint(self, path: Path) -> Optional[Dict[str, Any]]:
        """Read a checkpoint file.

        Args:
            path: Path to checkpoint

        Returns:
            Checkpoint data or None if not found
        """
        path = Path(path)
        if not path.exists():
            return None

        with open(path, "r") as f:
            return json.load(f)

    def is_sample_complete(self, sample_id: str) -> bool:
        """Check if a sample has been completed.

        Args:
            sample_id: Sample identifier

        Returns:
            True if sample is complete
        """
        sample_dir = self.base_dir / sample_id
        return (sample_dir / "base.json").exists()

    def get_completed_samples(self) -> list:
        """Get list of completed sample IDs.

        Returns:
            List of sample IDs
        """
        completed = []
        for sample_dir in self.base_dir.glob("arc_conv_*"):
            if (sample_dir / "base.json").exists():
                completed.append(sample_dir.name)
        return sorted(completed)

    def get_last_sample_idx(self) -> int:
        """Get the index of the last completed sample.

        Returns:
            Last sample index or -1 if none
        """
        completed = self.get_completed_samples()
        if not completed:
            return -1

        # Extract index from sample_id like "arc_conv_0042"
        last = completed[-1]
        try:
            return int(last.split("_")[-1])
        except ValueError:
            return -1

    def save_stage_checkpoint(
        self,
        stage_name: str,
        data: Dict[str, Any],
    ) -> Path:
        """Save a stage-level checkpoint.

        Args:
            stage_name: Name of the stage
            data: Checkpoint data

        Returns:
            Path to checkpoint file
        """
        checkpoint_path = self.base_dir / f"{stage_name}_checkpoint.json"
        self.atomic_write(checkpoint_path, data)
        return checkpoint_path

    def load_stage_checkpoint(
        self,
        stage_name: str,
    ) -> Optional[Dict[str, Any]]:
        """Load a stage-level checkpoint.

        Args:
            stage_name: Name of the stage

        Returns:
            Checkpoint data or None
        """
        checkpoint_path = self.base_dir / f"{stage_name}_checkpoint.json"
        return self.read_checkpoint(checkpoint_path)

    def mark_stage_complete(self, stage_name: str) -> None:
        """Mark a stage as complete.

        Args:
            stage_name: Name of the stage
        """
        flag_path = self.base_dir / f"{stage_name}_complete.flag"
        flag_path.touch()

    def is_stage_complete(self, stage_name: str) -> bool:
        """Check if a stage is complete.

        Args:
            stage_name: Name of the stage

        Returns:
            True if stage is complete
        """
        flag_path = self.base_dir / f"{stage_name}_complete.flag"
        return flag_path.exists()
