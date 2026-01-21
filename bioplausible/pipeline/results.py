import os
import json
import shutil
import glob
from datetime import datetime
from typing import List, Dict, Any, Optional

class ResultsManager:
    """Manages persistence of training results."""

    BASE_DIR = "results/runs"

    def __init__(self, base_dir=None):
        if base_dir:
            self.BASE_DIR = base_dir
        os.makedirs(self.BASE_DIR, exist_ok=True)

    def save_run(self, run_id: str, config: Dict[str, Any], metrics: Dict[str, Any]):
        """Save a training run."""
        run_dir = os.path.join(self.BASE_DIR, run_id)
        os.makedirs(run_dir, exist_ok=True)

        data = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "config": config,
            "metrics": metrics
        }

        with open(os.path.join(run_dir, "metadata.json"), "w") as f:
            json.dump(data, f, indent=2)

    def list_runs(self) -> List[Dict[str, Any]]:
        """List all saved runs."""
        runs = []
        for meta_path in glob.glob(os.path.join(self.BASE_DIR, "*/metadata.json")):
            try:
                with open(meta_path, "r") as f:
                    runs.append(json.load(f))
            except Exception as e:
                print(f"Error loading run metadata {meta_path}: {e}")

        # Sort by timestamp descending
        runs.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return runs

    def load_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Load specific run data."""
        path = os.path.join(self.BASE_DIR, run_id, "metadata.json")
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
        return None

    def delete_run(self, run_id: str):
        """Delete a run."""
        run_dir = os.path.join(self.BASE_DIR, run_id)
        if os.path.exists(run_dir):
            shutil.rmtree(run_dir)
