"""
Hyperopt Trial Execution Helper.
"""

import contextlib
import io
import shutil
import tempfile
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

from bioplausible.hyperopt.experiment import TrialRunner
from bioplausible.hyperopt.storage import HyperoptStorage


def run_single_trial_task(
    task: str,
    model_name: str,
    config: Dict[str, Any],
    storage_path: Optional[str] = None,
    job_id: Any = None,
    quick_mode: bool = True,
) -> Optional[Dict[str, float]]:
    """
    Run a single trial and return metrics.

    Args:
        task: Task name (e.g. 'shakespeare')
        model_name: Model architecture name
        config: Hyperparameter dictionary
        storage_path: Path to SQLite DB. If None, uses a temporary DB.
        quick_mode: If True, uses fewer data/iterations (default True).
    """
    temp_dir = None

    if storage_path is None:
        temp_dir = tempfile.mkdtemp()
        db_path = Path(temp_dir) / "worker_temp.db"
    else:
        db_path = Path(storage_path)

    try:
        storage = HyperoptStorage(str(db_path))

        # Create trial entry
        trial_id = storage.create_trial(model_name, config)

        # Create runner
        runner = TrialRunner(
            storage=storage, device="auto", task=task, quick_mode=quick_mode
        )

        # Override epochs if present
        if "epochs" in config:
            runner.epochs = int(config["epochs"])

        # Run
        # Suppress output to avoid cluttering the P2P log
        f = io.StringIO()
        with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            success = runner.run_trial(trial_id)

        if success:
            trial = storage.get_trial(trial_id)
            metrics = {
                "accuracy": trial.accuracy,
                "loss": trial.final_loss,
                "perplexity": trial.perplexity,
                "time": trial.iteration_time,
            }
            storage.close()
            return metrics
        else:
            storage.close()
            return None

    except Exception as e:
        print(f"Execution Error (Job {job_id}): {e}")
        traceback.print_exc()
        return None
    finally:
        if temp_dir:
            shutil.rmtree(temp_dir)
