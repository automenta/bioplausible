"""
AutoScientistCampaign: Orchestrates multi-day autonomous campaigns.

Combines the Scientist (execution) with AutoScientist (reasoning/proposal)
into a continuous discovery loop with:
 - Hypothesis generation
 - Experiment proposal
 - Execution via Scientist or CoreTrainer
 - Result analysis
 - KnowledgeBase update
"""

import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from bioplausible.autoscientist.proposer import ExperimentProposer
from bioplausible.autoscientist.reasoner import HypothesisReasoner
from bioplausible.core.exceptions import KnowledgeBaseError
from bioplausible.core.trainer import CoreTrainer, TrainerConfig
from bioplausible.knowledge import KnowledgeBase, KnowledgeEntry

__all__ = [
    "AutoScientistCampaign",
    "logger",
]
logger = logging.getLogger(__name__)


class AutoScientistCampaign:
    """
    Autonomous research campaign manager.

    Runs continuous discovery loops:
        1. Reason: Generate hypotheses from KnowledgeBase
        2. Propose: Convert hypotheses to experiment proposals
        3. Execute: Run experiments via CoreTrainer
        4. Learn: Update KnowledgeBase with results
    """

    def __init__(
        self,
        knowledge_base: KnowledgeBase | None = None,
        output_dir: str = "autoscientist_campaigns",
        max_concurrent: int = 1,
        human_approval_gate: bool = False,
    ):
        self.knowledge_base = knowledge_base or KnowledgeBase()
        self.proposer = ExperimentProposer(self.knowledge_base)
        self.reasoner = HypothesisReasoner(self.knowledge_base)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_concurrent = max_concurrent
        self.human_approval_gate = human_approval_gate
        self.campaign_log: list[dict[str, object]] = []
        self._iteration = 0

    def run_iteration(
        self,
        domain: str | None = None,
        n_experiments: int = 5,
        dry_run: bool = False,
    ) -> list[dict[str, object]]:
        """
        Run one iteration of the discovery loop.

        Args:
            domain: Optional domain filter.
            n_experiments: Number of experiments to propose and run.
            dry_run: If True, only propose without executing.

        Returns:
            List of experiment results.
        """
        self._iteration += 1
        logger.info("=== Campaign Iteration %s ===", self._iteration)

        # Step 1: Analyze KnowledgeBase
        insights = self.reasoner.analyze_knowledge_base()
        if insights:
            logger.info("KnowledgeBase insights (%s):", len(insights))
            for insight in insights[:3]:
                logger.info("  - %s", insight)

        # Step 2: Propose experiments
        proposals = self.proposer.propose_batch(
            domain=domain,
            n_proposals=n_experiments,
        )

        if not proposals:
            logger.warning("No proposals generated. Skipping iteration.")
            return []

        logger.info("Proposed %s experiments", len(proposals))

        # Step 3: Human approval gate
        if self.human_approval_gate:
            approved = self._human_approval(proposals)
            proposals = [p for i, p in enumerate(proposals) if i in approved]
            if not proposals:
                logger.info("No proposals approved. Skipping.")
                return []

        # Step 4: Execute experiments
        results = []
        if not dry_run:
            for i, proposal in enumerate(proposals):
                logger.info(
                    f"Executing proposal {i + 1}/{len(proposals)}: {proposal.model}"
                )
                try:
                    result = self._execute_proposal(proposal)
                    results.append(result)

                    # Update KnowledgeBase
                    self._update_knowledge_base(proposal, result)
                except Exception as e:  # noqa: BLE001  # broad: a failing trial must not stop the campaign
                    logger.error("Proposal %s failed: %s", i, e, exc_info=True)
                    results.append({
                        "proposal": proposal,
                        "status": "failed",
                        "error": str(e),
                    })

            # Log campaign progress
            self._log_iteration(proposals, results)

        return results

    def _execute_proposal(self, proposal) -> dict[str, object]:
        """Execute a single experiment proposal via CoreTrainer."""
        config = TrainerConfig(
            model=proposal.model,
            task=proposal.task,
            optimizer=proposal.optimizer,
            epochs=5,
            batch_size=64,
            track_energy=True,
            tags={
                "hypothesis": proposal.hypothesis,
                "autoscientist": True,
                "iteration": self._iteration,
            },
        )

        if proposal.propagator:
            config.propagator = proposal.propagator

        # Override with proposal hyperparams
        for k, v in proposal.hyperparams.items():
            if hasattr(config, k):
                setattr(config, k, v)

        trainer = CoreTrainer(config)
        history = trainer.fit()

        return {
            "proposal": {
                "hypothesis": proposal.hypothesis,
                "model": proposal.model,
                "task": proposal.task,
                "propagator": proposal.propagator,
                "optimizer": proposal.optimizer,
                "justification": proposal.justification,
            },
            "status": "completed",
            "metrics": [m.to_dict() for m in history],
            "final_accuracy": history[-1].val_accuracy if history else 0.0,
            "final_loss": history[-1].val_loss if history else 0.0,
            "train_accuracy": history[-1].train_accuracy if history else 0.0,
            "epochs_completed": len(history),
        }

    def _update_knowledge_base(self, proposal, result: dict[str, object]) -> None:
        """Store experiment result in KnowledgeBase with schema validation."""
        try:
            entry = KnowledgeEntry(
                id=f"campaign_{self._iteration}_{int(time.time())}",
                topic=f"experiment:{proposal.task}",
                model_family=proposal.model,
                finding=(
                    f"Accuracy: {result.get('final_accuracy', 'N/A'):.4f}, "
                    f"Loss: {result.get('final_loss', 'N/A'):.4f}"
                ),
                details=(
                    f"Hypothesis: {proposal.hypothesis}\n"
                    f"Propagator: {proposal.propagator}\n"
                    f"Optimizer: {proposal.optimizer}\n"
                    f"Config: {proposal.hyperparams}\n"
                    f"Epochs: {result.get('epochs_completed', 'N/A')}"
                ),
                confidence=float(result.get("final_accuracy", 0.0) or 0.0),
                tags=["experiment", proposal.task, proposal.model],
                source="experiment",
                metrics={
                    k: v
                    for k, v in result.items()
                    if isinstance(v, (int, float)) and v is not None
                },
                hyperparameters=(
                    proposal.hyperparams
                    if isinstance(proposal.hyperparams, dict)
                    else {"raw": str(proposal.hyperparams)}
                ),
                extra={"campaign_iteration": self._iteration},
            )
            self.knowledge_base.add_entry(entry)
        except (KnowledgeBaseError, OSError, ValueError) as e:
            logger.warning("Failed to update KnowledgeBase: %s", e)

    def _log_iteration(
        self,
        proposals: list,
        results: list[dict[str, object]],
    ) -> None:
        """Log campaign iteration to disk."""
        iteration_log = {
            "iteration": self._iteration,
            "timestamp": datetime.now().isoformat(),
            "n_proposals": len(proposals),
            "n_completed": sum(1 for r in results if r.get("status") == "completed"),
            "n_failed": sum(1 for r in results if r.get("status") == "failed"),
            "proposals": [
                {
                    "model": p.model,
                    "task": p.task,
                    "hypothesis": p.hypothesis[:100],
                }
                for p in proposals
            ],
            "results": [
                {
                    "status": r.get("status"),
                    "final_accuracy": r.get("final_accuracy"),
                }
                for r in results
            ],
        }
        self.campaign_log.append(iteration_log)

        # Save to disk
        log_path = self.output_dir / f"iteration_{self._iteration:04d}.json"
        with Path(log_path).open("w") as f:
            json.dump(iteration_log, f, indent=2, default=str)

        logger.info("Iteration log saved: %s", log_path)

    def _human_approval(
        self,
        proposals: list,
    ) -> list[int]:
        """Gate for human approval of expensive runs.

        When ``stdin`` is not a TTY (e.g. CI, batch jobs) the gate
        auto-approves everything. Otherwise each proposal is presented and
        the operator answers ``y/n``; non-``y`` answers reject that
        proposal index.
        """
        logger.info("Human approval gate: %d proposals pending", len(proposals))
        if not proposals:
            return []
        if not sys.stdin.isatty():
            auto_msg = (
                "stdin is not a TTY; auto-approving all proposals. "
                "Provide BIOPL_AUTO_APPROVE=0 to deny."
            )
            if os.environ.get("BIOPL_AUTO_APPROVE", "1") == "0":
                logger.warning(auto_msg + " Denying all (BIOPL_AUTO_APPROVE=0).")
                return []
            logger.info(auto_msg)
            return list(range(len(proposals)))
        approved: list[int] = []
        for idx, proposal in enumerate(proposals):
            label = getattr(proposal, "name", None) or str(proposal)
            try:
                answer = (
                    input(f"Approve proposal {idx} ({label})? [y/N] ").strip().lower()
                )
            except EOFError:
                logger.warning("EOF on stdin; auto-approving remaining proposals")
                approved.extend(range(idx, len(proposals)))
                return approved
            if answer == "y":
                approved.append(idx)
        return approved

    def get_summary(self) -> dict[str, object]:
        """Get campaign summary statistics."""
        completed = []
        for entry in self.campaign_log:
            for r in entry.get("results", []):
                if r.get("status") == "completed":
                    completed.append(r)

        total_experiments = sum(
            entry.get("n_proposals", 0) for entry in self.campaign_log
        )
        best_accuracy = 0.0
        if completed:
            best_accuracy = max(r.get("final_accuracy", 0) for r in completed)

        return {
            "iterations": self._iteration,
            "total_experiments": total_experiments,
            "completed": len(completed),
            "best_accuracy": best_accuracy,
            "output_dir": str(self.output_dir),
        }
