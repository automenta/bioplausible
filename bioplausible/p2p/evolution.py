"""
P2P Evolutionary Controller.

Manages the autonomous discovery loop using the DHT.
"""

import threading
import time
import os
import logging
import random
from typing import Optional, Dict

from bioplausible.p2p.dht import DHTNode
from bioplausible.hyperopt.search_space import get_search_space
from bioplausible.p2p.node import Worker # Reuse worker logic for running jobs

logger = logging.getLogger("P2PEvolution")

class P2PEvolution:
    def __init__(self, bootstrap_ip: str = None, bootstrap_port: int = 8468):
        self.bootstrap_nodes = [(bootstrap_ip, bootstrap_port)] if bootstrap_ip else []
        self.dht = None
        self.worker_logic = Worker("dummy") # Used for _run_job
        self.running = False
        self.thread = None

        # State
        self.current_task = "shakespeare" # Default
        self.points = 0
        self.jobs_done = 0
        self.current_status = "Stopped"

        # Signals
        self.on_status_change = None
        self.on_log = None

    def _log(self, msg):
        logger.info(msg)
        if self.on_log:
            self.on_log(msg)

    def _update_status(self, status):
        self.current_status = status
        if self.on_status_change:
            self.on_status_change(status, self.points, self.jobs_done)

    def start(self, auto_nice=True):
        if self.running: return

        if auto_nice and hasattr(os, 'nice'):
            try:
                os.nice(10) # Lower priority
                self._log("Process priority lowered (Nice +10)")
            except Exception as e:
                self._log(f"Could not lower priority: {e}")

        # Start DHT
        try:
            # Random local port to avoid conflicts if running multiple on same machine
            local_port = 8468 + random.randint(0, 1000)
            self.dht = DHTNode(port=local_port, bootstrap_nodes=self.bootstrap_nodes)
            self.dht.start()
        except Exception as e:
            self._log(f"Failed to start DHT: {e}")
            return

        self.running = True
        self.thread = threading.Thread(target=self._evolution_loop, daemon=True)
        self.thread.start()
        self._update_status("Starting P2P Mesh...")

    def stop(self):
        self.running = False
        if self.dht:
            self.dht.stop()
        if self.thread:
            self.thread.join(timeout=2)
        self._update_status("Stopped")

    def _evolution_loop(self):
        self._log("Joined P2P Mesh network.")

        while self.running:
            try:
                # 1. Fetch Global Best
                self._update_status("Syncing with Mesh...")
                best_record = self.dht.get_best_model(self.current_task)

                model_name = "EqProp MLP" # Default starting point
                config = {}
                score_to_beat = -float('inf')

                if best_record:
                    config = best_record.get('config', {})
                    score_to_beat = best_record.get('score', -float('inf'))
                    model_name = config.get('model_name', model_name) # Ensure model name is preserved
                    self._log(f"Found global best: {score_to_beat:.4f}")
                else:
                    self._log("No global best found. Seeding new...")
                    space = get_search_space(model_name)
                    config = space.sample()
                    # Ensure model name is in config for downstream logic
                    config['model_name'] = model_name

                # 2. Mutate
                self._update_status("Mutating Genome...")
                space = get_search_space(model_name)
                # Ensure we have a valid config to mutate
                if not config: config = space.sample()

                # Apply Mutation
                mutated_config = space.mutate(config)
                # Fix parameters for quick loop
                mutated_config['epochs'] = 1
                if 'steps' in mutated_config:
                    mutated_config['steps'] = max(5, min(mutated_config['steps'], 20))

                # 3. Evaluate
                self._update_status(f"Evaluating: {model_name}")

                # Use Worker's logic to run job locally
                # job_id is random for local logging
                job_id = random.randint(1000, 9999)

                metrics = self.worker_logic._run_job(
                    job_id=job_id,
                    task=self.current_task,
                    model_name=model_name,
                    config=mutated_config
                )

                if metrics:
                    acc = metrics.get('accuracy', 0.0)
                    self.jobs_done += 1
                    self.points += 5 # 5 points for DHT contribution

                    self._log(f"Evaluation complete. Acc: {acc:.4f} (Beat: {score_to_beat:.4f})")

                    # 4. Publish if better
                    if acc > score_to_beat:
                        self._update_status("Publishing Discovery...")
                        # Store model_name inside config for retrieval
                        mutated_config['model_name'] = model_name
                        self.dht.publish_best_model(self.current_task, mutated_config, acc)
                        self.points += 50 # Bonus for improvement
                        self._log(f"🎉 New Best Model Discovered! ({acc:.4f})")

                else:
                    self._log("Evaluation failed.")

            except Exception as e:
                self._log(f"Evolution Loop Error: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(5)

            # Rest
            if self.running:
                self._update_status("Resting...")
                time.sleep(2)
