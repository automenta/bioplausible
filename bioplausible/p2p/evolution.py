"""
P2P Evolutionary Controller.

Manages the autonomous discovery loop using the DHT.
"""

import threading
import time
import os
import logging
import random
from typing import Optional, Dict, Any

from bioplausible.p2p.dht import DHTNode
from bioplausible.hyperopt.search_space import get_search_space, SEARCH_SPACES
from bioplausible.p2p.node import Worker # Reuse worker logic for running jobs
from bioplausible.hyperopt.runner import run_single_trial_task
from bioplausible.p2p.state import load_state, save_state

logger = logging.getLogger("P2PEvolution")

class P2PEvolution:
    def __init__(self, bootstrap_ip: str = None, bootstrap_port: int = 8468,
                 discovery_mode: str = 'quick', constraints: Dict[str, Any] = None):
        self.bootstrap_nodes = [(bootstrap_ip, bootstrap_port)] if bootstrap_ip else []
        self.dht = None
        self.discovery_mode = discovery_mode
        self.constraints = constraints or {}

        self.running = False
        self.thread = None

        # State
        self.current_task = "shakespeare" # Default
        self.local_best_config = None
        self.local_best_score = -float('inf')

        state = load_state()
        self.points = state.get('points', 0)
        self.jobs_done = state.get('jobs_done', 0)

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
            # Try to bind to a port, with retries
            base_port = 8468 + random.randint(0, 1000)
            for i in range(10):
                try:
                    local_port = base_port + i
                    self.dht = DHTNode(port=local_port, bootstrap_nodes=self.bootstrap_nodes)
                    self.dht.start()
                    self._log(f"DHT started on port {local_port}")
                    break
                except Exception as e:
                    self._log(f"Port {local_port} busy/failed, retrying... ({e})")
                    if i == 9: raise # Rethrow on last attempt
                    time.sleep(0.5)

        except Exception as e:
            self._log(f"Failed to start DHT after retries: {e}")
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

                global_best_config = {}
                global_best_score = -float('inf')
                global_model_name = "EqProp MLP" # Default starting point

                if best_record:
                    global_best_config = best_record.get('config', {})
                    global_best_score = best_record.get('score', -float('inf'))
                    global_model_name = global_best_config.get('model_name', global_model_name)
                    self._log(f"Found global best: {global_best_score:.4f} ({global_model_name})")
                else:
                    self._log("No global best found. Will seed new...")

                # 2. Decide Strategy (New Arch, Crossover, or Mutate)
                action = "mutate"
                rnd = random.random()

                # Chance to switch architecture entirely (exploration)
                if rnd < 0.05:
                    action = "new_arch"
                # Chance to crossover if we have a local best compatible with global
                elif (self.local_best_config and best_record and
                      self.local_best_config.get('model_name') == global_model_name and
                      rnd < 0.35):
                    action = "crossover"
                else:
                    action = "mutate"

                # 3. Prepare Genome
                target_config = {}
                target_model_name = global_model_name

                if action == "new_arch":
                    self._update_status("Exploring New Architecture...")
                    # Pick random model from registry spaces
                    available_models = list(SEARCH_SPACES.keys())
                    target_model_name = random.choice(available_models)
                    space = get_search_space(target_model_name)
                    target_config = space.sample()
                    target_config['model_name'] = target_model_name
                    self._log(f"Selected new architecture: {target_model_name}")

                elif action == "crossover":
                    self._update_status("Crossing Over Genomes...")
                    space = get_search_space(global_model_name)
                    target_config = space.crossover(global_best_config, self.local_best_config)
                    target_config['model_name'] = global_model_name # Persist name
                    # Add small mutation to avoid stagnation
                    target_config = space.mutate(target_config, mutation_rate=0.1)
                    target_model_name = global_model_name

                else: # Mutate
                    self._update_status("Mutating Genome...")
                    # Decide which parent to mutate
                    # Favor global best, but sometimes use local best or random restart
                    parent_config = global_best_config
                    parent_model = global_model_name

                    if not best_record: # Bootstrap
                         space = get_search_space(global_model_name)
                         parent_config = space.sample()
                         parent_config['model_name'] = global_model_name
                    elif self.local_best_config and random.random() < 0.3:
                         parent_config = self.local_best_config
                         parent_model = parent_config.get('model_name', "EqProp MLP")

                    space = get_search_space(parent_model)
                    target_config = space.mutate(parent_config)
                    target_config['model_name'] = parent_model
                    target_model_name = parent_model

                # Re-fetch space with constraints applied for final verification/mutation context
                space = get_search_space(target_model_name)
                if self.constraints:
                    space = space.apply_constraints(self.constraints)
                    # Mutate again with constrained space to ensure we are in bounds
                    target_config = space.mutate(target_config, mutation_rate=0.0) # Rate 0 just clamps if implemented or we can just assume mutate clamps

                    # Manually clamp common keys if space.mutate doesn't enforce stricter bounds on existing values
                    if 'max_hidden' in self.constraints and 'hidden_dim' in target_config:
                        target_config['hidden_dim'] = min(target_config['hidden_dim'], self.constraints['max_hidden'])
                    if 'max_layers' in self.constraints and 'num_layers' in target_config:
                        target_config['num_layers'] = min(target_config['num_layers'], self.constraints['max_layers'])
                    if 'max_steps' in self.constraints and 'steps' in target_config:
                        target_config['steps'] = min(target_config['steps'], self.constraints['max_steps'])

                # Apply Mode Settings (Quick vs Deep)
                if self.discovery_mode == 'quick':
                    target_config['epochs'] = 1
                    if 'steps' in target_config:
                        target_config['steps'] = min(target_config['steps'], 15)
                elif self.discovery_mode == 'deep':
                    target_config['epochs'] = 5
                    # Allow larger steps

                # 4. Evaluate
                self._update_status(f"Evaluating: {target_model_name}")

                # Use Worker's logic to run job locally
                job_id = random.randint(1000, 9999)

                metrics = run_single_trial_task(
                    task=self.current_task,
                    model_name=target_model_name,
                    config=target_config,
                    storage_path="results/hyperopt.db",
                    job_id=job_id
                )

                if metrics:
                    acc = metrics.get('accuracy', 0.0)
                    self.jobs_done += 1
                    self.points += 5
                    save_state(self.points, self.jobs_done)

                    self._log(f"Eval complete: {acc:.4f} (Global Best: {global_best_score:.4f})")

                    # Update Local Best
                    if acc > self.local_best_score:
                        self.local_best_score = acc
                        self.local_best_config = target_config
                        self._log(f"New Local Best! ({acc:.4f})")

                    # Publish if Global Best
                    if acc > global_best_score:
                        self._update_status("Publishing Discovery...")
                        # Ensure model_name inside config
                        target_config['model_name'] = target_model_name
                        self.dht.publish_best_model(self.current_task, target_config, acc)
                        self.points += 50
                        save_state(self.points, self.jobs_done)
                        self._log(f"🎉 New Global Best Discovered! ({acc:.4f})")

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
