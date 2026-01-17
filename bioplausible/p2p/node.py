import threading
import json
import time
import socket
import logging
import uuid
import os
import shutil
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from urllib.request import Request, urlopen
from urllib.error import URLError
from typing import Optional, Dict, Any, List

from bioplausible.hyperopt.search_space import get_search_space
from bioplausible.hyperopt.runner import run_single_trial_task

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("P2PNode")

class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True

class CoordinatorHandler(BaseHTTPRequestHandler):
    def _send_response(self, data: Dict[str, Any], status=200):
        self.send_response(status)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))

    def do_GET(self):
        if self.path == '/status':
            self._send_response({
                "status": "online",
                "nodes": len(self.server.coordinator.nodes),
                "jobs_completed": self.server.coordinator.jobs_completed
            })
        elif self.path.startswith('/get_job'):
            # Parse query params (simple)
            # /get_job?client_id=xyz
            job = self.server.coordinator.get_job()
            if job:
                self._send_response(job)
            else:
                self._send_response({"status": "no_jobs"})
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == '/submit_result':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            try:
                result = json.loads(post_data.decode('utf-8'))
                self.server.coordinator.submit_result(result)
                self._send_response({"status": "accepted"})
            except Exception as e:
                logger.error(f"Error processing result: {e}")
                self.send_error(500)
        elif self.path == '/register':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            client_id = data.get('client_id', 'unknown')
            self.server.coordinator.register_node(client_id)
            self._send_response({"status": "registered"})
        else:
            self.send_error(404)

    def log_message(self, format, *args):
        pass # Suppress default logging

class Coordinator:
    def __init__(self, host='0.0.0.0', port=8000):
        self.host = host
        self.port = port
        self.server = None
        self.thread = None
        self.running = False
        self.nodes = set()
        self.jobs_completed = 0
        self.job_counter = 0
        self.lock = threading.Lock()

        # Simple job queue
        self.job_queue = []
        self._populate_initial_jobs()

    def _populate_initial_jobs(self):
        # Generate some random jobs for testing
        # In a real system, this would come from an evolutionary algorithm
        tasks = ["shakespeare", "mnist"]
        models = ["EqProp MLP", "Backprop Baseline"]

        for i in range(10):
            model = models[i % len(models)]
            space = get_search_space(model)
            config = space.sample()
            # Force small steps for quick testing/demo
            if 'epochs' not in config: config['epochs'] = 1
            if 'steps' in config: config['steps'] = 5

            task = tasks[i % len(tasks)]

            self.job_queue.append({
                "job_id": self.job_counter,
                "task": task,
                "model_name": model,
                "config": config
            })
            self.job_counter += 1

    def start(self):
        if self.running: return
        self.server = ThreadingHTTPServer((self.host, self.port), CoordinatorHandler)
        self.server.coordinator = self
        self.thread = threading.Thread(target=self.server.serve_forever)
        self.thread.daemon = True
        self.thread.start()
        self.running = True
        logger.info(f"Coordinator started on {self.host}:{self.port}")

    def stop(self):
        if self.server:
            self.server.shutdown()
            self.server.server_close()
        self.running = False
        logger.info("Coordinator stopped")

    def get_job(self) -> Optional[Dict]:
        with self.lock:
            if not self.job_queue:
                self._populate_initial_jobs() # Replenish

            if self.job_queue:
                return self.job_queue.pop(0)
            return None

    def submit_result(self, result: Dict):
        with self.lock:
            self.jobs_completed += 1
        job_id = result.get('job_id')
        acc = result.get('accuracy', 0.0)
        logger.info(f"Job {job_id} completed. Acc: {acc:.4f}")
        # Here we would feed back into the evolutionary algo

    def register_node(self, client_id: str):
        with self.lock:
            self.nodes.add(client_id)

from bioplausible.p2p.state import load_state, save_state

class Worker:
    def __init__(self, coordinator_url: str, client_id: str = None):
        self.coordinator_url = coordinator_url.rstrip('/')
        self.client_id = client_id or str(uuid.uuid4())[:8]
        self.running = False

        # Load state
        state = load_state()
        self.points = state.get('points', 0)
        self.jobs_done = state.get('jobs_done', 0)

        self.current_status = "Idle"

        # Signals/Callbacks
        self.on_status_change = None # func(status, points, jobs)
        self.on_log = None # func(msg)

    def log(self, msg):
        logger.info(msg)
        if self.on_log:
            self.on_log(msg)

    def start_loop(self):
        self.running = True
        self.thread = threading.Thread(target=self._loop)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        self.running = False

    def _loop(self):
        self.log(f"Worker {self.client_id} started. Connecting to {self.coordinator_url}...")

        # Register
        try:
            self._post('/register', {"client_id": self.client_id})
        except Exception as e:
            self.log(f"Failed to register: {e}")
            # Continue anyway, maybe transient

        while self.running:
            try:
                # 1. Get Job
                self._update_status("Fetching Job...")
                job = self._get('/get_job?client_id=' + self.client_id)

                if job and 'job_id' in job:
                    job_id = job['job_id']
                    task = job.get('task', 'shakespeare')
                    model_name = job.get('model_name')
                    config = job.get('config')

                    self.log(f"Got Job {job_id}: {model_name} on {task}")
                    self._update_status(f"Running Job {job_id}...")

                    # 2. Run Job
                    result_metrics = self._run_job(job_id, task, model_name, config)

                    if result_metrics:
                        # 3. Submit Result
                        self._update_status("Submitting Result...")
                        payload = {
                            "job_id": job_id,
                            "client_id": self.client_id,
                            **result_metrics
                        }
                        self._post('/submit_result', payload)

                        self.points += 10 # Gamification
                        self.jobs_done += 1
                        save_state(self.points, self.jobs_done) # Persist
                        self.log(f"Job {job_id} submitted! Points: {self.points}")
                    else:
                        self.log(f"Job {job_id} failed locally.")

                else:
                    self._update_status("Idle (No jobs)")
                    time.sleep(5)

            except URLError:
                self._update_status("Connection Failed")
                self.log("Cannot connect to coordinator. Retrying in 10s...")
                time.sleep(10)
            except Exception as e:
                self.log(f"Worker Error: {e}")
                time.sleep(5)

            time.sleep(1) # Breath

        self._update_status("Stopped")

    def _run_job(self, job_id, task, model_name, config) -> Optional[Dict]:
        # Remote jobs are stored in the main DB so they appear in visualizations
        # Use default storage path: "results/hyperopt.db"
        return run_single_trial_task(
            task=task,
            model_name=model_name,
            config=config,
            storage_path="results/hyperopt.db",
            job_id=job_id
        )

    def _update_status(self, status):
        self.current_status = status
        if self.on_status_change:
            self.on_status_change(status, self.points, self.jobs_done)

    def _get(self, endpoint) -> Dict:
        url = f"{self.coordinator_url}{endpoint}"
        req = Request(url)
        with urlopen(req, timeout=10) as response:
            return json.loads(response.read().decode())

    def _post(self, endpoint, data: Dict) -> Dict:
        url = f"{self.coordinator_url}{endpoint}"
        req = Request(url, data=json.dumps(data).encode(), headers={'Content-Type': 'application/json'})
        with urlopen(req, timeout=10) as response:
            return json.loads(response.read().decode())
