
from PyQt6.QtCore import QThread, pyqtSignal, QMutex, QMutexLocker
import torch
import time

from .types import TrainingState
from .algorithms import AlgorithmWrapper

class TrainingWorker(QThread):
    """Background thread for training loop."""
    update_signal = pyqtSignal(str, TrainingState) # name, state
    
    def __init__(self, algorithms, data, device):
        super().__init__()
        self.algorithms = algorithms
        self.data = data
        self.device = device
        self.running = True
        self.mutex = QMutex()
        self.iteration = 0
        self.paused = False

    def run(self):
        while self.running:
            with QMutexLocker(self.mutex):
                if self.paused:
                    time.sleep(0.1)
                    continue
                
                # Check consistency
                if not self.algorithms:
                     time.sleep(0.1)
                     continue

                self.iteration += 1
                
                # Generate batch
                idx = torch.randint(0, len(self.data) - 65, (32,))
                x = torch.stack([self.data[i:i+64] for i in idx]).to(self.device)
                y = torch.stack([self.data[i+64] for i in idx]).to(self.device)
                
                # Step all algorithms
                for name, algo in self.algorithms.items():
                    state = algo.train_step(x, y, self.iteration)
                    self.update_signal.emit(name, state)
            
            # Avoid hogging CPU completely
            time.sleep(0.001)

    def stop(self):
        self.running = False
        self.wait()
    
    def pause(self):
        with QMutexLocker(self.mutex):
            self.paused = True
            
    def resume(self):
         with QMutexLocker(self.mutex):
            self.paused = False
