""" 最佳适应算法 """
import numpy as np

from static_schedule.offline_scheduler import Scheduler

class BestFitScheduler(Scheduler):
    def __init__(self, input_path: str, pod_affinity, node_affinity):
        super().__init__(input_path, pod_affinity, node_affinity)
        self.scheduler_name = "best_fit_scheduler"

    def schedule(self) -> np.ndarray:
        return self.schedule_fit("best")

    def schedule_without_gpu(self) -> np.ndarray:
        return self.schedule_fit("best", preserve_gpu_nodes=False)
