""" 首次适应算法 """
import numpy as np
from static_schedule.offline_scheduler import Scheduler


class FirstFitScheduler(Scheduler):
    def __init__(self, input_path: str, pod_affinity, node_affinity):
        super().__init__(input_path, pod_affinity, node_affinity)
        self.scheduler_name = "first_fit_scheduler"

    def schedule(self) -> np.ndarray:
        return self.schedule_fit("first")

    def schedule_without_gpu(self) -> np.ndarray:
        return self.schedule_fit("first", preserve_gpu_nodes=False)


if __name__ == '__main__':
    scheduler = FirstFitScheduler("/offline-scheduler/data/input")

    ### schedule
    plan = scheduler.schedule()

    ### check
    scheduler.check_and_output(scheduler, "W:/agents/data/output", plan)
