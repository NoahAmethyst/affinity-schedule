""" 最差适应算法 """
import numpy as np

from static_schedule.offline_scheduler import Scheduler


class WorstFitScheduler(Scheduler):
    def __init__(self, input_path: str, pod_affinity, node_affinity):
        super().__init__(input_path, pod_affinity, node_affinity)
        self.scheduler_name = "worst_fit_scheduler"

    def schedule(self) -> np.ndarray:
        """ 考虑gpu优先 """
        return self.schedule_fit("worst")

    def schedule_without_gpu(self) -> np.ndarray:
        """ 不考虑gpu """
        return self.schedule_fit("worst", preserve_gpu_nodes=False)


if __name__ == '__main__':
    scheduler = WorstFitScheduler("/offline-scheduler/data/input")

    ### schedule
    plan = scheduler.schedule()

    ### check
    scheduler.check_and_output(scheduler, "W:/agents/data/output", plan)
