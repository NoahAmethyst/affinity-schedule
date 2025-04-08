""" 首次适应算法 """
import copy
import numpy as np
from static_schedule.offline_scheduler import Scheduler
from util.logger import logger


class FirstFitScheduler(Scheduler):
    def __init__(self, input_path: str,pod_affinity,node_affinity):
        super().__init__(input_path,pod_affinity,node_affinity)
        self.scheduler_name = "first_fit_scheduler"

    def schedule(self) -> np.ndarray:
        pods = self.pods
        nodes = copy.deepcopy(self.nodes)
        gpu_nodes_idx = []
        normal_nodes_idx = []
        for idx, node in enumerate(nodes):
            if node.gpu > 0:
                gpu_nodes_idx.append(idx)
            else:
                normal_nodes_idx.append(idx)
        plan = np.zeros(len(self.pods), dtype=np.int32)

        for i, pod in enumerate(pods):
            place_node = None
            if pod.gpu == 0: # 优先从普通节点选择
                for j in normal_nodes_idx:
                    node = nodes[j]
                    if node >= pod:
                        place_node = j
                        break
            if place_node is not None: # 找到了直接返回
                plan[i] = place_node
                nodes[place_node] = nodes[place_node] - pod
                continue
            for j in gpu_nodes_idx: # 从gpu节点找
                node = nodes[j]
                if node >= pod:
                    place_node = j
                    break
            if place_node is None:
                logger.warn('fail to place pods')
                return None
            plan[i] = place_node
            nodes[place_node] = nodes[place_node] - pod
        self.plan = plan
        return plan

    def schedule_without_gpu(self) -> np.ndarray:
        pods = self.pods
        nodes = copy.deepcopy(self.nodes)
        plan = np.zeros(len(self.pods), dtype=np.int32)
        for i, pod in enumerate(pods):
            place_node = None
            for j, node in enumerate(nodes):
                remain = node - pod
                if remain.is_not_empty():
                    place_node = j
                    plan[i] = j
                    break
            if place_node is None:
                logger.warn('fail to place pods')
                return None
            nodes[place_node] = nodes[place_node] - pod
        self.plan = plan
        return plan


if __name__ == '__main__':
    scheduler = FirstFitScheduler("/offline-scheduler/data/input")

    ### schedule
    plan = scheduler.schedule()

    ### check
    scheduler.check_and_output(scheduler, "W:/agents/data/output", plan)
