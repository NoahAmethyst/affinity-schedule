""" 最差适应算法 """
import copy
import math
import numpy as np

from static_schedule.offline_scheduler import Scheduler
from util.logger import logger


class WorstFitScheduler(Scheduler):
    def __init__(self, input_path: str,pod_affinity,node_affinity):
        super().__init__(input_path,pod_affinity,node_affinity)
        self.scheduler_name = "worst_fit_scheduler"

    def schedule(self) -> np.ndarray:
        """ 考虑gpu优先 """
        pods = self.pods
        nodes = copy.deepcopy(self.nodes)
        gpu_nodes_idx = []
        normal_nodes_idx = []
        for idx, node in enumerate(nodes):
            if node.gpu > 0:
                gpu_nodes_idx.append(idx)
            else:
                normal_nodes_idx.append(idx)
        plan = np.zeros(len(self.pods), dtype=int)

        for i, pod in enumerate(pods):
            place_node = None
            min_value = math.inf
            if pod.gpu == 0:  # 优先从普通节点选择
                for j in normal_nodes_idx:
                    node = nodes[j]
                    v = node.max_usage(pod)
                    if node >= pod and min_value > v and 1 >= v:  # node资源比pod多最少的节点
                        min_value = v
                        place_node = j
            if place_node is not None:  # 找到了，直接返回
                plan[i] = place_node
                nodes[place_node] = nodes[place_node] - pod
                continue
            for j in gpu_nodes_idx:  # 从gpu节点找
                node = nodes[j]
                v = node.max_usage(pod)
                if node >= pod and min_value > v and 1 >= v:  # node资源比pod多最少的节点
                    min_value = v
                    place_node = j
            if place_node is None:
                logger.warn('fail to place pods')
                return None
            plan[i] = place_node
            nodes[place_node] = nodes[place_node] - pod
        self.plan = plan
        return plan

    def schedule_without_gpu(self) -> np.ndarray:
        """ 不考虑gpu """
        pods = self.pods
        nodes = copy.deepcopy(self.nodes)
        plan = np.zeros(len(self.pods), dtype=int)

        for i, pod in enumerate(pods):
            place_node = None
            min_value = math.inf
            for j, node in enumerate(nodes):
                v = node.max_usage(pod)
                if min_value > v and 1 >= v:  # node资源比pod多最少的节点
                    min_value = v
                    place_node = j
            if place_node is None:
                logger.warn('fail to place pods')
                return None
            plan[i] = place_node
            nodes[place_node] = nodes[place_node] - pod
        self.plan = plan
        return plan


if __name__ == '__main__':
    scheduler = WorstFitScheduler("/offline-scheduler/data/input")

    ### schedule
    plan = scheduler.schedule()

    ### check
    scheduler.check_and_output(scheduler, "W:/agents/data/output", plan)
