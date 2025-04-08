""" 最佳适应算法 """
import copy
import numpy as np

from static_schedule.offline_scheduler import Scheduler
from util.logger import init_logger,logger

class BestFitScheduler(Scheduler):
    def __init__(self, input_path: str,pod_affinity,node_affinity):
        super().__init__(input_path,pod_affinity,node_affinity)
        self.scheduler_name = "best_fit_scheduler"

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
        plan = np.zeros(len(self.pods), dtype=int)
        for i, pod in enumerate(pods):
            place_node = None
            max_value = 0
            if pod.gpu == 0: # 优先从普通节点找
                for j in normal_nodes_idx:
                    node = nodes[j]
                    v = node.max_usage(pod)
                    if node >= pod and 1 >= v > max_value:  # pod占node剩余资源越多
                        max_value = v
                        place_node = j
            if place_node is not None: # 找到了，直接返回
                plan[i] = place_node
                nodes[place_node] = nodes[place_node] - pod
                continue
            for j in gpu_nodes_idx: # 从gpu节点找
                node = nodes[j]
                v = node.max_usage(pod)
                if node >= pod and 1 >= v > max_value:  # node资源比pod多很多的节点
                    max_value = v
                    place_node = j
            if place_node is None:
                logger.warn("fail to place pods")
                return None
            plan[i] = place_node
            nodes[place_node] = nodes[place_node] - pod
        self.plan = plan
        return plan

    def schedule_without_gpu(self) -> np.ndarray:
        pods = self.pods
        nodes = copy.deepcopy(self.nodes)
        plan = np.zeros(len(self.pods), dtype=int)
        for i, pod in enumerate(pods):
            place_node = None
            max_value = 0
            for j, node in enumerate(nodes):
                v = node.max_usage(pod)
                if 1 >= v > max_value:  # node资源比pod多很多的节点
                    max_value = v
                    place_node = j
            if place_node is None:
                logger.warn("fail to place pods")
                return None
            plan[i] = place_node
            nodes[place_node] = nodes[place_node] - pod
        self.plan = plan
        return plan
