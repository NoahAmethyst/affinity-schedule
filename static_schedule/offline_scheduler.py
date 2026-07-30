# 静态调度框架
import copy
from numbers import Integral
import os
import random
from typing import Any

import pandas as pd
import yaml
import numpy as np

from affinity.resource import BaseNode, BasePod, BaseObject, SingleSchedulerPlan
from util.logger import init_logger, logger

init_logger()


class SchedulingError(RuntimeError):
    """调度输入或调度结果不满足约束。"""


class Scheduler:
    # podIdx2name: [str]
    # nodeIdx2name: [str]
    # podName2idx: dict[str, int]
    # nodeName2idx: dict[str, int]

    affinity_weight = 1
    avg_usage_weight = 1
    var_usage_weight = 1

    def __init__(self, input_path: str, pod_affinity, node_affinity):
        self.pods: [BasePod] = []
        self.nodes: [BaseNode] = []
        self.pod_affinity = pod_affinity
        self.node_affinity = node_affinity
        self.plan = None
        self.pod_yaml = {}
        self.read_input(input_path)
        self.scheduler_name = ""

    def schedule(self) -> [int]:
        raise NotImplementedError

    def affinity(self, plan: [int]) -> int:
        """ 计算节点直接亲和性 """
        res = 0
        affinity_pod = self.pod_affinity
        ### calc affinity between pods
        for pod1_idx in range(len(self.pods)):
            for pod2_idx in range(pod1_idx + 1, len(self.pods)):
                # affinity
                if plan[pod1_idx] == plan[pod2_idx]:
                    res += affinity_pod[pod1_idx, pod2_idx]
                if plan[pod1_idx] != plan[pod2_idx]:
                    res -= affinity_pod[pod1_idx, pod2_idx]
        return res

    def calc_cost(self, plan: [int]) -> float:
        min_usage, max_usage, usage = self.usage(plan)

        affinity_cost = self.affinity(plan)
        avg_usage_cost = np.average(max_usage)
        var_usage_cost = np.var(max_usage)
        logger.info(f'affinity: {affinity_cost}, avg: {avg_usage_cost}, var: {var_usage_cost}')
        return (
            var_usage_cost * self.var_usage_weight
            - affinity_cost * self.affinity_weight
            - avg_usage_cost * self.avg_usage_weight
        )

    def usage(self, plan: np.ndarray) -> ([float], [float], [BaseObject]):
        """输出每个节点的 资源最低利用率 最高利用率 和 每个资源的利用率"""
        plan = self.validate_plan_indices(plan)
        occupied = [None for _ in range(len(self.nodes))]
        for pod_idx, node_idx in enumerate(plan):
            if occupied[node_idx] is not None:
                tmp = occupied[node_idx]
            else:
                tmp = BaseObject("", 0, 0, 0, 0)
            tmp += self.pods[pod_idx]
            occupied[node_idx] = tmp

        min_usage = [0 for _ in range(len(occupied))]
        max_usage = [0 for _ in range(len(occupied))]
        usage = [BaseObject("", 0, 0, 0, 0) for _ in range(len(occupied))]
        for node_idx, u in enumerate(occupied):
            if u is None:
                min_usage[node_idx] = 0
                continue
            min_usage[node_idx] = self.nodes[node_idx].min_usage(u)
            max_usage[node_idx] = self.nodes[node_idx].max_usage(u)
            usage[node_idx] = self.nodes[node_idx].usage(u)
        return min_usage, max_usage, usage

    def used(self, plan: np.ndarray) -> [BaseObject]:
        """计算每个节点已经占用的资源。"""
        plan = self.validate_plan_indices(plan)
        used = [BaseObject() for _ in range(len(self.nodes))]

        for pod_idx, node_idx in enumerate(plan):
            used[node_idx] += self.pods[pod_idx]

        return used

    def read_pod_yamls(self, pods_dir: str) -> dict[str, Any]:
        entries = os.listdir(pods_dir)
        pods = {}
        for entry in entries:
            path = os.path.join(pods_dir, entry)
            if not os.path.isfile(path) or not entry.endswith((".yaml", ".yml")):
                continue
            with open(path, 'r', encoding="utf-8") as file:
                data = yaml.safe_load(file)
            try:
                pod_name = data["metadata"]["name"]
            except (KeyError, TypeError) as exc:
                raise SchedulingError(f"YAML 文件缺少 metadata.name: {path}") from exc
            if pod_name in pods:
                raise SchedulingError(f"YAML 目录包含重复 Pod: {pod_name}")
            pods[pod_name] = data
        self.pod_yaml = pods
        return pods

    def save_pod_yamls(self, dir: str):
        if self.plan is None:
            raise SchedulingError("请先生成调度计划")
        if not self.pod_yaml:
            raise SchedulingError("请先读取 Pod YAML")

        plan = self.validate_plan_indices(self.plan)
        os.makedirs(dir, exist_ok=True)
        for pod_idx, pod in enumerate(self.pods):
            if pod.name not in self.pod_yaml:
                raise SchedulingError(f"缺少 Pod YAML: {pod.name}")
            one_yaml = copy.deepcopy(self.pod_yaml[pod.name])
            node_name = self.nodes[plan[pod_idx]].name
            if one_yaml.get("kind") == "Deployment":
                one_yaml.setdefault("spec", {}).setdefault("template", {}).setdefault("spec", {})[
                    "nodeName"
                ] = node_name
            else:
                one_yaml.setdefault("spec", {})["nodeName"] = node_name
            with open(os.path.join(dir, pod.name + '.yaml'), 'w', encoding="utf-8") as file:
                yaml.safe_dump(one_yaml, file, default_flow_style=False, allow_unicode=True)

    def read_input(self, input_path: str):
        """ 读取输入数据 """
        ### read pods
        pods, pod2idx = read_pods_csv(f"{input_path}/pods.csv")
        self.pods = pods
        self.podName2idx = pod2idx

        ### read nodes
        nodes, node2idx = read_nodes_csv(f"{input_path}/nodes.csv")
        self.nodes = nodes
        self.nodeName2idx = node2idx

        ### read affinity
        if self.node_affinity is None or self.pod_affinity is None:
            logger.warning(f'node and pod affinity are None.Load affinity from {input_path}')
            self.pod_affinity = np.load(f"{input_path}/pod_affinity.npy")
            self.node_affinity = np.load(f"{input_path}/node_affinity.npy")
        self._validate_affinity_shapes()
        ### shuffle
        self.shuffle()

    def shuffle(self):
        """ 将输入的pod顺序打乱 """
        index = list(range(len(self.pods)))  # index[i]表示原位置
        random.Random(42).shuffle(index)
        pods = [self.pods[i] for i in index]
        self.pods = pods
        self.pod_affinity = self.pod_affinity[np.ix_(index, index)]
        self.node_affinity = self.node_affinity[index, :]
        self.podName2idx = {pod.name: idx for idx, pod in enumerate(self.pods)}
        self.nodeName2idx = {node.name: idx for idx, node in enumerate(self.nodes)}
        self.podIdx2name = [pod.name for pod in self.pods]
        self.nodeIdx2name = [node.name for node in self.nodes]

    def _validate_affinity_shapes(self):
        pod_count = len(self.pods)
        node_count = len(self.nodes)
        if not isinstance(self.pod_affinity, np.ndarray):
            raise SchedulingError("pod_affinity 必须是 numpy 数组")
        if not isinstance(self.node_affinity, np.ndarray):
            raise SchedulingError("node_affinity 必须是 numpy 数组")
        if self.pod_affinity.shape != (pod_count, pod_count):
            raise SchedulingError(
                f"pod_affinity 尺寸错误: 期望 {(pod_count, pod_count)}，实际 {self.pod_affinity.shape}"
            )
        if self.node_affinity.shape != (pod_count, node_count):
            raise SchedulingError(
                f"node_affinity 尺寸错误: 期望 {(pod_count, node_count)}，实际 {self.node_affinity.shape}"
            )

    def validate_plan_indices(self, plan) -> np.ndarray:
        if plan is None:
            raise SchedulingError("调度器未能生成计划")
        values = list(plan)
        if len(values) != len(self.pods):
            raise SchedulingError(
                f"计划长度 {len(values)} 与 Pod 数量 {len(self.pods)} 不一致"
            )
        normalized = np.empty(len(values), dtype=np.int64)
        for pod_idx, node_idx in enumerate(values):
            if isinstance(node_idx, bool) or not isinstance(node_idx, Integral):
                raise SchedulingError(
                    f"Pod {self.pods[pod_idx].name} 的节点索引不是整数: {node_idx!r}"
                )
            if not 0 <= int(node_idx) < len(self.nodes):
                raise SchedulingError(
                    f"Pod {self.pods[pod_idx].name} 的节点索引越界: {node_idx}"
                )
            normalized[pod_idx] = int(node_idx)
        return normalized

    def can_place(self, pod_idx: int, node_idx: int, nodes: list[BaseNode]) -> bool:
        return (
            self.node_affinity[pod_idx, node_idx] > 0
            and nodes[node_idx] >= self.pods[pod_idx]
        )

    def _candidate_node_indices(
        self,
        pod_idx: int,
        nodes: list[BaseNode],
        preserve_gpu_nodes: bool,
    ) -> list[int]:
        pod = self.pods[pod_idx]
        if not preserve_gpu_nodes:
            ordered = list(range(len(nodes)))
        elif pod.gpu > 0:
            ordered = [idx for idx, node in enumerate(nodes) if node.gpu > 0]
        else:
            normal = [idx for idx, node in enumerate(nodes) if node.gpu == 0]
            gpu = [idx for idx, node in enumerate(nodes) if node.gpu > 0]
            ordered = normal + gpu
        return [idx for idx in ordered if self.can_place(pod_idx, idx, nodes)]

    def schedule_fit(self, strategy: str, preserve_gpu_nodes: bool = True) -> np.ndarray:
        if strategy not in {"first", "best", "worst"}:
            raise ValueError(f"未知适应策略: {strategy}")
        nodes = copy.deepcopy(self.nodes)
        plan = np.empty(len(self.pods), dtype=np.int64)

        for pod_idx, pod in enumerate(self.pods):
            candidates = self._candidate_node_indices(
                pod_idx,
                nodes,
                preserve_gpu_nodes,
            )
            if not candidates:
                raise SchedulingError(f"Pod {pod.name} 没有满足资源和亲和性约束的节点")
            if strategy == "first":
                node_idx = candidates[0]
            elif strategy == "best":
                node_idx = max(candidates, key=lambda idx: nodes[idx].max_usage(pod))
            else:
                node_idx = min(candidates, key=lambda idx: nodes[idx].max_usage(pod))
            plan[pod_idx] = node_idx
            nodes[node_idx] = nodes[node_idx] - pod

        self.plan = plan
        return plan

    def get_node_num(self):
        return len(self.nodes)

    def get_pod_num(self):
        return len(self.pods)

    def check(self, plan: np.ndarray) -> bool:
        """ 检查放置方案是否合法 """
        try:
            plan = self.validate_plan_indices(plan)
            used = self.used(plan)
        except SchedulingError as exc:
            logger.error(str(exc))
            return False
        for u, node in zip(used, self.nodes):
            if not (node - u).is_not_empty():
                return False
        for pod_idx, node_idx in enumerate(plan):
            if self.node_affinity[pod_idx, node_idx] <= 0:
                return False
        return True

    def save_plan(self, save_path: str, plan: []):
        """保存调度结果"""
        plan = self.validate_plan_indices(plan)
        if not self.check(plan):
            raise SchedulingError("计划未通过资源或亲和性约束检查")
        entries = [
            [self.pods[pod_idx].name, self.nodes[node_idx].name]
            for pod_idx, node_idx in enumerate(plan)
        ]
        os.makedirs(save_path, exist_ok=True)
        df = pd.DataFrame(entries, columns=["name", "node"])
        df.to_csv(os.path.join(save_path, f'{self.scheduler_name}.csv'), index=False)

    @classmethod
    def check_and_output(cls, scheduler, save_path: str, plan: [int]):
        ### check
        result = scheduler.check(plan)
        if not result:
            logger.info('check failed')
            return

        ### 计算cost
        cost = scheduler.calc_cost(plan)
        logger.info(f'cost: {cost}')

        ### 计算利用率
        min_usage, max_usage, usage = scheduler.usage(plan)
        for i, v in enumerate(zip(min_usage, max_usage, usage)):
            min_u, max_u, u = v
            logger.info(f'node({i}): min_usage:{min_u}, max_usage:{max_u}, usage:[{u.to_string()}]')

        ### 保存结果
        scheduler.save_plan(save_path, plan)

    def check_and_gen(self, scheduler, plan: [int]) -> list[SingleSchedulerPlan] | None:
        """Validate a scheduling plan and generate execution details."""
        # Early return if plan is invalid
        if not scheduler.check(plan):
            logger.info('Plan validation failed')
            return None

        plan = self.validate_plan_indices(plan)

        # Log cost information
        cost = scheduler.calc_cost(plan)
        logger.info(f'Total cost: {cost}')

        # Generate plan objects with validation
        return [
            SingleSchedulerPlan(self.pods[pod].name, self.nodes[node].name)
            for pod, node in enumerate(plan)
        ]



def read_pods_csv(path: str) -> ([], {}):
    data = _read_resource_csv(path, BasePod.get_columns())
    pods = []
    pod2idx = {}
    for idx, (_, row) in enumerate(data.iterrows()):
        pod = BasePod.from_dataframe(row)
        if pod.name in pod2idx:
            raise SchedulingError(f"Pod 资源文件包含重复名称: {pod.name}")
        pods.append(pod)
        pod2idx[pod.name] = idx
    return pods, pod2idx


def read_nodes_csv(path: str) -> ([], {}):
    data = _read_resource_csv(path, BaseNode.get_columns())
    nodes = []
    node2idx = {}
    for idx, (_, row) in enumerate(data.iterrows()):
        node = BaseNode.from_dataframe(row)
        if node.name in node2idx:
            raise SchedulingError(f"Node 资源文件包含重复名称: {node.name}")
        nodes.append(node)
        node2idx[node.name] = idx
    return nodes, node2idx


def _read_resource_csv(path: str, required_columns: list[str]) -> pd.DataFrame:
    try:
        data = pd.read_csv(path)
    except (OSError, pd.errors.ParserError) as exc:
        raise SchedulingError(f"无法读取资源文件 {path}: {exc}") from exc
    missing = [column for column in required_columns if column not in data.columns]
    if missing:
        raise SchedulingError(f"资源文件 {path} 缺少列: {', '.join(missing)}")
    return data


if __name__ == '__main__':
    scheduler = Scheduler("/offline-scheduler/data/input")
    scheduler.shuffle()
