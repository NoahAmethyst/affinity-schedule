""" 多阶段调度 """
import os
import matplotlib
import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from scipy.optimize import linear_sum_assignment

from affinity.resource import BaseNode, BasePod, BaseObject
from static_schedule.offline_scheduler import Scheduler, SchedulingError
import copy

from util.logger import logger


class MultiStageScheduler(Scheduler):
    ### fine_tuning 节点利用率最大差值
    fine_tuning_max_diff = 0.1

    def __init__(self, input_path: str, pod_affinity, node_affinity):
        super().__init__(input_path, pod_affinity, node_affinity)
        self.scheduler_name = "multi_stage_scheduler"
        self.enable_drawing = False

    def schedule(self, enable_draw=False) -> [int]:
        if enable_draw:
            self.draw_init()
        ### 聚类
        clusters, cluster_sum, affinity = self.gpu_cluster()

        ### 映射到Node
        clusters = self.first_fit_mapper(clusters)
        # clusters = self.mapper(clusters)
        ### 超过节点资源的重新分配
        clusters = self.fine_tuning(clusters)

        ### 获得结果
        plan = self.cluster_to_plan(clusters)
        if not self.check(plan):
            raise SchedulingError("多阶段调度结果未通过资源或亲和性约束检查")
        self.plan = np.asarray(plan, dtype=np.int64)
        return self.plan

    def gpu_cluster(self):
        if not self.nodes:
            raise SchedulingError("没有可用节点")
        gpu_node_num = sum(node.gpu > 0 for node in self.nodes)
        max_obj = BaseNode(
            "",
            max(node.cpu for node in self.nodes),
            max(node.mem for node in self.nodes),
            max(node.gpu for node in self.nodes),
            max(node.disk for node in self.nodes),
            max(node.net for node in self.nodes),
        )

        max_gpu_pod_per_node = len(self.pods)
        max_normal_pod_per_node = 17

        ### 先根据gpu进行聚类
        gpu_affinity = np.copy(self.pod_affinity)
        exclude = []
        gpu_pod_num = 0
        for idx, pod in enumerate(self.pods):
            if pod.gpu == 0:
                exclude.append(True)
            else:
                exclude.append(False)
                gpu_pod_num += 1
        for i in range(len(self.pods)):
            for j in range(len(self.pods)):
                if exclude[i] or exclude[j]:
                    gpu_affinity[i][j] = -gpu_affinity[i][j]
        normal_pod_num = len(self.pods) - gpu_pod_num
        gpu_cluster_target = min(gpu_node_num, gpu_pod_num)
        if gpu_pod_num and gpu_cluster_target == 0:
            raise SchedulingError("存在 GPU Pod，但没有 GPU 节点")
        n_cluster = normal_pod_num + gpu_cluster_target
        clusters, cluster_sum, affinity = self.cluster(
            n_cluster,
            gpu_affinity,
            copy.deepcopy(self.pods),
            max_obj,
            draw=self.draw_merge,
            max_num=max_gpu_pod_per_node,
            exclude=exclude,
        )
        ### 再进行全体的聚类
        for i in range(len(clusters)):
            for j in range(len(clusters)):
                if affinity[i, j] < 0:
                    affinity[i, j] = -affinity[i, j]
        n_cluster = len(self.nodes)

        clusters, cluster_sum, affinity = self.cluster(
            n_cluster=n_cluster,
            affinity=affinity,
            cluster_sum=cluster_sum,
            max_obj=max_obj,
            draw=self.draw_merge,
            max_num=max_normal_pod_per_node,
            exclude=None,
            clusters=clusters,
        )
        # print(len(clusters))
        # print([pod.__dict__ for pod in cluster_sum])

        return clusters, cluster_sum, affinity

    def first_fit_mapper(self, clusters: [[int]]):
        """ [首次匹配算法]将聚类结果匹配到node """
        if len(clusters) != len(self.nodes):
            raise SchedulingError(
                f"簇数量 {len(clusters)} 与节点数量 {len(self.nodes)} 不一致"
            )
        mapped = [None] * len(self.nodes)
        available_nodes = set(range(len(self.nodes)))
        ordered_clusters = sorted(
            clusters,
            key=lambda cluster: any(self.pods[pod_idx].gpu > 0 for pod_idx in cluster),
            reverse=True,
        )
        for cluster in ordered_clusters:
            used = BasePod()
            for pod_idx in cluster:
                used += self.pods[pod_idx]
            candidates = [
                node_idx
                for node_idx in sorted(available_nodes)
                if self.nodes[node_idx] >= used
                and all(self.node_affinity[pod_idx, node_idx] > 0 for pod_idx in cluster)
            ]
            if not candidates:
                pod_names = [self.pods[pod_idx].name for pod_idx in cluster]
                raise SchedulingError(f"簇没有可用节点: {pod_names}")
            node_idx = min(
                candidates,
                key=lambda candidate: self.nodes[candidate].max_usage(used),
            )
            mapped[node_idx] = cluster
            available_nodes.remove(node_idx)
        return mapped

    def mapper(self, clusters: [[int]]):
        """ [节点匹配算法] 建模成指派问题，使用匈牙利算法求解 """
        usage = np.zeros(shape=(len(clusters), len(clusters)))
        for c_idx, cluster in enumerate(clusters):
            for n_idx, node in enumerate(self.nodes):
                used = BasePod()
                for pod in cluster:
                    used += self.pods[pod]
                max_usage = node.max_usage(used)
                usage[c_idx, n_idx] = max_usage
        ### 使用匈牙利算法求解
        row_ind, col_ind = linear_sum_assignment(usage)
        ### 最小cost
        # min_cost = usage[row_ind, col_ind].sum()
        result = [None for i in range(len(clusters))]
        for r, c in zip(row_ind, col_ind):
            result[c] = clusters[r]
        return result

    def fine_tuning(self, clusters: [[int]]):
        """ 基于贪心算法的调整策略  """
        def calculate_used(current_clusters):
            result = []
            for cluster in current_clusters:
                total = BasePod()
                for pod_idx in cluster:
                    total += self.pods[pod_idx]
                result.append(total)
            return result

        current = copy.deepcopy(clusters)
        while True:
            used = calculate_used(current)
            usage = np.asarray(
                [
                    self.nodes[node_idx].max_usage(node_used)
                    for node_idx, node_used in enumerate(used)
                ],
                dtype=float,
            )
            current_cost = self.calc_cost(self.cluster_to_plan(current))
            improved = False

            for from_idx in np.argsort(usage)[::-1]:
                for to_idx in np.argsort(usage):
                    if from_idx == to_idx:
                        continue
                    if usage[from_idx] - usage[to_idx] < self.fine_tuning_max_diff:
                        continue
                    for pod_idx in list(current[from_idx]):
                        if self.node_affinity[pod_idx, to_idx] <= 0:
                            continue
                        destination_used = used[to_idx] + self.pods[pod_idx]
                        if not self.nodes[to_idx] >= destination_used:
                            continue
                        candidate = copy.deepcopy(current)
                        candidate[from_idx].remove(pod_idx)
                        candidate[to_idx].append(pod_idx)
                        candidate_cost = self.calc_cost(self.cluster_to_plan(candidate))
                        if candidate_cost < current_cost:
                            logger.info(
                                f"fine tuning pod {pod_idx} from node {from_idx} to node {to_idx}"
                            )
                            current = candidate
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                return current

    def cluster_to_plan(self, clusters: [[int]]):
        """ 类别模式转成调度计划 """
        plan = [0 for i in range(len(self.pods))]
        for node, cluster in enumerate(clusters):
            for pod in cluster:
                plan[pod] = node
        return plan

    @classmethod
    def cluster(cls,
                n_cluster,
                affinity: np.ndarray,
                cluster_sum: [BasePod],
                max_obj: BaseNode,
                draw=None,
                max_num=100,
                exclude=None,
                clusters=None,
                ) -> tuple[list[list[int]], ndarray | ndarray]:
        """ 层次聚类 """

        def merge_cluster(
                clusters: [[int]],
                cluster_sum: [BaseObject],
                x: int, y: int) -> ([[int]], [BaseObject], bool):
            """ 合并y簇到x簇 """
            tmp = cluster_sum[x] + cluster_sum[y]
            if not max_obj >= tmp:
                return None, None, False
            if len(clusters[x]) + len(clusters[y]) > max_num:
                return None, None, False
            cluster_sum[x] = tmp
            del cluster_sum[y]
            if exclude is not None:
                del exclude[y]
            clusters[x].extend(clusters[y])
            del clusters[y]
            return clusters, cluster_sum, True

        if cluster_sum is None:
            raise ValueError("cluster_sum 不能为空")
        if clusters is None:
            clusters = [[i] for i in range(len(cluster_sum))]
        affinity = copy.deepcopy(affinity)

        while len(clusters) > n_cluster:
            v = np.max(affinity)
            if v == 0:
                logger.warning('failed to cluster')
                break
            x, y = np.unravel_index(np.argmax(affinity), affinity.shape)
            # 是否 exclude
            if exclude is not None:
                if exclude[x] or exclude[y]:
                    affinity[x, y] = 0
                    affinity[y, x] = 0
                    continue
            # 确保 x < y
            if x > y:
                x, y = y, x
            c, cs, ok = merge_cluster(clusters, cluster_sum, x, y)
            if ok:
                if draw is not None:
                    draw(x, y)
                ### 聚合 x 和 y
                # 计算新簇与其他簇的距离（取 x 和 y 的平均值）
                new_line = (affinity[x, :] + affinity[y, :])
                new_line[x] = 0
                new_line[y] = 0
                new_line = np.delete(new_line, [y])
                # 删除 x 和 y 对应的行和列
                affinity = np.delete(affinity, [y], axis=0)
                affinity = np.delete(affinity, [y], axis=1)
                affinity[x, :] = new_line
                affinity[:, x] = new_line

                cluster_sum = cs
                clusters = c
            else:
                affinity[x, y] = 0
                affinity[y, x] = 0
        return clusters, cluster_sum, affinity

    def draw_init(self):
        """ 绘图数据初始化 """
        n_cluster = len(self.pods)

        self.linkage_matrix = []
        self.m = list(range(n_cluster))  # cluster_idx: draw_idx
        self.weight = [1 for i in range(n_cluster)]  # cluster num
        self.enable_drawing = True
        self.n = n_cluster

    def draw_merge(self, x, y, weight=None):
        """ 聚合x和y """
        if not self.enable_drawing:
            return
        if weight is None:
            weight = self.weight[x] + self.weight[y]
        draw_cluster = self.n
        self.n += 1
        self.linkage_matrix.append([self.m[x], self.m[y], float(weight), weight])
        self.m[x] = draw_cluster
        del (self.m[y])
        self.weight[x] = weight
        del (self.weight[y])

    def draw(self, save_path: str):
        # 绘制树状图
        while len(self.m) > 1:
            self.draw_merge(len(self.m) - 2, len(self.m) - 1, weight=20)
        logger.info(self.linkage_matrix)
        matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
        # matplotlib.rcParams['axes.unicode_minus'] = False  # 正常显示负号
        plt.figure(figsize=(8 * 2, 6 * 2))
        sch.dendrogram(self.linkage_matrix,
                       labels=[pod.name for pod in self.pods],  # 可选：数据点的标签
                       color_threshold=1.0)  # 可选：颜色阈值
        plt.title('基于亲和性的层次聚类过程', fontsize=36)
        plt.xlabel('智能体', fontsize=24)
        plt.ylabel('距离', fontsize=24)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=16)
        plt.show()
        plt.savefig(os.path.join(save_path, 'cluster.png'))


if __name__ == '__main__':
    input_dir = "/offline-scheduler/data/input"
    output_dir = "/offline-scheduler/data/output"

    scheduler = MultiStageScheduler(input_dir)

    ### schedule
    plan = scheduler.schedule(enable_draw=True)

    ### check
    scheduler.check_and_output(scheduler, output_dir, plan)

    ### draw
    scheduler.draw('W:/agents/data/others')
