import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
import torch

from affinity.resource import SingleSchedulerPlan
from dynamic_schedule import model
from util.logger import init_logger, logger


@dataclass(frozen=True)
class ScheduleFiles:
    node_resources: Path
    node_capacities: Path | None
    running_pods: Path
    tasks: Path
    pod_index: Path
    output: Path


class Node:
    def __init__(
        self,
        name: str,
        cpu_used: float,
        cpu_free: float,
        memory_used: float,
        memory_free: float,
        net_used: float,
        net_free: float,
        gpu_used: float = 0,
        gpu_free: float = 0,
        disk_used: float = 0,
        disk_free: float = 0,
    ):
        self.name = name
        self.cpu_used = float(cpu_used)
        self.cpu_free = float(cpu_free)
        self.memory_used = float(memory_used)
        self.memory_free = float(memory_free)
        self.net_used = float(net_used)
        self.net_free = float(net_free)
        self.gpu_used = float(gpu_used)
        self.gpu_free = float(gpu_free)
        self.disk_used = float(disk_used)
        self.disk_free = float(disk_free)
        self.agents: list[str] = []

    def set_running_agents(self, agents: list[str]):
        self.agents = agents

    def can_run(self, cpu: float, memory: float, gpu: float, disk: float) -> bool:
        return (
            self.cpu_free >= cpu
            and self.memory_free >= memory
            and self.gpu_free >= gpu
            and self.disk_free >= disk
        )

    def placement_usage(self, cpu: float, memory: float, gpu: float, disk: float) -> float:
        ratios = [
            _resource_ratio(self.cpu_used + cpu, self.cpu_used + self.cpu_free),
            _resource_ratio(
                self.memory_used + memory,
                self.memory_used + self.memory_free,
            ),
            _resource_ratio(self.gpu_used + gpu, self.gpu_used + self.gpu_free),
            _resource_ratio(self.disk_used + disk, self.disk_used + self.disk_free),
        ]
        return max(ratios)

    def place(self, cpu: float, memory: float, gpu: float, disk: float, name: str):
        if not self.can_run(cpu, memory, gpu, disk):
            raise ValueError(f"节点 {self.name} 资源不足，无法放置 {name}")
        self.cpu_used += cpu
        self.cpu_free -= cpu
        self.memory_used += memory
        self.memory_free -= memory
        self.gpu_used += gpu
        self.gpu_free -= gpu
        self.disk_used += disk
        self.disk_free -= disk
        self.agents.append(name)

    def __str__(self):
        agent_str = ", ".join(self.agents) if self.agents else "None"
        return (
            f"Node Name: {self.name}\n"
            f"CPU Used: {self.cpu_used} ({_percent(self.cpu_used, self.cpu_free):.2f}%)\n"
            f"CPU Free: {self.cpu_free}\n"
            f"Memory Used: {self.memory_used} "
            f"({_percent(self.memory_used, self.memory_free):.2f}%)\n"
            f"Memory Free: {self.memory_free}\n"
            f"Net Used: {self.net_used} ({_percent(self.net_used, self.net_free):.2f}%)\n"
            f"Net Free: {self.net_free}\n"
            f"Running Agents: {agent_str}"
        )


def _percent(used: float, free: float) -> float:
    total = used + free
    return used / total * 100 if total > 0 else 0.0


def _resource_ratio(used: float, total: float) -> float:
    if total > 0:
        return used / total
    return float("inf") if used > 0 else 0.0


def new_nodes(
    resource_file: str | Path,
    pod_file: str | Path,
    capacity_file: str | Path | None = None,
) -> dict[str, Node]:
    nodes: dict[str, Node] = {}
    with open(resource_file, "r", newline="", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)
        for line_number, row in enumerate(reader, start=2):
            if len(row) < 7:
                raise ValueError(f"节点资源文件第 {line_number} 行字段不足")
            optional = row[7:11] + ["0"] * max(0, 4 - len(row[7:11]))
            try:
                node = Node(row[0], *row[1:7], *optional[:4])
            except ValueError as exc:
                raise ValueError(f"节点资源文件第 {line_number} 行包含非法数值") from exc
            if node.name in nodes:
                raise ValueError(f"节点资源文件包含重复节点: {node.name}")
            nodes[node.name] = node
    if not nodes:
        raise ValueError("节点资源文件中没有节点")

    if capacity_file is not None and Path(capacity_file).exists():
        with open(capacity_file, "r", newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            required_columns = {"name", "gpu", "disk"}
            missing = required_columns.difference(reader.fieldnames or [])
            if missing:
                raise ValueError(f"节点容量文件缺少列: {', '.join(sorted(missing))}")
            for line_number, row in enumerate(reader, start=2):
                node_name = row["name"]
                if node_name not in nodes:
                    continue
                try:
                    nodes[node_name].gpu_free = float(row["gpu"])
                    nodes[node_name].disk_free = float(row["disk"])
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"节点容量文件第 {line_number} 行包含非法数值") from exc

    with open(pod_file, "r", newline="", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)
        for line_number, row in enumerate(reader, start=2):
            if len(row) < 2:
                raise ValueError(f"运行 Pod 文件第 {line_number} 行字段不足")
            node_name = row[0]
            if node_name not in nodes:
                raise ValueError(
                    f"运行 Pod 文件第 {line_number} 行引用了不存在的节点: {node_name}"
                )
            agents = [agent for agent in row[1].split(",") if agent]
            nodes[node_name].set_running_agents(agents)
    return nodes


def get_model_input(
    feasible: list[bool],
    affinity: list[float],
    resource_usage: list[float],
    model_node_count: int,
) -> torch.FloatTensor:
    actual_node_count = len(feasible)
    if not (actual_node_count == len(affinity) == len(resource_usage)):
        raise ValueError("模型输入特征长度不一致")
    if actual_node_count > model_node_count:
        raise ValueError(
            f"模型最多支持 {model_node_count} 个节点，当前输入包含 {actual_node_count} 个节点"
        )

    padding = model_node_count - actual_node_count
    features = [
        [1.0 if value else 0.0 for value in feasible] + [0.0] * padding,
        affinity + [0.0] * padding,
        resource_usage + [0.0] * padding,
    ]
    return torch.tensor(features, dtype=torch.float32).transpose(0, 1).flatten()


def get_resource_usage(
    nodes: dict[str, Node],
    node_names: list[str],
    cpu: float,
    memory: float,
    gpu: float,
    disk: float,
) -> list[float]:
    return [
        nodes[node_name].placement_usage(cpu, memory, gpu, disk)
        if nodes[node_name].can_run(cpu, memory, gpu, disk)
        else 0.0
        for node_name in node_names
    ]


def get_affinity_score(
    nodes: dict[str, Node],
    node_names: list[str],
    agent: str,
    affinity: np.ndarray,
    pod_index: dict[str, int],
) -> list[float]:
    if agent not in pod_index:
        raise ValueError(f"亲和性索引中不存在 Pod: {agent}")
    source_index = pod_index[agent]
    scores = []
    for node_name in node_names:
        score = 0.0
        for pod in nodes[node_name].agents:
            if pod not in pod_index:
                raise ValueError(f"亲和性索引中不存在正在运行的 Pod: {pod}")
            score += float(affinity[source_index, pod_index[pod]])
        scores.append(score)
    return scores


def get_schedule_node(
    action: int,
    node_names: list[str],
    affinity_score: list[float],
    resource_usage: list[float],
    feasible: list[bool],
) -> str:
    if not any(feasible):
        raise ValueError("没有资源充足的节点")
    selected_scores = resource_usage if action == 0 else affinity_score
    masked_scores = np.asarray(
        [score if can_run else -np.inf for score, can_run in zip(selected_scores, feasible)],
        dtype=float,
    )
    return node_names[int(np.argmax(masked_scores))]


def get_model(model_path: str | Path | None = None) -> model.DQN:
    path = Path(model_path) if model_path else Path(__file__).with_name("model.pth")
    try:
        checkpoint = torch.load(path, map_location="cpu")
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"无法加载动态调度模型 {path}: {exc}") from exc
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    try:
        input_width = int(state_dict["fc1.weight"].shape[1])
    except (KeyError, AttributeError, TypeError) as exc:
        raise ValueError(f"模型文件 {path} 缺少 fc1.weight") from exc
    if input_width % 3 != 0:
        raise ValueError(f"模型输入宽度 {input_width} 不是 3 的倍数")

    dqn = model.DQN(input_width // 3)
    try:
        dqn.load_state_dict(state_dict)
    except RuntimeError as exc:
        raise ValueError(f"模型文件 {path} 与 DQN 结构不兼容: {exc}") from exc
    dqn.eval()
    return dqn


def load_pod_index(path: str | Path) -> dict[str, int]:
    with open(path, "r", newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        if not reader.fieldnames or "name" not in reader.fieldnames:
            raise ValueError(f"Pod 索引文件 {path} 缺少 name 列")
        result = {}
        for line_number, row in enumerate(reader, start=2):
            name = row["name"]
            if not name:
                raise ValueError(f"Pod 索引文件第 {line_number} 行名称为空")
            if name in result:
                raise ValueError(f"Pod 索引文件包含重复名称: {name}")
            result[name] = len(result)
    return result


def schedule_from_files(
    files: ScheduleFiles,
    affinity: np.ndarray,
    model_path: str | Path | None = None,
) -> list[SingleSchedulerPlan]:
    nodes = new_nodes(
        files.node_resources,
        files.running_pods,
        files.node_capacities,
    )
    node_names = list(nodes)
    pod_index = load_pod_index(files.pod_index)
    if affinity.shape != (len(pod_index), len(pod_index)):
        raise ValueError(
            f"亲和性矩阵尺寸 {affinity.shape} 与 Pod 数量 {len(pod_index)} 不一致"
        )
    dqn = get_model(model_path)

    plan = []
    output_rows = [["name", "node"]]
    warning_count = 0
    started_at = time.perf_counter()

    with open(files.tasks, "r", newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        required_columns = {"name", "cpu", "memory", "gpu", "disk"}
        missing_columns = required_columns.difference(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(f"待调度任务文件缺少列: {', '.join(sorted(missing_columns))}")

        for line_number, row in enumerate(reader, start=2):
            name = row["name"]
            try:
                cpu = float(row["cpu"])
                memory = float(row["memory"])
                gpu = float(row["gpu"])
                disk = float(row["disk"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"待调度任务文件第 {line_number} 行包含非法资源数值") from exc
            if min(cpu, memory, gpu, disk) < 0:
                raise ValueError(f"待调度任务文件第 {line_number} 行包含负资源需求")

            feasible = [
                nodes[node_name].can_run(cpu, memory, gpu, disk)
                for node_name in node_names
            ]
            if not any(feasible):
                warning_count += 1
                output_rows.append([name, ""])
                continue

            affinity_score = get_affinity_score(
                nodes,
                node_names,
                name,
                affinity,
                pod_index,
            )
            resource_usage = get_resource_usage(
                nodes,
                node_names,
                cpu,
                memory,
                gpu,
                disk,
            )
            input_tensor = get_model_input(
                feasible,
                affinity_score,
                resource_usage,
                dqn.input_nodes,
            )
            with torch.no_grad():
                selected_action = dqn.get_action(input_tensor)
            scheduled_node = get_schedule_node(
                selected_action,
                node_names,
                affinity_score,
                resource_usage,
                feasible,
            )
            nodes[scheduled_node].place(cpu, memory, gpu, disk, name)
            plan.append(SingleSchedulerPlan(name, scheduled_node))
            output_rows.append([name, scheduled_node])

    files.output.parent.mkdir(parents=True, exist_ok=True)
    with open(files.output, "w", newline="", encoding="utf-8") as output_file:
        csv.writer(output_file).writerows(output_rows)

    elapsed = time.perf_counter() - started_at
    logger.info(f"Processed {len(plan)} agents in {elapsed:.3f}s")
    if warning_count:
        logger.warning(f"{warning_count} agents failed scheduling due to resource constraints")
    return plan


def dynamic_schedule(
    input_dir: str,
    affinity,
    output_dir: str,
    model_path: str | Path | None = None,
) -> list[SingleSchedulerPlan]:
    input_path = Path(input_dir)
    return schedule_from_files(
        ScheduleFiles(
            node_resources=input_path / "node_resource.csv",
            node_capacities=input_path / "nodes.csv",
            running_pods=input_path / "pod_node.csv",
            tasks=input_path / "agents.csv",
            pod_index=input_path / "pods.csv",
            output=Path(output_dir),
        ),
        np.asarray(affinity),
        model_path,
    )


def main():
    project_dir = Path(__file__).resolve().parents[1]
    input_dir = project_dir / "data" / "input"
    parser = argparse.ArgumentParser(description="动态调度智能体")
    parser.add_argument("-n", "--nodes", default=input_dir / "node_resource.csv", type=Path)
    parser.add_argument("--node-capacities", default=input_dir / "nodes.csv", type=Path)
    parser.add_argument("-p", "--pods", default=input_dir / "pod_node.csv", type=Path)
    parser.add_argument("-a", "--affinity", default=project_dir / "data/output/pod_affinity.npy", type=Path)
    parser.add_argument("-t", "--tasks", default=input_dir / "agents.csv", type=Path)
    parser.add_argument("--pod-index", default=input_dir / "pods.csv", type=Path)
    parser.add_argument("-m", "--model", default=Path(__file__).with_name("model.pth"), type=Path)
    parser.add_argument("-o", "--output", default=project_dir / "data/plan.csv", type=Path)
    args = parser.parse_args()

    init_logger()
    logger.info(f"init args: {args}")
    affinity = np.load(args.affinity)
    schedule_from_files(
        ScheduleFiles(
            node_resources=args.nodes,
            node_capacities=args.node_capacities,
            running_pods=args.pods,
            tasks=args.tasks,
            pod_index=args.pod_index,
            output=args.output,
        ),
        affinity,
        args.model,
    )
    logger.info("finished schedule the agents")


if __name__ == "__main__":
    main()
