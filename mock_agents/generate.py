import argparse
from contextlib import contextmanager
import csv
from dataclasses import dataclass
import logging
from pathlib import Path

import yaml


AGENT_IMAGE = "registry.cn-hangzhou.aliyuncs.com/lexmargin/agent:v0.5"


@dataclass
class Agent:
    name: str
    cpus: int
    memory: int
    gpus: int
    disk: int
    target: str = ""
    frequency: float = 1.0
    package: int = 1
    amount: int = 1
    node: str = ""


def init_logger() -> logging.Logger:
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(console_handler)
    return logger


@contextmanager
def _dict_reader(path: str | Path, required_columns: set[str]):
    with open(path, "r", newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        missing = required_columns.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV 文件 {path} 缺少列: {', '.join(sorted(missing))}")
        yield reader


def read_csv_and_construct_agents(
    resource_file: str | Path,
    node_file: str | Path,
) -> dict[str, Agent]:
    agents: dict[str, Agent] = {}
    with _dict_reader(
        resource_file,
        {"name", "cpu", "memory", "gpu", "disk"},
    ) as reader:
        for line_number, row in enumerate(reader, start=2):
            name = row["name"]
            if not name:
                raise ValueError(f"资源文件第 {line_number} 行名称为空")
            if name in agents:
                raise ValueError(f"资源文件包含重复智能体: {name}")
            try:
                values = [
                    int(row[column] or 0)
                    for column in ("cpu", "memory", "gpu", "disk")
                ]
            except ValueError as exc:
                raise ValueError(f"资源文件第 {line_number} 行包含非法数值") from exc
            if min(values) < 0:
                raise ValueError(f"资源文件第 {line_number} 行包含负资源需求")
            agents[name] = Agent(name, *values)

    assigned = set()
    with _dict_reader(node_file, {"name", "node"}) as reader:
        for line_number, row in enumerate(reader, start=2):
            name = row["name"]
            if name not in agents:
                raise ValueError(
                    f"部署文件第 {line_number} 行引用了不存在的智能体: {name}"
                )
            if name in assigned:
                raise ValueError(f"部署文件包含重复智能体: {name}")
            if not row["node"]:
                raise ValueError(f"部署文件第 {line_number} 行节点名称为空")
            agents[name].node = row["node"]
            assigned.add(name)
    missing_assignments = sorted(set(agents).difference(assigned))
    if missing_assignments:
        raise ValueError(f"以下智能体没有节点分配: {missing_assignments}")
    return agents


def read_csv_and_generate_yamls(
    logger: logging.Logger,
    agents: dict[str, Agent],
    comm_file: str | Path,
    out_file: str | Path,
) -> None:
    generated_sources = set()
    with _dict_reader(
        comm_file,
        {"source", "target", "frequency", "package", "count"},
    ) as reader:
        for line_number, row in enumerate(reader, start=2):
            source = row["source"]
            target = row["target"]
            if source not in agents or target not in agents:
                missing = [name for name in (source, target) if name not in agents]
                raise ValueError(
                    f"通信文件第 {line_number} 行引用了不存在的智能体: {missing}"
                )
            try:
                frequency = float(row["frequency"])
                package = int(row["package"])
                amount = int(row["count"])
            except ValueError as exc:
                raise ValueError(f"通信文件第 {line_number} 行包含非法数值") from exc
            if frequency <= 0 or package <= 0 or amount <= 0:
                raise ValueError(f"通信文件第 {line_number} 行的通信参数必须大于 0")
            if source in generated_sources:
                logger.warning(f"智能体 {source} 存在多个通信目标，仅保留第一个")
                continue

            agents[source].target = target
            agents[source].frequency = frequency
            agents[source].package = package
            agents[source].amount = amount
            generated_sources.add(source)
    generate(agents, out_file)


def generate(agents: dict[str, Agent], out_file: str | Path) -> None:
    documents = []
    for agent in agents.values():
        documents.extend(generate_resources(agent))
    output = Path(out_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as outfile:
        yaml.safe_dump_all(
            documents,
            outfile,
            explicit_start=True,
            sort_keys=False,
            allow_unicode=True,
        )


def generate_resources(agent: Agent) -> list[dict]:
    name = agent.name.replace("_", "-")
    target = agent.target.replace("_", "-")
    requests = {
        "cpu": str(agent.cpus),
        "memory": f"{agent.memory}Gi",
        "ephemeral-storage": f"{agent.disk}Mi",
    }
    if agent.gpus:
        requests["nvidia.com/gpu"] = str(agent.gpus)

    deployment = {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {"name": name, "labels": {"app": name}},
        "spec": {
            "replicas": 1,
            "selector": {"matchLabels": {"app": name}},
            "template": {
                "metadata": {"name": name, "labels": {"app": name}},
                "spec": {
                    "nodeSelector": {"agent": agent.node},
                    "containers": [
                        {
                            "name": name,
                            "image": AGENT_IMAGE,
                            "command": [
                                "python3",
                                "/agent/main.py",
                                "-c",
                                str(agent.cpus),
                                "-m",
                                str(agent.memory),
                                "-f",
                                str(agent.frequency),
                                "-p",
                                str(agent.package),
                                "-t",
                                target,
                                "-a",
                                str(agent.amount),
                            ],
                            "resources": {
                                "requests": requests,
                                "limits": requests.copy(),
                            },
                            "ports": [
                                {"containerPort": 11111},
                                {"containerPort": 11112},
                            ],
                        }
                    ],
                },
            },
        },
    }
    service = {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {"name": name, "labels": {"app": "agents"}},
        "spec": {
            "selector": {"app": name},
            "ports": [
                {
                    "protocol": "TCP",
                    "port": 11111,
                    "targetPort": 11111,
                    "name": "server",
                },
                {
                    "protocol": "TCP",
                    "port": 11112,
                    "targetPort": 11112,
                    "name": "metrics",
                },
            ],
            "type": "ClusterIP",
        },
    }
    return [deployment, service]


def main():
    parser = argparse.ArgumentParser(description="生成模拟智能体 Kubernetes 资源")
    parser.add_argument("-p", "--pods", required=True, type=Path)
    parser.add_argument("-c", "--communication", required=True, type=Path)
    parser.add_argument("-n", "--nodename", required=True, type=Path)
    parser.add_argument("-o", "--output", required=True, type=Path)
    args = parser.parse_args()

    logger = init_logger()
    logger.info(f"init args: {args}")
    agents = read_csv_and_construct_agents(args.pods, args.nodename)
    read_csv_and_generate_yamls(logger, agents, args.communication, args.output)
    logger.info("finish generate yamls successfully")


if __name__ == "__main__":
    main()
