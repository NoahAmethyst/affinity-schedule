import argparse
import logging
import csv
import numpy as np

import torch
import time

from affinity.resource import SingleSchedulerPlan
from dynamic_schedule import model
from util import logger
from util.logger import init_logger, logger

NODE_COUNT = 8
NODE_NAME = ['node-1', 'node-2', 'node-3', 'node-4', 'node-5', 'node-6', 'node-7', 'node-8']


class Node:
    def __init__(self, name: str, cpu_used: float, cpu_free: float, memory_used: int, memory_free: int, net_used: int,
                 net_free: int):
        self.name = name
        self.cpu_used = float(cpu_used)
        self.cpu_free = float(cpu_free)
        self.memory_used = int(memory_used)
        self.memory_free = int(memory_free)
        self.net_used = int(net_used)
        self.net_free = int(net_free)

        self.agents = []

    def set_running_agents(self, agents: list[str]):
        self.agents = agents

    def __str__(self):
        cpu_total = self.cpu_used + self.cpu_free
        cpu_used_percent = (self.cpu_used / cpu_total) * 100 if cpu_total > 0 else 0
        memory_total = self.memory_used + self.memory_free
        memory_used_percent = (self.memory_used / memory_total) * 100 if memory_total > 0 else 0
        net_total = self.net_used + self.net_free
        net_used_percent = (self.net_used / net_total) * 100 if net_total > 0 else 0

        agent_str = ', '.join(self.agents) if self.agents else 'None'

        return f"Node Name: {self.name}\n" \
               f"CPU Used: {self.cpu_used} ({cpu_used_percent:.2f}%)\n" \
               f"CPU Free: {self.cpu_free}\n" \
               f"Memory Used: {self.memory_used} ({memory_used_percent:.2f}%)\n" \
               f"Memory Free: {self.memory_free}\n" \
               f"Net Used: {self.net_used} ({net_used_percent:.2f}%)\n" \
               f"Net Free: {self.net_free}\n" \
               f"Running Agents: {agent_str}"


def new_nodes(resource_file: str, pod_file: str) -> dict[str:Node]:
    nodes = {}

    with open(resource_file, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader)  # 读取列名，这里假设第一行是列名，跳过这一行

        for row in reader:
            nodes[row[0]] = Node(row[0], row[1], row[2], row[3], row[4], row[5], row[6])

    with open(pod_file, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader)  # 读取列名，这里假设第一行是列名，跳过这一行

        for row in reader:
            agents = row[1].split(',')
            if nodes.get(row[0]) is None:
                continue
            nodes[row[0]].set_running_agents(agents)

    return nodes


def get_model_input(resource_usage: list, affinity: list) -> torch.FloatTensor:
    can_run = list(map(lambda x: 0.0 if x == 0 else 1.0, resource_usage))
    res = [can_run, affinity, resource_usage]
    res = torch.FloatTensor(res)

    return res.transpose(0, 1).flatten()


def get_resource_usage(nodes: dict[str:Node], cpu, memory, gpu, disk) -> list:
    res = []

    for node_name in NODE_NAME:
        if nodes[node_name].cpu_free < cpu or nodes[node_name].memory_free < memory:
            res.append(0)
            continue

        res.append((nodes[node_name].cpu_used + cpu) / (nodes[node_name].cpu_used + nodes[node_name].cpu_free))

    return res


def get_balance_score(nodes: dict[str:Node], cpu, memory, gpu, disk) -> list:
    scores = []

    for node_name in NODE_NAME:
        if nodes[node_name].cpu_free < cpu or nodes[node_name].memory_free < memory:
            scores.append(-1)
            continue

        scores.append(nodes[node_name].cpu_free - cpu)

    return scores


def get_affinity_score(nodes: dict[str:Node], agent: str, affinity) -> list:
    scores = []
    sk = get_affinity_key(agent)

    for node_name in NODE_NAME:
        score = 0
        for pod in nodes[node_name].agents:
            tk = get_affinity_key(pod)
            score += affinity[sk][tk]
        scores.append(score)

    return scores


def get_affinity_key(name: str) -> int:
    return int(name.split("-")[1]) - 1


def get_schedule_node(action: int, affinity_score: list, resource_usage: list) -> str:
    score = []

    if action == 0:
        score = np.array(resource_usage)
    else:
        score = np.array(affinity_score)

    return NODE_NAME[np.argmax(score)]


def update_nodes(nodes: dict[str:Node], scheduled_node: str, cpu, memory, gpu, disk, name):
    nodes[scheduled_node].cpu_free -= cpu
    nodes[scheduled_node].memory_free -= memory
    nodes[scheduled_node].agents.append(name)


def get_model() -> model.DQN:
    dqn = model.DQN(NODE_COUNT)
    dqn.load_state_dict(torch.load('model.pth'))
    return dqn


def dynamic_schedule(input_dir: str, affinity) -> list[SingleSchedulerPlan]:
    """Optimized dynamic scheduling function with improved efficiency and readability."""
    # Constants for file names
    NODES_RESOURCE_FILE = 'node_resource.csv'
    PODS_FILE = 'pod_node.csv'
    TASKS_FILE = 'agents.csv'

    # Load data once
    nodes = new_nodes(
        f'{input_dir}/{NODES_RESOURCE_FILE}',
        f'{input_dir}/{PODS_FILE}'
    )
    dqn = get_model()

    # Use list for faster string building
    plan = []
    warning_count = 0

    with open(f'{input_dir}/{TASKS_FILE}', 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header

        start_time = time.perf_counter()  # More precise timing

        for row in reader:
            name, cpu, memory, gpu, disk, platform = row
            cpu, memory = float(cpu), int(memory)

            # Pre-check resource availability
            if not _has_sufficient_resources(nodes, cpu, memory):
                warning_count += 1
                continue

            affinity_score = get_affinity_score(nodes, name, affinity)
            resource_usage = get_resource_usage(nodes, cpu, memory, gpu, disk)

            input_tensor = get_model_input(resource_usage, affinity_score)
            selected_action = dqn.get_action(input_tensor)
            scheduled_node = get_schedule_node(selected_action, affinity_score, resource_usage)

            update_nodes(nodes, scheduled_node, cpu, memory, gpu, disk, name)

            plan.append(SingleSchedulerPlan(name, scheduled_node))

        elapsed = time.perf_counter() - start_time
        print(f'Processed {len(plan)} agents in {elapsed:.3f}s')

    if warning_count:
        logger.warning(f'{warning_count} agents failed scheduling due to resource constraints')

    return plan  # Efficient string concatenation


def _has_sufficient_resources(nodes: dict[str:Node], cpu: float, memory: int) -> bool:
    """Helper function to check resource availability."""
    return any(
        node.cpu_free >= cpu and node.memory_free >= memory
        for node in nodes.values()
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='please enter the status of running pods and nodes')

    parser.add_argument('-n', '--nodes', type=str, help='please enter the file path of node resource usage')
    parser.add_argument('-p', '--pods', type=str, help='please enter the file path of running pod deployment')
    parser.add_argument('-a', '--affinity', type=str, help='please enter the file path of affinity score')
    parser.add_argument('-t', '--tasks', type=str, help='please enter the file path of agents need to schedule')
    parser.add_argument('-o', '--output', type=str, help='please enter the file path of schedule result')

    args = parser.parse_args()

    init_logger()

    if args.nodes is None:
        args.nodes = '../data/input/node_resource.csv'
    if args.pods is None:
        args.pods = '../data/input/pod_node.csv'
    if args.affinity is None:
        args.affinity = '../data/output/pod_affinity.npy'
    if args.tasks is None:
        args.tasks = '../data/input/agents.csv'
    if args.output is None:
        args.output = '/Users/amethyst/PycharmProjects/affinity-schedule/data/plan.csv'

    logger.info(f'init args: {args}')
    logger.info("start to schedule the agents")

    nodes = new_nodes(args.nodes, args.pods)

    affinity = np.load(args.affinity)

    dqn = get_model()

    res = "name,node\n"

    with open(args.tasks, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader)  # 读取列名，这里假设第一行是列名，跳过这一行

        start_time = time.time()

        for row in reader:
            name, cpu, memory, gpu, disk, platform = row
            cpu, memory, gpu, disk = float(cpu), int(memory), int(gpu), int(disk)
            affinity_score = get_affinity_score(nodes, name, affinity)
            resource_usage = get_resource_usage(nodes, cpu, memory, gpu, disk)

            if all(usage == 0 for usage in resource_usage):
                res += f'{name},\n'
                logger.warning(f'agent: {name} schedule failed, because of shortage of resources.')
                continue

            input = get_model_input(resource_usage, affinity_score)
            selected_action = dqn.get_action(input)

            scheduled_node = get_schedule_node(selected_action, affinity_score, resource_usage)

            update_nodes(nodes, scheduled_node, cpu, memory, gpu, disk, name)

            res += f'{name},{scheduled_node}\n'

        end_time = time.time()
        print(f'process time: {end_time - start_time}')

    with open(args.output, 'w') as file:
        file.write(res)

    logger.info("finished schedule the agents")
