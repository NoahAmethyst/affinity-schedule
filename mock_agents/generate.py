import csv
import argparse
import logging


class Agent:
    def __init__(self, name:str, cpus:int, memory:int,gpus:int,disk:int)->None:
        self.name = name
        self.cpus = cpus
        self.memory = memory
        self.gpus = gpus
        self.disk = disk
        self.target = ""
        self.frequency = 1.0
        self.package = 1
        self.amount = 1
        self.node = ""

def init_logger() -> logging.Logger:
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # log to console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # log to file
    # file_handler = logging.FileHandler('app.log')
    # file_handler.setLevel(logging.DEBUG)
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # file_handler.setFormatter(formatter)
    # logger.addHandler(file_handler)

    return logger


def read_csv_and_construct_agents(resource_file:str, node_file:str)->dict[str, Agent]:
    agents_dict = {}
    with open(resource_file, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader)  # 读取列名，这里假设第一行是列名，跳过这一行

        for row in reader:
            name = row[0]
            cpus = int(row[1]) if row[1] else 0  # 转换为整数，若为空则设为0
            memory = int(row[2]) if row[2] else 0
            gpus = int(row[3]) if row[3] else 0
            disk = int(row[4]) if row[4] else 0

            agent_obj = Agent(name, cpus, memory, gpus, disk)
            agents_dict[name] = agent_obj
    
    with open(node_file, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader)  # 读取列名，这里假设第一行是列名，跳过这一行

        for row in reader:
            name = row[0]
            node = row[1]

            if agents_dict[name] == None:
                logger.error(f'pod {name} dose not exist!')

            agents_dict[name].node = node
        
    return agents_dict


# def read_csv_and_generate_yamls(logger:logging.Logger, agents:dict[str, Agent], comm_file:str, out_file:str)->None:
#     generated = set()
#     with open(out_file, 'w') as outfile:
#         with open(comm_file, 'r', newline='') as csvfile:
#             reader = csv.reader(csvfile)
#             headers = next(reader)  # 读取列名，这里假设第一行是列名，跳过这一行

#             for row in reader:
#                 source, target, frequency, package, amount = row[0], row[1], float(row[2]) if row[2] else 0.0, int(row[3]) if row[3] else 0, int(row[4]) if row[4] else 0
#                 if frequency == 0 or package == 0 or amount == 0:
#                     logger.warning(f'some parameters are illegal: frequency: {frequency}, package: {package}, amount: {amount}!')
#                     continue

#                 if source in generated:
#                     logger.warning(f'{source} has been used before!')
#                     continue

#                 if agents[source] == None or agents[target] == None:
#                     logger.warning(f'{source} or {target} is not defined in pods resource configuration file before!')
#                     continue
                
#                 agents[source].target = target
#                 agents[source].frequency = frequency
#                 agents[source].package = package
#                 agents[source].amount = amount
#                 outfile.write(generate_yamls(source, agents[source].cpus, agents[source].memory, frequency, package, target, amount))
#                 generated.add(source)

def read_csv_and_generate_yamls(logger:logging.Logger, agents:dict[str, Agent], comm_file:str, out_file:str)->None:
    generated = set()
    with open(comm_file, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader)  # 读取列名，这里假设第一行是列名，跳过这一行

        for row in reader:
            source, target, frequency, package, amount = row[0], row[1], float(row[2]) if row[2] else 0.0, int(row[3]) if row[3] else 0, int(row[4]) if row[4] else 0
            if frequency == 0 or package == 0 or amount == 0:
                logger.warning(f'some parameters are illegal: frequency: {frequency}, package: {package}, amount: {amount}!')
                continue

            if source in generated:
                logger.warning(f'{source} has been used before!')
                continue

            if agents[source] == None or agents[target] is None:
                logger.warning(f'{source} or {target} is not defined in pods resource configuration file before!')
                continue
            
            agents[source].target = target
            agents[source].frequency = frequency
            agents[source].package = package
            agents[source].amount = amount
            generated.add(source)
    
    generate(logger, agents, out_file)

def generate(logger:logging.Logger, agents:dict[str, Agent], out_file:str)->None:
    with open(out_file, 'w') as outfile:
        for agent in agents.values():
            outfile.write(generate_yamls(agent.name, agent.cpus, agent.memory, agent.frequency, agent.package, agent.target, agent.amount, agent.node))
    

def generate_yamls(name:str, cpu:int, memory:int, frequency:float, package:int, target:str, amount:int, node:str)->str:
    return f"""
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {name}
  labels:
    app: {name}
spec:
  replicas: 1
  selector:
    matchLabels:
      app: {name}
  template:
    metadata:
      name: {name}
      labels:
        app: {name}
    spec:
      nodeSelector:
        agent: {node}
      containers:
        - name: {name}
          image: registry.cn-hangzhou.aliyuncs.com/lexmargin/agent:v0.5
          command: ["python3", "/agent/main.py", "-c", "{cpu}", "-m", "{memory}", "-f", "{frequency}", "-p", "{package}", "-t", "{target}", "-a", "{amount}"]
          ports:
          - containerPort: 11111
          - containerPort: 11112
---
apiVersion: v1
kind: Service
metadata:
  name: {name}
  labels:
    app: agents
spec:
  selector:
    app: {name}
  ports:
    - protocol: TCP
      port: 11111  # 对外提供服务的端口，可以根据实际需求修改
      targetPort: 11111  # Pod 内实际监听的端口，要和 Pod 中应用监听的端口对应
      name: server
    - protocol: TCP
      port: 11112  # 对外提供服务的端口，可以根据实际需求修改
      targetPort: 11112  # Pod 内实际监听的端口，要和 Pod 中应用监听的端口对应
      name: metrics
  type: ClusterIP  # 服务类型，这里使用 ClusterIP，可根据需求换成其他类型（如 NodePort、LoadBalancer 等）""".format(name, cpu, memory, frequency, package, target, amount)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='please enter the configuration of the intelligent agent')

    parser.add_argument('-p', '--pods', type=str, help='please enter the file name of pods resource configuration')
    parser.add_argument('-c', '--communication', type=str, help='please enter the file name of communication configuration')
    parser.add_argument('-n', '--nodename', type=str, help='please enter the file name of deployment node configuration')
    parser.add_argument('-o', '--output', type=str, help='please enter the file name of output yamls')
    
    args = parser.parse_args()

    logger = init_logger()

    logger.info(f'init args: {args}')
    logger.info("start to generate the agents")

    agents = read_csv_and_construct_agents(args.pods, args.nodename)
    logger.info(f'init the agents source usage successfully')

    read_csv_and_generate_yamls(logger, agents, args.communication, args.output)

    logger.info("finish generate yamls successfully")





