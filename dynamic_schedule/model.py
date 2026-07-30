import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 自定义一个简单环境示例（仅用于演示，实际可替换为真实环境）
class TrainEnv:
    def __init__(self, episode_length=20):
        self.n = 8
        self.episode_length = episode_length
        self.current_step = 0
        self.state = self.generate_state()
        self.done = False
        print('加载历史数据，构建训练环境')

    def reset(self):
        self.current_step = 0
        self.state = self.generate_state()
        self.done = False
        return self.state

    def step(self, action):
        state = self.state.reshape(self.n, 3)
        feasible = state[:, 0] > 0
        if not feasible.any():
            reward = -1.0
        else:
            score_column = 2 if action == 0 else 1
            reward = float(np.max(state[feasible, score_column]))

        self.current_step += 1
        self.done = self.current_step >= self.episode_length
        self.state = self.generate_state()
        return self.state, reward, self.done, {}

    def generate_state(self):
        state = np.zeros((self.n, 3))

        for i in range(self.n):
            state[i][0] = np.random.randint(0, 2)
            state[i][1] = np.random.rand()

            if state[i][0] == 1:
                state[i][2] = np.random.rand()
    
        return state.flatten()


# 定义超参数
BATCH_SIZE = 64
LR = 0.001
GAMMA = 0.99
MEMORY_CAPACITY = 10000
EPSILON = 0.9
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
TARGET_UPDATE_FREQ = 100

# 定义经验回放缓冲区类
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        index_list = list(range(len(self.buffer)))
        sampled_index = np.random.choice(index_list, batch_size, replace=False)
        sampled_data = [self.buffer[i] for i in sampled_index]
        states, actions, rewards, next_states, dones = zip(*sampled_data)
        return np.array(states).astype(float), np.array(actions).astype(int), np.array(rewards).reshape(-1, 1), np.array(next_states).astype(float), np.array(dones).reshape(-1, 1)

    def __len__(self):
        return len(self.buffer)

# 定义DQN网络模型，适配n*3的输入状态且动作空间为2
class DQN(nn.Module):
    def __init__(self, n):
        super(DQN, self).__init__()
        self.input_nodes = n
        self.fc1 = nn.Linear(3 * n, 128)  # 根据输入特征数量调整输入层节点数
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)  # 明确输出层节点数为2，对应2个动作

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def get_action(self, x)->int:
        with torch.no_grad():
            return self(x).argmax().item()


if __name__ == "__main__":
    print('准备训练环境')
    # 创建环境实例,加载历史数据
    env = TrainEnv()
    n = env.n  # 获取环境中定义的n值

    # 创建主网络和目标网络
    policy_net = DQN(n)
    target_net = DQN(n)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    # 定义优化器和损失函数
    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    criterion = nn.MSELoss()

    # 创建经验回放缓冲区
    memory = ReplayBuffer(MEMORY_CAPACITY)

    # 训练循环
    step_counter = 0

    round=500
    print(f'设置训练轮次：{round}')

    print('开始训练')
    for episode in range(round):
        state = env.reset()
        state_tensor = torch.FloatTensor(state)
        done = False
        episode_reward = 0
        while not done:
            step_counter += 1
            # 根据epsilon贪心策略选择动作
            if np.random.random() < EPSILON:
                with torch.no_grad():
                    q_values = policy_net(state_tensor)
                    action = q_values.argmax().item()
            else:
                action = np.random.choice(2)  # 从2个动作中随机选择

            next_state, reward, done, _ = env.step(action)
            
            memory.push(state, action, reward, next_state, done)
            episode_reward += reward

            if len(memory) >= BATCH_SIZE:
                states, actions, rewards, next_states, dones = memory.sample(BATCH_SIZE)
                actions = torch.from_numpy(actions)
                states = torch.from_numpy(states).float()
                next_states = torch.from_numpy(next_states).float()
                rewards = torch.from_numpy(rewards).float()
                
                q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    next_q_values = target_net(next_states).max(1)[0]
                    target_q_values = rewards.squeeze(1) + GAMMA * next_q_values * (1 - dones.squeeze(1))
                    target_q_values = target_q_values.squeeze(0)

                
                loss = criterion(q_values.float(), target_q_values.float())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            state = next_state

            if step_counter % TARGET_UPDATE_FREQ == 0:
                target_net.load_state_dict(policy_net.state_dict())

        print(f'第: {episode}轮训练, 奖励: {episode_reward}')
        if EPSILON > EPSILON_MIN:
            EPSILON *= EPSILON_DECAY

        time.sleep(1)

    print('模型训练结束')
    # 测试环节

    test_episodes = 100

    print(f'开始进行模型测试，轮次{test_episodes}')
    for i in range(test_episodes):
        state = env.reset()
        state = torch.FloatTensor(state)
        done = False
        episode_reward = 0
        while not done:
            with torch.no_grad():
                q_values = policy_net(state)
                action = q_values.argmax().item()

            next_state, reward, done, _ = env.step(action)
            next_state = torch.FloatTensor(next_state)
            episode_reward += reward
            state = next_state
        print(f'测试，轮次{test_episodes}, 奖励: {episode_reward}')
        time.sleep(1)

    # 保存模型
    print('测试结束，保存模型')
    model_path = Path(__file__).with_name("model.pth")
    torch.save(policy_net.state_dict(), model_path)
    
    # 加载模型
    print('模型结构：')
    model = DQN(env.n)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    state = env.reset()
    state = torch.FloatTensor(state)
    print(model(state))
