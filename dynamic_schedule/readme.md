# 使用说明
智能体动态调度
## 准备
请提前按需要准备以下环境：
- python3环境

## 介绍
文件内容：
- input：存放动态调度的输入文件
- output：存放动态调度的结果
- main.py：动态调度代码
- model.pth：DQN强化学习权重
- model.py：DQN强化学习代码

## 使用步骤
### 模型训练
根据自身的集群状态和环境对DQN模型进行重训练
### 准备输入文件
1. agents.csv：需要动态调度的智能体
```
name,cpu,memory,gpu,disk,platform
pod-10,200,20,0,10,"aaaaa"
pod-11,3,30,0,10,"bbbbb"
```
2. node_resource.csv：目前集群的资源使用情况
```
name,cpu_used(cores),cpu_free(cores),memory_used(GiB),memory_free(GiB),network_used(Mb/s),network_free(Mb/s)
node-1,20,44,20,236,2000,8000
node-2,20,44,20,236,2000,8000
node-3,20,44,20,236,2000,8000
node-4,20,44,20,236,2000,8000
node-5,20,44,20,236,2000,8000
node-6,20,44,20,236,2000,8000
node-7,20,44,20,236,2000,8000
node-8,20,44,20,236,2000,8000
```
3. pod_node.csv：目前各个节点上运行的智能体

调用`get_running_pod_status.py`从之前静态调度结果中获取每个节点运行的pod：
```
python3 get_running_pod_status.py -i <静态调度结果中.csv> -o <智能体运行的节点统计.csv>
```

```
name,agents
node-1,"pod-1,pod-2,pod-3"
node-2,"pod-4,pod-5,pod-6"
node-3,"pod-7,pod-8"
node-4,"pod-9,pod-10"
node-5,"pod-11,pod-12"
node-6,"pod-13,pod-14"
node-7,"pod-15,pod-16"
node-8,"pod-17,pod-18"
```
4. pod_affinity.npy：更新后的亲和性矩阵

修改pods.csv对智能体的添加,删除的话不能在这里体现，不然就亲和性index就乱了，删除pod把它从通信关系csv里面删除就好

修改communication.csv对智能体的通讯关系进行修改

运行静态调度章节的代码重新生成亲和性矩阵
### 运行
运行以下命令：
```
python3 main.py -n <资源使用.csv> -p <节点上运行的智能体.csv> -a <亲和性矩阵.npy>  -t <待调度的智能体.csv> -o <调度结果.csv>
```