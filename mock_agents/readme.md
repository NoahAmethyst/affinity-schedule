# 使用说明

## 准备
请提前按需要准备以下环境：
- python3环境
- docker环境（用于本地生成镜像）
- K8s集群（用于部署生成的智能体）
- Promethues服务（用于监控智能体的处理延迟）

## 制作镜像
运行以下命令：
```
docker build -t <image_name>:<version> .
```

## 生成智能体
### 准备智能体资源需求文件
- name: 智能体的名称
- cpu: 智能体使用的CPU核数（int）
- memory: 智能体使用的内存（GB-int）
- gpu: 智能体使用的GPU数量（没有模拟）
- disk: 智能体使用的硬盘资源数量（没有模拟）
```
name,cpu,memory,gpu,disk
pod-1,4,4,1,2
pod-2,4,5,2,3
pod-3,5,3,1,2
pod-4,4,3,1,2
pod-5,5,3,1,2
pod-6,4,5,2,2
pod-7,4,5,2,3
pod-8,1,1,1,1
```
### 准备智能体通信配置文件
- source: 进行通信的源智能体名称
- target: 进行通信的目的智能体名称
- frequency: 通信频率（次(float)/s）
- package: 单词通信的包数据大小（MB(int)/s）
- amount: 总通信次数（次(int)/s）
```
source,target,frequency,package,amount
pod-1,pod-2,3,100,10000
pod-2,pod-3,4,100,10000
pod-3,pod-1,5,100,10000
pod-4,pod-5,3,200,10000
pod-5,pod-6,4,200,10000
pod-6,pod-7,5,200,10000
```
参考：

|网络带宽|频率|包大小|
| :----: | :----:| :----:|
|380Mb/s|0.5|100|
|700Mb/s|1|100|
|2.1Gb/s|5|100|
|2.8Gb/s|10|100|
|3.1Gb/s|15|100|
|3.2Gb/s|10|200|
|3.8Gb/s|10|400|
|3.7Gb/s|10|600|

### 准备智能体部署需求文件
- name: 智能体的名称
- node: 智能体运行的节点的label(请提前在对应的节点上打label:<agent:name>)
```
name,node
name,node
pod-1,node-1
pod-2,node-2
pod-3,node-3
pod-4,node-4
pod-5,node-5
pod-6,node-1
pod-7,node-2
pod-8,node-3
```


### 生成智能体相关的YAML部署文件
运行以下命令：
```
python3 generate.py -p <智能体资源需求.csv> -c <智能体通信配置.csv> -n <智能体部署配置.csv> -o <部署文件.yaml>
``` 

### 部署处理延迟监控
运行以下命令：
```
kubectl apply -f yamls/monitor.yaml
```
部署后需要自己在grafana中按照需要配置相应的图表

### 部署智能体
运行以下命令：
```
kubectl apply -f <部署文件.yaml>
```