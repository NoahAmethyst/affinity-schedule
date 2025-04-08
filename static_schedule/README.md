# 静态亲和性调度
## 使用说明
指定 `input_dir` 和 `output_dir`：
* input_dir：需包含下列文件：
  * command.csv：指挥关系文件
    ![](data/others/command-data.png)
  * communication.csv：通信关系文件
    ![](data/others/communication.png)
  * nodes.csv：节点属性信息
    ![](data/others/node.png)
  * pods.csv：智能体pod属性信息
    ![](data/others/pods.png)
  * pod_affinity.npy：pod间亲和性矩阵
  * node_affinity.npy：pod和node间亲和性矩阵
* output_dir：
  * output_dir/multi_stage_scheduler.csv：调度结果
    ![](data/others/offline-scheduler-output.png)
### 目录文件介绍
* data
  * input：静态亲和性调度输入数据
  * output：静态亲和性调度输出结果
  * others：图片文件
* offline_scheduler.py：基类
* multi_stage_scheduler.py：本文多阶段调度算法实现
* worst_fit_scheduler.py：K8s默认调度算法复现

