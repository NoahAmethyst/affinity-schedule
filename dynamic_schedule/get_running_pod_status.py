import csv
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='please enter the deployment of running pods')

    parser.add_argument('-i', '--input', type=str, help='please enter the file path of deployment')
    parser.add_argument('-o', '--output', type=str, help='please enter the file path of output')

    node_pods = {}
    args = parser.parse_args()
    with open(args.input, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader)  # 读取列名，这里假设第一行是列名，跳过这一行

        for row in reader:
            pod, node = row[0], row[1]

            if node not in  node_pods:
                node_pods[node] = [pod]
            else:
                node_pods[node].append(pod)
    
    res = "node,agents\n"
    for k, v in node_pods.items():
        agents = ",".join(sorted(v, key=lambda x: int(x.split('-')[1])))
        res += f"{k},\"{agents}\"\n"  # 添加换行符作为每次累加的分隔符

    print(res)
    with open(args.output, 'w', encoding='utf-8') as file:
        file.write(res)

