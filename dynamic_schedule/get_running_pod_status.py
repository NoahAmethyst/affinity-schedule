import argparse
import csv
from pathlib import Path


def _pod_sort_key(name: str):
    prefix, separator, suffix = name.rpartition("-")
    if separator and suffix.isdigit():
        return prefix, int(suffix)
    return name, 0


def group_running_pods(input_path: str | Path, output_path: str | Path):
    node_pods: dict[str, list[str]] = {}
    with open(input_path, "r", newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        required_columns = {"name", "node"}
        missing = required_columns.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"调度结果缺少列: {', '.join(sorted(missing))}")
        for line_number, row in enumerate(reader, start=2):
            pod = row["name"]
            node = row["node"]
            if not pod or not node:
                raise ValueError(f"调度结果第 {line_number} 行的 Pod 或节点名称为空")
            node_pods.setdefault(node, []).append(pod)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["node", "agents"])
        for node, pods in node_pods.items():
            writer.writerow([node, ",".join(sorted(pods, key=_pod_sort_key))])


def main():
    parser = argparse.ArgumentParser(description="汇总节点上正在运行的 Pod")
    parser.add_argument("-i", "--input", required=True, type=Path)
    parser.add_argument("-o", "--output", required=True, type=Path)
    args = parser.parse_args()
    group_running_pods(args.input, args.output)


if __name__ == "__main__":
    main()
