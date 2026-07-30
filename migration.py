import argparse
import copy
from pathlib import Path
import time

from kubernetes import client, config
from kubernetes.client.exceptions import ApiException


DEFAULT_NAMESPACE = "baowj"
DEFAULT_TIMEOUT_SECONDS = 120


class MigrationError(RuntimeError):
    pass


def wait_for_pod_ready(
    api: client.CoreV1Api,
    pod_name: str,
    namespace: str,
    timeout_seconds: int,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            status = api.read_namespaced_pod_status(pod_name, namespace).status
        except ApiException as exc:
            if exc.status == 404:
                time.sleep(1)
                continue
            raise MigrationError(f"读取 Pod {pod_name} 状态失败: {exc}") from exc

        if status.phase == "Running":
            conditions = status.conditions or []
            if any(
                condition.type == "Ready" and condition.status == "True"
                for condition in conditions
            ):
                return
        if status.phase in {"Failed", "Succeeded"}:
            raise MigrationError(f"Pod {pod_name} 进入终止状态: {status.phase}")
        time.sleep(1)
    raise MigrationError(f"等待 Pod {pod_name} Ready 超时（{timeout_seconds} 秒）")


def create_replacement_pod(
    api: client.CoreV1Api,
    source_pod,
    new_pod_name: str,
    namespace: str,
    node_name: str,
) -> None:
    spec = copy.deepcopy(source_pod.spec)
    spec.node_name = node_name
    replacement = client.V1Pod(
        api_version="v1",
        kind="Pod",
        metadata=client.V1ObjectMeta(
            name=new_pod_name,
            labels=copy.deepcopy(source_pod.metadata.labels),
            annotations=copy.deepcopy(source_pod.metadata.annotations),
        ),
        spec=spec,
    )
    try:
        api.create_namespaced_pod(namespace=namespace, body=replacement)
    except ApiException as exc:
        raise MigrationError(f"创建替代 Pod {new_pod_name} 失败: {exc}") from exc


def delete_pod(api: client.CoreV1Api, pod_name: str, namespace: str) -> None:
    try:
        api.delete_namespaced_pod(pod_name, namespace)
    except ApiException as exc:
        if exc.status != 404:
            raise MigrationError(f"删除 Pod {pod_name} 失败: {exc}") from exc


def migrate_pod(
    pod_name: str,
    node_name: str,
    namespace: str = DEFAULT_NAMESPACE,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    allow_managed: bool = False,
) -> str:
    api = client.CoreV1Api()
    try:
        source_pod = api.read_namespaced_pod(pod_name, namespace)
    except ApiException as exc:
        raise MigrationError(f"读取源 Pod {pod_name} 失败: {exc}") from exc

    if source_pod.metadata.owner_references and not allow_managed:
        owners = [
            f"{owner.kind}/{owner.name}"
            for owner in source_pod.metadata.owner_references
        ]
        raise MigrationError(
            f"Pod {pod_name} 由 {owners} 管理；请迁移控制器或显式使用 --allow-managed"
        )

    wait_for_pod_ready(api, pod_name, namespace, timeout_seconds)
    new_pod_name = f"{pod_name}-new"
    replacement_created = False
    try:
        create_replacement_pod(
            api,
            source_pod,
            new_pod_name,
            namespace,
            node_name,
        )
        replacement_created = True
        wait_for_pod_ready(api, new_pod_name, namespace, timeout_seconds)
        delete_pod(api, pod_name, namespace)
    except Exception:
        if replacement_created:
            delete_pod(api, new_pod_name, namespace)
        raise
    return new_pod_name


def main():
    parser = argparse.ArgumentParser(description="将独立 Pod 迁移到指定节点")
    parser.add_argument("pod_name")
    parser.add_argument("node_name")
    parser.add_argument("-n", "--namespace", default=DEFAULT_NAMESPACE)
    parser.add_argument(
        "-t",
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
    )
    parser.add_argument("--allow-managed", action="store_true")
    parser.add_argument("--kubeconfig", type=Path)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("timeout 必须大于 0")

    config.load_kube_config(config_file=str(args.kubeconfig) if args.kubeconfig else None)
    new_pod_name = migrate_pod(
        args.pod_name,
        args.node_name,
        args.namespace,
        args.timeout,
        args.allow_managed,
    )
    print(f"Pod migrated: {args.pod_name} -> {new_pod_name}")


if __name__ == "__main__":
    main()
