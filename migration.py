# pod在线迁移测试脚本
import sys
import time
from kubernetes import client, config

namespace = "baowj"


def deletePod(pod_name, pod_namespace):
    api = client.CoreV1Api()
    api.delete_namespaced_pod(pod_name, pod_namespace)
    time.sleep(5)
    print("Delete pod: {}".format(pod_name))


def checkPodReady(pod_name, pod_namespace) -> bool:
    api = client.CoreV1Api()
    while True:
        pod = api.read_namespaced_pod(pod_name, pod_namespace).status
        if pod.phase == "Running":
            print("Pod-{} is {}.".format(pod_name, pod.phase))
            return True
        else:
            print("Pod-{} is {}.".format(pod_name, pod.phase))
            time.sleep(1)


def createNewPod(pod_name, new_pod_name, pod_namespace, node_name) -> bool:
    api = client.CoreV1Api()
    pod = api.read_namespaced_pod(pod_name, pod_namespace)
    new_pod = client.V1Pod(
        api_version="v1",
        kind="Pod",
        # metadata=pod.metadata,
        metadata=client.V1ObjectMeta(name=new_pod_name, labels=pod.metadata.labels),
        spec=pod.spec
    )
    # new_pod.metadata.labels = pod.metadata.labels
    print(pod.spec.node_name)
    new_pod.spec.node_name = node_name
    api.create_namespaced_pod(namespace=namespace, body=new_pod)
    print("Pod created.")
    time.sleep(5)


def main(args):
    pod_name = args[0]
    node_name = args[1]
    new_pod_name = pod_name + "-new"
    exit_code = 0

    ### Load config
    config.load_kube_config()

    ### Check old pod ready
    checkPodReady(pod_name, namespace)

    ### Create new pod
    createNewPod(pod_name, new_pod_name, namespace, node_name)

    ### Check pod ready
    checkPodReady(new_pod_name, namespace)

    ### Delete previous pod
    deletePod(pod_name, namespace)
    return exit_code


if __name__ == '__main__':
    args = sys.argv[1:]
    # args = ["migration-test", "pasak8s-18"]
    exit(main(args))
