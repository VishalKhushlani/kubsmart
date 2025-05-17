"""
Script to continuously fetch and display Kubernetes pod metrics.
"""

from time import sleep

import numpy as np

from converter import cpu_to_milli, memory_to_Mi
from kubernetes_environment_rl import K8SEnv


HEADER = (
    "Pod Name\tCPU Usage (millicores)\tMemory Usage (MiB)\t"
    "CPU Request (millicores)\tMemory Request (MiB)\t"
    "CPU Limit (millicores)\tMemory Limit (MiB)"
)


def main():
    """
    Continuously fetch and print pod metrics every 10 seconds.
    """
    env = K8SEnv(namespace="default")

    while True:
        metrics = env.fetch_pod_metrics()
        print(metrics)
        print(HEADER)

        for (
            pod_name,
            cpu_usage,
            memory_usage,
            cpu_request,
            memory_request,
            cpu_limit,
            memory_limit,
        ) in metrics:
            print(
                f"{pod_name}\t"
                f"{cpu_to_milli(cpu_usage)}m\t"
                f"{memory_to_Mi(memory_usage):.2f}Mi\t"
                f"{cpu_to_milli(cpu_request)}m\t"
                f"{memory_to_Mi(memory_request):.2f}Mi\t"
                f"{cpu_to_milli(cpu_limit)}m\t"
                f"{memory_to_Mi(memory_limit):.2f}Mi"
            )

        sleep(10)


if __name__ == "__main__":
    main()
