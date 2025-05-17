from time import sleep
from converter import cpu_to_milli, memory_to_Mi
import numpy as np

# Import the K8SEnv class defined earlier
# If K8SEnv is defined in another file, you'll need to import it appropriately

# Replace 'your-namespace' with the appropriate namespace for your cluster
from kubernetes_environment_rl import K8SEnv

env = K8SEnv(namespace='default')

while True:
    # Fetch pod metrics
    # Fetch pod metrics for the specified namespace
    state = env.fetch_pod_metrics()

    # Print the metrics
    # Print the metrics
    print(state)
    print(
        "Pod Name\tCPU Usage (millicores)\tMemory Usage (MiB)\tCPU Request (millicores)\tMemory Request (MiB)\tCPU Limit (millicores)\tMemory Limit (MiB)")

    for pod_name, cpu_usage, memory_usage, cpu_request, memory_request, cpu_limit, memory_limit in state:
        print(
            f"{pod_name}\t"
            f"{cpu_to_milli(cpu_usage)}m\t\t\t"
            f"{memory_to_Mi(memory_usage):.2f}Mi\t\t\t"
            f"{cpu_to_milli(cpu_request)}m\t\t\t"
            f"{memory_to_Mi(memory_request):.2f}Mi\t\t\t"
            f"{cpu_to_milli(cpu_limit)}m\t\t\t"
            f"{memory_to_Mi(memory_limit):.2f}Mi"
        )

    sleep(10)
