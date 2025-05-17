import gym
import yaml
from gym import spaces
import kubernetes
from kubernetes import client, config
import time
from converter import cpu_to_nano, memory_to_Mi


class K8SEnv(gym.Env):
    ACTIONS = {
        0: 'increase_cpu_request',
        1: 'decrease_cpu_request',
        2: 'increase_memory_request',
        3: 'decrease_memory_request',
        4: 'increase_cpu_limit',
        5: 'decrease_cpu_limit',
        6: 'increase_memory_limit',
        7: 'decrease_memory_limit'
    }

    def __init__(self, namespace):
        super(K8SEnv, self).__init__()

        # Load the Kubernetes configuration from the specified file path
        config_path = "/Users/vishalkhushlani/.kube/config"
        config.load_kube_config(config_file=config_path)

        self.namespace = namespace
        self.api_instance = client.CoreV1Api()
        self.apps_v1_api = client.AppsV1Api()
        self.custom_api_instance = client.CustomObjectsApi()
        self.deployment_name = ""

        # Define action and observation space
        self.action_space = spaces.Discrete(len(self.ACTIONS))
        # Let's consider 6 metrics for observation: CPU Usage, Memory Usage, CPU Request, Memory Request, CPU Limit, Memory Limit

        self.observation_space = spaces.Box(low=0, high=float('inf'), shape=(6,), dtype=float)

    def step(self, action):
        self.apply_action(action)
        next_state = self.fetch_pod_metrics()
        reward = self.calculate_reward(next_state)
        done = False  # Here, you might want to add some conditions to define when an episode is done
        return next_state, reward, done, {}

    def reset(self):
        return self.fetch_pod_metrics()

    def render(self, mode='human', close=False):
        pass

    # Updated apply_action method
    def apply_action(self, action):
        action_name = self.ACTIONS[action]

        # Get the metrics of one of the pods (since the configuration is applied deployment-wide)
        pod_data = self.fetch_pod_metrics()[0]  # Assuming this returns a list of pod data

        cpu_usage, mem_usage, cpu_request, mem_request, cpu_limit, mem_limit = pod_data

        # Apply the action logic here.
        if action_name == "increase_cpu_request":
            cpu_request += 10
        elif action_name == "decrease_cpu_request":
            cpu_request -= 10
        elif action_name == "increase_memory_request":
            mem_request += 100
        elif action_name == "decrease_memory_request":
            mem_request -= 100
        elif action_name == "increase_cpu_limit":
            cpu_limit += 10
        elif action_name == "decrease_cpu_limit":
            cpu_limit -= 10
        elif action_name == "increase_memory_limit":
            mem_limit += 100
        elif action_name == "decrease_memory_limit":
            mem_limit -= 100

        # Ensure requests are never zero or negative
        cpu_request = max(cpu_request, 50)  # Set a minimum threshold for CPU request
        mem_request = max(mem_request, 100)  # Set a minimum threshold for memory request

        cpu_limit = max(cpu_limit, 100)  # Set a minimum threshold for CPU limit
        mem_limit = max(mem_limit, 200)  # Set a minimum threshold for memory limit

        # Apply updated resources to the deployment
        self.update_deployment_resources(self.deployment_name, cpu_limit, mem_limit, cpu_request, mem_request)

    # The calculate_reward function remains unchanged as per the provided content
    def calculate_reward(self, state):
        cpu_usage, mem_usage, cpu_request, mem_request, cpu_limit, mem_limit = state[0]

        reward = 0

        # Reward when usage is near the request
        reward -= abs(cpu_usage - cpu_request) * 0.5  # Adjust this weight as needed
        reward -= abs(mem_usage - mem_request) * 0.5  # Adjust this weight as needed

        # Penalize when CPU usage is close to the limit
        cpu_headroom = (cpu_limit - cpu_usage) / cpu_limit
        if cpu_headroom < 0.1:  # If usage is within 10% of the limit, penalize
            reward -= (0.1 - cpu_headroom) * 5  # Adjust this penalty weight as needed

        # Heavily penalize when memory usage is close to the limit
        mem_headroom = (mem_limit - mem_usage) / mem_limit
        if mem_headroom < 0.1:  # If usage is within 10% of the limit, heavily penalize
            reward -= (0.1 - mem_headroom) * 10  # Adjust this penalty weight as needed

        return reward

    # Since the environment is not instantiated here, these are just the function definitions.

    def fetch_pod_metrics(self):
        # Fetch pod metrics for the specified namespace
        yaml_file_path = "/Users/vishalkhushlani/Work/QMUL/MscProject/KubSmart/kubernetes-config.yml"
        group = "metrics.k8s.io"
        version = "v1beta1"
        plural = "pods"
        pod_metrics = self.custom_api_instance.list_namespaced_custom_object(group, version, self.namespace, plural)

        print(pod_metrics)

        # Load the YAML data
        with open(yaml_file_path, "r") as file:
            yaml_data = list(yaml.safe_load_all(file))

        deployment_data = {}
        for doc in yaml_data:
            if doc["kind"] == "Deployment":
                deployment_name = doc["metadata"]["name"]
                self.deployment_name = deployment_name
                containers = doc["spec"]["template"]["spec"]["containers"]
                for container in containers:
                    if "resources" in container:
                        deployment_data[deployment_name] = container["resources"]

        metrics_data = []
        for item in pod_metrics["items"]:
            pod_name = item["metadata"]["name"]
            deployment_name = pod_name.rsplit('-', 2)[0]

            if "containers" in item and len(item["containers"]) > 0:
                cpu_usage = cpu_to_nano(item["containers"][0]["usage"].get("cpu", "N/A"))
                memory_usage = memory_to_Mi(item["containers"][0]["usage"].get("memory", "N/A"))

            if deployment_name in deployment_data:
                resources = deployment_data[deployment_name]
                cpu_request = cpu_to_nano(resources["requests"].get("cpu", "N/A"))
                memory_request = memory_to_Mi(resources["requests"].get("memory", "N/A"))
                cpu_limit = cpu_to_nano(resources["limits"].get("cpu", "N/A"))
                memory_limit = memory_to_Mi(resources["limits"].get("memory", "N/A"))

            metrics_data.append(
                (cpu_usage, memory_usage, cpu_request, memory_request, cpu_limit, memory_limit))

        return metrics_data

    def update_deployment_resources(self, deployment_name, cpu_limit, memory_limit, cpu_request, memory_request):
        # Convert and format the values
        cpu_limit = f"{int(cpu_limit / 10 ** 6)}m"
        memory_limit = f"{int(memory_limit / 1024)}Mi"
        cpu_request = f"{int(cpu_request / 10 ** 6)}m"
        memory_request = f"{int(memory_request / 1024)}Mi"

        print(cpu_limit, memory_limit, cpu_request, memory_request)
        self.update_yaml_resources(cpu_limit, memory_limit, cpu_request, memory_request)

        MAX_RETRIES = 3
        for attempt in range(MAX_RETRIES):
            # Retrieve the current deployment object using AppsV1Api
            deployment = self.apps_v1_api.read_namespaced_deployment(deployment_name, self.namespace)

            # Update the deployment's pod resource specifications
            deployment.spec.template.spec.containers[0].resources = client.V1ResourceRequirements(
                limits={"cpu": cpu_limit, "memory": memory_limit},
                requests={"cpu": cpu_request, "memory": memory_request}
            )

            try:
                # Apply the updated configuration to the deployment using AppsV1Api
                self.apps_v1_api.replace_namespaced_deployment(deployment_name, self.namespace, deployment)
                break  # if successful, break out of the retry loop
            except kubernetes.client.exceptions.ApiException as e:
                if e.status == 409:  # Conflict
                    print(f"Conflict detected on attempt {attempt + 1}. Retrying...")
                    continue  # Retry the loop
                else:
                    raise  # Re-raise the exception if it's not a conflict

        # Wait for the deployment to be updated
        timeout = 300  # maximum time to wait in seconds
        poll_interval = 10  # time between checks in seconds
        start_time = time.time()

        while time.time() - start_time < timeout:
            # Retrieve the updated deployment status
            updated_deployment = self.apps_v1_api.read_namespaced_deployment(deployment_name, self.namespace)

            # Check if the update is finished
            if updated_deployment.status.updated_replicas == updated_deployment.status.replicas:
                print(f"Deployment {deployment_name} updated successfully.")
                self.wait_for_pod_metrics_length(4)
                return

            time.sleep(poll_interval)

        # If the loop completes without returning, it means the update took too long
        raise RuntimeError(f"Timeout while waiting for deployment {deployment_name} to update.")


    def update_yaml_resources(self, cpu_limit, memory_limit, cpu_request, memory_request):
        with open("/Users/vishalkhushlani/Work/QMUL/MscProject/KubSmart/kubernetes-config.yml", 'r') as stream:
            try:
                config_data = list(yaml.safe_load_all(stream))

                # Locate the deployment and the container
                for item in config_data:
                    if item and item['kind'] == 'Deployment':
                        containers = item['spec']['template']['spec']['containers']
                        for container in containers:
                            if container['name'] == 'intensive-task-app':
                                # Update the resources
                                container['resources']['limits']['cpu'] = cpu_limit
                                container['resources']['limits']['memory'] = memory_limit
                                container['resources']['requests']['cpu'] = cpu_request
                                container['resources']['requests']['memory'] = memory_request
                                break

                # Save the changes back to the YAML file
                with open("/Users/vishalkhushlani/Work/QMUL/MscProject/KubSmart/kubernetes-config.yml", 'w') as outfile:
                    for doc in config_data:
                        yaml.dump(doc, outfile, default_flow_style=False)


            except yaml.YAMLError as exc:
                print(exc)

    def wait_for_pod_metrics_length(self, expected_length=4, poll_interval=3):
        group = "metrics.k8s.io"
        version = "v1beta1"
        plural = "pods"

        while True:
            # Retrieve the custom object metrics
            pod_metrics = self.custom_api_instance.list_namespaced_custom_object(group, version, self.namespace, plural)

            current_length = len(pod_metrics.get('items', []))
            print(f"Current pod metrics length: {current_length}. Expected: {expected_length}.")

            # Check the length of the 'items' list in the pod metrics
            if len(pod_metrics.get('items', [])) == expected_length:
                print(f"Found {expected_length} items in pod metrics.")
                return

            # If not found, wait for a short interval before trying again
            time.sleep(poll_interval)



