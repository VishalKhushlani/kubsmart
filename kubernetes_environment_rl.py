import os
import time

import gym
from gym import spaces
import yaml

from kubernetes import client, config
import kubernetes

from converter import cpu_to_nano, memory_to_Mi


# Constants for file paths
KUBE_CONFIG_PATH = os.path.expanduser("~/.kube/config")
YAML_FILE_PATH = os.path.expanduser(
    "~/Work/QMUL/MscProject/KubSmart/kubernetes-config.yml"
)


class K8SEnv(gym.Env):
    """
    Gym environment for tuning Kubernetes Deployment resources.

    Actions adjust CPU/memory requests and limits;
    observations report current usage and resource specs.
    """

    ACTIONS = {
        0: 'increase_cpu_request',
        1: 'decrease_cpu_request',
        2: 'increase_memory_request',
        3: 'decrease_memory_request',
        4: 'increase_cpu_limit',
        5: 'decrease_cpu_limit',
        6: 'increase_memory_limit',
        7: 'decrease_memory_limit',
    }

    def __init__(self, namespace):
        super().__init__()

        # Load Kubernetes configuration
        config.load_kube_config(config_file=KUBE_CONFIG_PATH)

        self.namespace = namespace
        self.api_instance = client.CoreV1Api()
        self.apps_v1_api = client.AppsV1Api()
        self.custom_api = client.CustomObjectsApi()
        self.deployment_name = ''

        # Define action and observation spaces
        self.action_space = spaces.Discrete(len(self.ACTIONS))
        # Observations: CPU usage, memory usage, CPU request,
        # memory request, CPU limit, memory limit
        self.observation_space = spaces.Box(
            low=0,
            high=float('inf'),
            shape=(6,),
            dtype=float,
        )

    def step(self, action):
        self._apply_action(action)
        next_state = self._fetch_pod_metrics()
        reward = self._calculate_reward(next_state)
        done = False  # Define episode termination as needed
        return next_state, reward, done, {}

    def reset(self):
        return self._fetch_pod_metrics()

    def render(self, mode='human', close=False):  # pylint: disable=unused-argument
        pass

    def _apply_action(self, action):
        """
        Modify resource requests/limits based on selected action.
        """
        action_name = self.ACTIONS[action]
        pod_metrics = self._fetch_pod_metrics()
        cpu_usage, mem_usage, cpu_req, mem_req, cpu_lim, mem_lim = pod_metrics[0]

        # Adjust based on action
        if action_name == 'increase_cpu_request':
            cpu_req += 10
        elif action_name == 'decrease_cpu_request':
            cpu_req -= 10
        elif action_name == 'increase_memory_request':
            mem_req += 100
        elif action_name == 'decrease_memory_request':
            mem_req -= 100
        elif action_name == 'increase_cpu_limit':
            cpu_lim += 10
        elif action_name == 'decrease_cpu_limit':
            cpu_lim -= 10
        elif action_name == 'increase_memory_limit':
            mem_lim += 100
        elif action_name == 'decrease_memory_limit':
            mem_lim -= 100

        # Enforce minimums
        cpu_req = max(cpu_req, 50)
        mem_req = max(mem_req, 100)
        cpu_lim = max(cpu_lim, 100)
        mem_lim = max(mem_lim, 200)

        self._update_deployment_resources(
            self.deployment_name,
            cpu_lim,
            mem_lim,
            cpu_req,
            mem_req,
        )

    def _calculate_reward(self, state):
        """
        Reward is higher when usage is close to requests but
        below limits, with penalties near limits.
        """
        cpu_usage, mem_usage, cpu_req, mem_req, cpu_lim, mem_lim = state[0]
        reward = 0.0

        # Penalize deviation from requests
        reward -= abs(cpu_usage - cpu_req) * 0.5
        reward -= abs(mem_usage - mem_req) * 0.5

        # Penalize low headroom
        cpu_headroom = (cpu_lim - cpu_usage) / cpu_lim
        if cpu_headroom < 0.1:
            reward -= (0.1 - cpu_headroom) * 5

        mem_headroom = (mem_lim - mem_usage) / mem_lim
        if mem_headroom < 0.1:
            reward -= (0.1 - mem_headroom) * 10

        return reward

    def _fetch_pod_metrics(self):
        """
        Retrieve current pod metrics and deployed resource specs.
        """
        # Load deployment name and resources from YAML
        with open(YAML_FILE_PATH, 'r') as stream:
            docs = list(yaml.safe_load_all(stream))

        deployment_resources = {}
        for doc in docs:
            if doc.get('kind') == 'Deployment':
                name = doc['metadata']['name']
                self.deployment_name = name
                for container in doc['spec']['template']['spec']['containers']:
                    resources = container.get('resources', {})
                    deployment_resources[name] = resources

        # Fetch actual metrics
        group = 'metrics.k8s.io'
        version = 'v1beta1'
        plural = 'pods'

        pod_metrics = self.custom_api.list_namespaced_custom_object(
            group,
            version,
            self.namespace,
            plural,
        )

        metrics_data = []
        for item in pod_metrics.get('items', []):
            name = item['metadata']['name']
            deploy = name.rsplit('-', 2)[0]
            container = item['containers'][0]

            cpu_use = cpu_to_nano(container['usage'].get('cpu', '0'))
            mem_use = memory_to_Mi(container['usage'].get('memory', '0'))

            resources = deployment_resources.get(deploy, {})
            reqs = resources.get('requests', {})
            lims = resources.get('limits', {})

            cpu_req = cpu_to_nano(reqs.get('cpu', '0'))
            mem_req = memory_to_Mi(reqs.get('memory', '0'))
            cpu_lim = cpu_to_nano(lims.get('cpu', '0'))
            mem_lim = memory_to_Mi(lims.get('memory', '0'))

            metrics_data.append((
                cpu_use,
                mem_use,
                cpu_req,
                mem_req,
                cpu_lim,
                mem_lim,
            ))

        return metrics_data

    def _update_deployment_resources(
        self,
        deployment_name,
        cpu_lim,
        mem_lim,
        cpu_req,
        mem_req,
    ):
        """
        Patch the Kubernetes Deployment with new resource values.
        """
        # Format for Kubernetes
        cpu_lim_str = f"{int(cpu_lim / 10 ** 6)}m"
        mem_lim_str = f"{int(mem_lim / 1024)}Mi"
        cpu_req_str = f"{int(cpu_req / 10 ** 6)}m"
        mem_req_str = f"{int(mem_req / 1024)}Mi"

        # Update YAML file
        self._update_yaml_resources(
            cpu_lim_str,
            mem_lim_str,
            cpu_req_str,
            mem_req_str,
        )

        # Apply to cluster with retries
        for _ in range(3):
            deployment = self.apps_v1_api.read_namespaced_deployment(
                deployment_name,
                self.namespace,
            )
            deployment.spec.template.spec.containers[0].resources = (
                client.V1ResourceRequirements(
                    limits={
                        'cpu': cpu_lim_str,
                        'memory': mem_lim_str,
                    },
                    requests={
                        'cpu': cpu_req_str,
                        'memory': mem_req_str,
                    },
                )
            )
            try:
                self.apps_v1_api.replace_namespaced_deployment(
                    deployment_name,
                    self.namespace,
                    deployment,
                )
                break
            except kubernetes.client.exceptions.ApiException as exc:
                if exc.status == 409:
                    time.sleep(1)
                    continue
                raise

        # Wait for rollout
        timeout = 300
        start = time.time()
        while time.time() - start < timeout:
            status = self.apps_v1_api.read_namespaced_deployment(
                deployment_name,
                self.namespace,
            ).status
            if status.updated_replicas == status.replicas:
                self._wait_for_pod_metrics_length(len(status.replicas or []))
                return
            time.sleep(10)

        raise RuntimeError(
            f"Timeout waiting for deployment {deployment_name} rollout"
        )

    def _update_yaml_resources(
        self,
        cpu_lim_str,
        mem_lim_str,
        cpu_req_str,
        mem_req_str,
    ):
        """
        Update the local YAML config with new resource values.
        """
        try:
            with open(YAML_FILE_PATH, 'r') as stream:
                docs = list(yaml.safe_load_all(stream))

            for doc in docs:
                if doc.get('kind') != 'Deployment':
                    continue
                for container in doc['spec']['template']['spec']['containers']:
                    if container.get('name') != 'intensive-task-app':
                        continue
                    container['resources']['limits']['cpu'] = cpu_lim_str
                    container['resources']['limits']['memory'] = mem_lim_str
                    container['resources']['requests']['cpu'] = cpu_req_str
                    container['resources']['requests']['memory'] = mem_req_str

            with open(YAML_FILE_PATH, 'w') as outfile:
                yaml.dump_all(docs, outfile, default_flow_style=False)

        except yaml.YAMLError as exc:
            print(exc)

    def _wait_for_pod_metrics_length(
        self,
        expected_length,
        poll_interval=3,
    ):
        """
        Block until the pod metrics list has the expected number of items.
        """
        group = 'metrics.k8s.io'
        version = 'v1beta1'
        plural = 'pods'

        while True:
            pod_metrics = self.custom_api.list_namespaced_custom_object(
                group,
                version,
                self.namespace,
                plural,
            )
            current = len(pod_metrics.get('items', []))
            print(f"Found {current} of {expected_length} metrics items.")
            if current == expected_length:
                return
            time.sleep(poll_interval)
