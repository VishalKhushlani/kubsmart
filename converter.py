def cpu_to_nano(value):
    if value.endswith("m"):
        return int(value.rstrip("m")) * 10**6
    if value.endswith("n"):
        return int(value.rstrip("n"))
    return int(value)


def memory_to_Mi(value):
    if value.endswith("Gi"):
        return int(value.rstrip("Gi")) * 1024 * 1024
    if value.endswith("Mi"):
        return int(value.rstrip("Mi")) * 1024
    if value.endswith("Ki"):
        return int(value.rstrip("Ki"))
    return int(value) / (1024**2)
