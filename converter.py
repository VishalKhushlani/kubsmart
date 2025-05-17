"""
Utility functions for converting CPU and memory resource strings.
"""


def cpu_to_nano(value):
    """
    Convert a CPU resource string to nano cores.

    Examples:
        '500m' -> 500 * 10**6
        '2'    -> 2
        '100n' -> 100

    Args:
        value (str): CPU string (e.g., '500m', '2', '100n').

    Returns:
        int: CPU in nano cores.
    """
    if value.endswith('m'):
        return int(value.rstrip('m')) * 10 ** 6
    if value.endswith('n'):
        return int(value.rstrip('n'))
    return int(value)



def memory_to_Mi(value):
    """
    Convert a memory resource string to mebibyte (Mi).

    Examples:
        '1Gi'  -> 1 * 1024 * 1024
        '512Mi'-> 512 * 1024
        '1024Ki'-> 1024
        '1048576'-> 1.0

    Args:
        value (str): Memory string (e.g., '1Gi', '512Mi', '1024Ki', '1048576').

    Returns:
        float: Memory in mebibyte.
    """
    if value.endswith('Gi'):
        return int(value.rstrip('Gi')) * 1024 * 1024
    if value.endswith('Mi'):
        return int(value.rstrip('Mi')) * 1024
    if value.endswith('Ki'):
        return int(value.rstrip('Ki'))
    return int(value) / (1024 ** 2)
