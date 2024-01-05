import logging

import psutil
import torch


def get_resource_metrics():
    # resources metrics
    metrics: dict = {}
    in_gb = 1024.0 ** 3
    metrics["ram_gb_total"] = psutil.virtual_memory().total / in_gb
    metrics["ram_gb_available"] = psutil.virtual_memory().available / in_gb
    metrics["ram_gb_free"] = psutil.virtual_memory().free / in_gb
    metrics["ram_gb_used"] = psutil.virtual_memory().used / in_gb
    metrics["ram_gb_used_main_thread"] = psutil.Process().memory_info().rss / in_gb
    metrics["ram_percent"] = psutil.virtual_memory().percent
    metrics["cpu_percent"] = psutil.cpu_percent()

    if torch.cuda.is_available():
        metrics["gpu_gb_total"] = torch.cuda.get_device_properties(0).total_memory / in_gb
        metrics["cuda_gb_gpu_memory_used"] = (torch.cuda.mem_get_info()[1] -
                                              torch.cuda.mem_get_info()[0]) / in_gb

    ram_percent_limit = 95
    if metrics["ram_percent"] > ram_percent_limit:
        logging.warning("ram_percent > " + str(ram_percent_limit))


    return metrics
