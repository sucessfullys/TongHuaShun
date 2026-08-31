# -*- coding: utf-8 -*-
import os
import time
import argparse
import random
import json
import numpy as np
import torch
from multiprocessing import Process, Manager
from datetime import datetime, timedelta


def set_parser():
    parser = argparse.ArgumentParser(description='GPU Memory Manager')
    parser.add_argument('-p', '--proportion', type=float, default=0.31,
                        help='The ratio of GPU free memory to trigger alarm')
    parser.add_argument('-u', '--upper_limit', type=float, default=0.5,
                        help='The ratio of GPU free memory to total memory')
    parser.add_argument('-i', '--interval', type=float, default=1,
                        help='Operation interval(s)')
    parser.add_argument('-c', '--cooldown', type=float, default=60.0,
                        help='Cooldown period after stopping a worker before starting it again (seconds)')
    parser.add_argument('--idle_threshold', type=float, default=60.0,
                        help='If GPU utilization stays below idle_util_limit for this many seconds, start worker even if memory is occupied')
    parser.add_argument('--idle_util_limit', type=int, default=5,
                        help='Utilization percentage considered as idle (default 5%%)')
    parser.add_argument('--idle_memory_mb', type=float, default=1.0,
                        help='Approximate memory budget in MB for lightweight idle worker')
    parser.add_argument('-e', '--email_conf', type=str, default='./email_conf.json',
                        help='The path to email config')
    return parser.parse_args()


def parse(results):
    result_np = []
    for line in results[1:]:
        result_np.append([''.join(filter(str.isdigit, word)) for word in line.split(',')])
    result_np = np.array(result_np)
    return result_np


def calculate_size(total_memory_mib, target_proportion):
    total_memory_bytes = total_memory_mib * 1048 * 1024
    target_memory = total_memory_bytes * target_proportion
    element_size = 8
    total_elements = target_memory / element_size / 3
    size = int(total_elements ** (1 / 3))
    return size


def calculate_idle_matrix_size(memory_mb):
    target_bytes = max(memory_mb, 0.25) * 1024 * 1024
    # Two float32 input matrices share the configured memory budget.
    size = int((target_bytes / (2 * 4)) ** 0.5)
    return max(size, 64)


def query_gpu_utilization(gpu_id):
    qargs = ['index', 'utilization.gpu']
    cmd = f"nvidia-smi -i {gpu_id} --query-gpu={','.join(qargs)} --format=csv"
    results = os.popen(cmd).readlines()
    gpu_status = parse(results)
    for status in gpu_status:
        g_id, utilization = status
        if int(g_id) == gpu_id:
            return int(utilization)
    return 0


def worker(gpu_id, total_memory, proportion, upper_limit, light_mode=False, idle_memory_mb=1.0, idle_util_limit=5):
    device = f'cuda:{gpu_id}'
    if light_mode:
        size = calculate_idle_matrix_size(idle_memory_mb)
        a = torch.randn((size, size), dtype=torch.float32, device=device)
        b = torch.randn((size, size), dtype=torch.float32, device=device)
        mode_desc = f"lightweight mode with ~{idle_memory_mb}MB inputs"
    else:
        size = calculate_size(total_memory, proportion)
        a = torch.randn((size, size, size), dtype=torch.double, device=device)
        b = torch.randn((size, size, size), dtype=torch.double, device=device)
        mode_desc = f"matrix size {size} to increase memory usage"
    current_time = datetime.now()
    formatted_time = current_time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"{formatted_time}  Worker on GPU {gpu_id} running in {mode_desc}.")
    count = 0
    while True:
        count += 1
        c = torch.matmul(a, b)
        if random.random() > 0.5:
            time.sleep(0.01)
        # 检查当前GPU的使用情况
        if count >= 20:
            count = 0
            if light_mode:
                utilization = query_gpu_utilization(gpu_id)
                if utilization > idle_util_limit:
                    current_time = datetime.now()
                    formatted_time = current_time.strftime('%Y-%m-%d %H:%M:%S')
                    print(f"{formatted_time}  GPU {gpu_id} utilization is {utilization}%, stopping lightweight worker.")
                    break
            else:
                used_proportion = query_gpu_usage(gpu_id)
                if used_proportion > upper_limit:
                    current_time = datetime.now()
                    formatted_time = current_time.strftime('%Y-%m-%d %H:%M:%S')
                    print(f"{formatted_time}  GPU {gpu_id} usage is above the upper limit. Stopping worker.")
                    break
    current_time = datetime.now()
    formatted_time = current_time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"{formatted_time}  Worker on GPU {gpu_id} stopped.")


def query_gpu():
    """ 查看单个GPU的信息 """
    qargs = ['index', 'memory.free', 'memory.total', 'utilization.gpu']
    cmd = 'nvidia-smi --query-gpu={} --format=csv'.format(','.join(qargs))
    try:
        results = os.popen(cmd).readlines()
        return parse(results)
    except Exception as e:
        print(f"Failed to query GPU status: {e}")
        return np.array([])


def query_gpu_usage(gpu_id):
    """ 查看单个GPU占用的比例 """
    qargs = ['index', 'memory.free', 'memory.total']
    cmd = f"nvidia-smi -i {gpu_id} --query-gpu={','.join(qargs)} --format=csv"
    results = os.popen(cmd).readlines()
    gpu_status = parse(results)
    for status in gpu_status:
        g_id, free_memory, total_memory = status
        if int(g_id) == gpu_id:
            return 1 - int(free_memory) / int(total_memory)
    return 0


def manage_gpus(args):

    gpu_status = query_gpu()
    if gpu_status.size == 0:
        print("No GPU status information available.")
        return

    processes = {}
    process_modes = {}
    cooldown_times = {}
    idle_seconds = {}

    while True:
        gpu_status = query_gpu()
        if gpu_status.size == 0:
            print("No GPU status information available.")
            time.sleep(args.interval)
            continue

        gpu_status = gpu_status.astype('int')
        current_time = datetime.now()
        for status in gpu_status:
            gpu_id, free_memory, total_memory, utilization = status
            used_proportion = 1 - free_memory / total_memory
            used_memory_mb = total_memory - free_memory

            if utilization <= args.idle_util_limit:
                idle_seconds[gpu_id] = idle_seconds.get(gpu_id, 0) + args.interval
            else:
                idle_seconds[gpu_id] = 0

            low_mem_idle = utilization == 0 and used_proportion < 0.3
            prolonged_idle = idle_seconds.get(gpu_id, 0) >= args.idle_threshold

            should_start = low_mem_idle or prolonged_idle

            if should_start:
                existing_process = processes.get(gpu_id)
                existing_mode = process_modes.get(gpu_id)
                if low_mem_idle and existing_process is not None and existing_process.is_alive() and existing_mode == "light":
                    formatted_time = current_time.strftime('%Y-%m-%d %H:%M:%S')
                    print(f"{formatted_time}  GPU {gpu_id} used_proportion {used_proportion:.2f} utilization 0%, replacing lightweight worker with heavy worker.")
                    existing_process.terminate()
                    existing_process.join(timeout=5)
                    processes.pop(gpu_id, None)
                    process_modes.pop(gpu_id, None)

                if gpu_id not in processes or not processes[gpu_id].is_alive():
                    if gpu_id not in cooldown_times or (current_time - cooldown_times[gpu_id]).total_seconds() > args.cooldown:
                        formatted_time = current_time.strftime('%Y-%m-%d %H:%M:%S')
                        reason = "utilization 0% and used_proportion < 30%" if low_mem_idle else f"idle for {idle_seconds[gpu_id]:.0f}s"
                        print(f"{formatted_time}  Starting worker for GPU {gpu_id}, used_memory {used_memory_mb}MiB used_proportion {used_proportion:.2f} utilization {utilization}% reason: {reason}")
                        light_mode = prolonged_idle and not low_mem_idle
                        p = Process(target=worker, args=(
                            gpu_id,
                            total_memory,
                            args.proportion,
                            args.upper_limit,
                            light_mode,
                            args.idle_memory_mb,
                            args.idle_util_limit,
                        ))
                        p.start()
                        processes[gpu_id] = p
                        process_modes[gpu_id] = "light" if light_mode else "heavy"
                        idle_seconds[gpu_id] = 0
            elif used_proportion > args.upper_limit:
                if gpu_id in processes and processes[gpu_id].is_alive():
                    cooldown_times[gpu_id] = current_time

        time.sleep(args.interval)

if __name__ == "__main__":
    args = set_parser()
    manage_gpus(args)


