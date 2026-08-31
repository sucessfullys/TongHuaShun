# -*- coding: utf-8 -*-
import os
import time
import argparse
import _thread
import random
import json
from smtplib import SMTP_SSL
from email.mime.text import MIMEText
from email.utils import formataddr
import numpy as np
import torch
import gc
import subprocess


def set_parser():
    parser = argparse.ArgumentParser(description='GPU Memory Manager')
    parser.add_argument('-p', '--proportion', type=float, default=0.31,
                        help='The ratio of GPU free memory to trigger alarm')
    parser.add_argument('-u', '--upper_limit', type=float, default=0.51,
                        help='The ratio of GPU free memory to total memory')
    parser.add_argument('-i', '--interval', type=float, default=0.1,
                        help='Operation interval(s)')
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
    # 计算目标显存占用大小（字节）
    total_memory_bytes = total_memory_mib * 1024 * 1024  # 将 MiB 转换为字节
    target_memory = total_memory_bytes * target_proportion
    # 计算矩阵元素大小（double 类型，每个元素 8 字节）
    element_size = 8
    # 计算目标矩阵元素总数
    total_elements = target_memory / element_size / 3
    # 计算目标矩阵的 size（假设立方体矩阵）
    size = int(total_elements ** (1/3))
    return size


class EmailSender(object):
    def __init__(self, host_server, user, pwd, sender):
        self.host_server = host_server
        self.user = user
        self.pwd = pwd
        self.sender = sender

    def send_email(self, receiver, subject, content):
        receiver = [receiver] if isinstance(receiver, str) else receiver
        message = MIMEText(content, 'plain', 'utf-8')
        message['Subject'] = subject
        message['From'] = formataddr(("GPUSnatcher", self.sender))
        message['To'] = ", ".join(receiver)

        try:
            smtp_obj = SMTP_SSL(self.host_server)
            smtp_obj.ehlo(self.host_server)
            smtp_obj.login(self.user, self.pwd)
            smtp_obj.sendmail(self.sender, receiver, message.as_string())
            smtp_obj.quit()
            print("The mail was sent successfully.")
        except Exception as e:
            print(e)


def worker(gpu_id, total_memory, target_proportion, stop_signal):
    size = calculate_size(total_memory, target_proportion)
    a = torch.randn((size, size, size), dtype=torch.double, device=f'cuda:{gpu_id}')
    b = torch.randn((size, size, size), dtype=torch.double, device=f'cuda:{gpu_id}')
    print(f"Worker on GPU {gpu_id} running with matrix size {size} to increase memory usage.")
    while gpu_id not in stop_signal or not stop_signal[gpu_id]:
        c = torch.matmul(a, b)
        if random.random() > 0.5:
            time.sleep(0.01)
    del a
    del b
    del c

    print(f"Worker on GPU {gpu_id} stopped.")


def query_gpu():
    qargs = ['index', 'memory.free', 'memory.total']
    cmd = 'nvidia-smi --query-gpu={} --format=csv'.format(','.join(qargs))
    result = subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output = result.stdout.decode('utf-8').splitlines()
    return parse(output)


def manage_gpus(args):
    with open(args.email_conf, "r") as f:
        email_conf = json.load(f)
    email_sender = EmailSender(email_conf['host'],
                               email_conf['user'],
                               email_conf['pwd'],
                               email_conf['sender'])
    workers = {}
    stop_signals = {}
    print('Start!')
    while True:
        gpu_status = query_gpu()

        gpu_status = gpu_status.astype('int')
        for status in gpu_status:
            gpu_id, free_memory, total_memory = status
            used_proportion = 1 - free_memory / total_memory
            if used_proportion < args.proportion:
                if gpu_id not in workers:
                    stop_signals[gpu_id] = False
                    _thread.start_new_thread(worker, (gpu_id, total_memory, args.proportion, stop_signals))
                    workers[gpu_id] = gpu_id
                    # time.sleep(0.01)
            elif used_proportion > args.upper_limit:
                if gpu_id in workers:
                    stop_signals[gpu_id] = True
                    # time.sleep(0.01)
                    # del workers[gpu_id]
                    # del stop_signals[gpu_id]

        gc.collect()
        # torch.cuda.empty_cache()
        print('running!')


if __name__ == "__main__":
    args = set_parser()
    manage_gpus(args)
