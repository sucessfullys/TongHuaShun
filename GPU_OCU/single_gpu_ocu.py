import torch
import time
import pynvml

# ── 配置 ──────────────────────────────────────────────────
gpu_ids            = [2, 3, 4, 5, 6, 7]  # 监控/占用的 GPU 列表
memory_gb_per_gpu  = 10                   # 每卡占用显存 (GB)
zero_util_threshold = 30                  # 利用率持续为 0 多少秒后触发占卡
check_interval     = 5                    # 每隔多少秒检查一次利用率
# ─────────────────────────────────────────────────────────


def get_gpu_utilization(handle):
    """返回 GPU 计算利用率百分比 (0-100)。"""
    return pynvml.nvmlDeviceGetUtilizationRates(handle).gpu


def allocate_tensor(gpu_id, memory_gb):
    """在指定 GPU 上分配显存张量并返回。"""
    device = torch.device(f'cuda:{gpu_id}')
    num_elements = int(memory_gb * 1024**3 // 4)
    tensor = torch.empty(num_elements, dtype=torch.float32, device=device)
    print(f"[占卡] GPU {gpu_id}: 已分配 {memory_gb} GB 显存。")
    return tensor


def release_tensor(gpu_id, tensor):
    """释放张量并清空 GPU 显存缓存。"""
    del tensor
    torch.cuda.empty_cache()
    print(f"[释放] GPU {gpu_id}: 检测到利用率，已释放占用显存。")


def monitor_and_occupy(gpu_ids, memory_gb_per_gpu, zero_util_threshold, check_interval):
    pynvml.nvmlInit()

    handles      = {gid: pynvml.nvmlDeviceGetHandleByIndex(gid) for gid in gpu_ids}
    zero_seconds = {gid: 0   for gid in gpu_ids}  # 各 GPU 连续利用率为 0 的秒数
    tensors      = {gid: None for gid in gpu_ids}  # 各 GPU 当前占用的张量

    print(f"开始监控 GPU {gpu_ids}，利用率持续 {zero_util_threshold}s 为 0 时自动占卡。")

    try:
        while True:
            for gid in gpu_ids:
                util = get_gpu_utilization(handles[gid])

                if util == 0:
                    zero_seconds[gid] += check_interval
                    status = f"利用率=0，已持续 {zero_seconds[gid]}s"
                else:
                    # 利用率非 0：若当前有占卡则释放
                    if tensors[gid] is not None:
                        release_tensor(gid, tensors[gid])
                        tensors[gid] = None
                    zero_seconds[gid] = 0
                    status = f"利用率={util}%，未占卡"

                # 达到阈值且尚未占卡 → 启动占卡
                if zero_seconds[gid] >= zero_util_threshold and tensors[gid] is None:
                    tensors[gid] = allocate_tensor(gid, memory_gb_per_gpu)

                print(f"  GPU {gid}: {status}")

            time.sleep(check_interval)

    except KeyboardInterrupt:
        print("\n程序中断，释放所有占用显存...")
        for gid in gpu_ids:
            if tensors[gid] is not None:
                release_tensor(gid, tensors[gid])
    finally:
        pynvml.nvmlShutdown()


monitor_and_occupy(gpu_ids, memory_gb_per_gpu, zero_util_threshold, check_interval)