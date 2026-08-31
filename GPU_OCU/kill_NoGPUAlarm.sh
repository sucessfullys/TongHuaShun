#!/bin/bash

# 获取所有包含 NoGPUAlarmNew.py 的进程
processes=$(ps aux | grep NoGPUAlarmNew.py | grep -v grep | awk '{print $2}')

# 检查是否有找到的进程
if [ -z "$processes" ]; then
  echo "No processes found with NoGPUAlarmNew.py"
else
  # 杀掉所有找到的进程
  for pid in $processes
  do
    kill -9 $pid
    echo "Killed process with PID: $pid"
  done
fi
