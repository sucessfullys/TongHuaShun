import os
import numpy as np

gpu_id=0

qargs = ['index', 'memory.free', 'memory.total', 'utilization.gpu']
cmd = 'nvidia-smi --query-gpu={} --format=csv'.format(','.join(qargs))
results = os.popen(cmd).readlines()
result_np = []
for line in results[1:]:
    result_np.append([''.join(filter(str.isdigit, word)) for word in line.split(',')])
result_np = np.array(result_np)
print(result_np)