python -m vllm.entrypoints.openai.api_server \
  --model /mnt/models/Gemma-4-31B-it \
  --served-model-name gemma4 \
  --tensor-parallel-size 2 \
  --host 0.0.0.0 \
  --port 25931 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 32768

# --max-model-len 32768
#   --max-model-len 12288