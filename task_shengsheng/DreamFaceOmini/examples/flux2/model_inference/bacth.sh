export DIFFSYNTH_MODEL_BASE_PATH="/mnt/data/image-edit/datasets/shensheng/models"
export DIFFSYNTH_SKIP_DOWNLOAD=true

python /mnt/data/image-edit/datasets/shensheng/code/dev/DiffSynth-Studio-new/examples/flux2/model_inference/batch_infer.py \
  --jsonl /mnt/data/image-edit/datasets/shensheng/datasets/benchmark/明星/demo/02-2.json\
  --output ./exp_out/batch_results_lora_cfg4_scale1_s2guidance-0.0_v2.161_e0\
  --num 69 \
  --seed 42 \
  --steps 50 \
  --gpus 0,1,2,3,4,5,6,7 \
  --cfg 4.0 \
  --height 1152 \
  --width 896 \
  --lora /mnt/data/image-edit/datasets/shensheng/code/stable/Dream/models/train/FLUX.2-klein-base-9B_lora_double_person_results_captions_merged/epoch-0.safetensors
  # --lora /mnt/data/image-edit/datasets/shensheng/v2.14-continous_d1e4.safetensors
    # --offload \

