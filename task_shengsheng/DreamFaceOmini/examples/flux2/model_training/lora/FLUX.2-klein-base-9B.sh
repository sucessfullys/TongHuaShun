# accelerate launch examples/flux2/model_training/train.py \
#   --dataset_base_path data/example_image_dataset \
#   --dataset_metadata_path data/example_image_dataset/metadata.csv \
#   --max_pixels 1048576 \
#   --dataset_repeat 50 \
#   --model_id_with_origin_paths "black-forest-labs/FLUX.2-klein-9B:text_encoder/*.safetensors,black-forest-labs/FLUX.2-klein-base-9B:transformer/*.safetensors,black-forest-labs/FLUX.2-klein-9B:vae/diffusion_pytorch_model.safetensors" \
#   --tokenizer_path "black-forest-labs/FLUX.2-klein-9B:tokenizer/" \
#   --learning_rate 1e-4 \
#   --num_epochs 5 \
#   --remove_prefix_in_ckpt "pipe.dit." \
#   --output_path "./models/train/FLUX.2-klein-base-9B_lora" \
#   --lora_base_model "dit" \
#   --lora_target_modules "to_q,to_k,to_v,to_out.0,add_q_proj,add_k_proj,add_v_proj,to_add_out,linear_in,linear_out,to_qkv_mlp_proj,single_transformer_blocks.0.attn.to_out,single_transformer_blocks.1.attn.to_out,single_transformer_blocks.2.attn.to_out,single_transformer_blocks.3.attn.to_out,single_transformer_blocks.4.attn.to_out,single_transformer_blocks.5.attn.to_out,single_transformer_blocks.6.attn.to_out,single_transformer_blocks.7.attn.to_out,single_transformer_blocks.8.attn.to_out,single_transformer_blocks.9.attn.to_out,single_transformer_blocks.10.attn.to_out,single_transformer_blocks.11.attn.to_out,single_transformer_blocks.12.attn.to_out,single_transformer_blocks.13.attn.to_out,single_transformer_blocks.14.attn.to_out,single_transformer_blocks.15.attn.to_out,single_transformer_blocks.16.attn.to_out,single_transformer_blocks.17.attn.to_out,single_transformer_blocks.18.attn.to_out,single_transformer_blocks.19.attn.to_out,single_transformer_blocks.20.attn.to_out,single_transformer_blocks.21.attn.to_out,single_transformer_blocks.22.attn.to_out,single_transformer_blocks.23.attn.to_out" \
#   --lora_rank 32 \
#   --use_gradient_checkpointing

# Edit
# accelerate launch examples/flux2/model_training/train.py \
#   --dataset_base_path data/example_image_dataset \
#   --dataset_metadata_path data/example_image_dataset/metadata_qwen_imgae_edit_multi.json \
#   --data_file_keys "image,edit_image" \
#   --extra_inputs "edit_image" \
#   --max_pixels 1048576 \
#   --dataset_repeat 50 \
#   --model_id_with_origin_paths "black-forest-labs/FLUX.2-klein-9B:text_encoder/*.safetensors,black-forest-labs/FLUX.2-klein-base-9B:transformer/*.safetensors,black-forest-labs/FLUX.2-klein-9B:vae/diffusion_pytorch_model.safetensors" \
#   --tokenizer_path "black-forest-labs/FLUX.2-klein-9B:tokenizer/" \
#   --learning_rate 1e-4 \
#   --num_epochs 5 \
#   --remove_prefix_in_ckpt "pipe.dit." \
#   --output_path "./models/train/FLUX.2-klein-base-9B_lora" \
#   --lora_base_model "dit" \
#   --lora_target_modules "to_q,to_k,to_v,to_out.0,add_q_proj,add_k_proj,add_v_proj,to_add_out,linear_in,linear_out,to_qkv_mlp_proj,single_transformer_blocks.0.attn.to_out,single_transformer_blocks.1.attn.to_out,single_transformer_blocks.2.attn.to_out,single_transformer_blocks.3.attn.to_out,single_transformer_blocks.4.attn.to_out,single_transformer_blocks.5.attn.to_out,single_transformer_blocks.6.attn.to_out,single_transformer_blocks.7.attn.to_out,single_transformer_blocks.8.attn.to_out,single_transformer_blocks.9.attn.to_out,single_transformer_blocks.10.attn.to_out,single_transformer_blocks.11.attn.to_out,single_transformer_blocks.12.attn.to_out,single_transformer_blocks.13.attn.to_out,single_transformer_blocks.14.attn.to_out,single_transformer_blocks.15.attn.to_out,single_transformer_blocks.16.attn.to_out,single_transformer_blocks.17.attn.to_out,single_transformer_blocks.18.attn.to_out,single_transformer_blocks.19.attn.to_out,single_transformer_blocks.20.attn.to_out,single_transformer_blocks.21.attn.to_out,single_transformer_blocks.22.attn.to_out,single_transformer_blocks.23.attn.to_out" \
#   --lora_rank 32 \
#   --use_gradient_checkpointing

# shellcheck disable=SC1091
source "${SHENSHENG_ROOT:-/mnt/data/image-edit/datasets/shensheng}/config/paths.sh"
export DIFFSYNTH_MODEL_BASE_PATH="${SHENSHENG_MODELS}"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export PYTHONPATH="$(pwd):$PYTHONPATH"
# Edit
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 accelerate launch examples/flux2/model_training/train.py \
  --dataset_base_path "/" \
  --dataset_metadata_path "${SHENSHENG_DATA}/merged_train_4349.jsonl" \
  --data_file_keys "image,edit_image" \
  --extra_inputs "edit_image" \
  --max_pixels 2073600 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "black-forest-labs/FLUX.2-klein-9B:text_encoder/*.safetensors,black-forest-labs/FLUX.2-klein-base-9B:transformer/*.safetensors,black-forest-labs/FLUX.2-klein-9B:vae/diffusion_pytorch_model.safetensors" \
  --tokenizer_path "black-forest-labs/FLUX.2-klein-9B:tokenizer/" \
  --learning_rate 1e-4 \
  --num_epochs 100 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/FLUX.2-klein-base-9B_lora_double_person_results_captions_merged-face_id_weight0.1" \
  --lora_base_model "dit" \
  --lora_target_modules "to_q,to_k,to_v,to_out.0,add_q_proj,add_k_proj,add_v_proj,to_add_out,linear_in,linear_out,to_qkv_mlp_proj,single_transformer_blocks.0.attn.to_out,single_transformer_blocks.1.attn.to_out,single_transformer_blocks.2.attn.to_out,single_transformer_blocks.3.attn.to_out,single_transformer_blocks.4.attn.to_out,single_transformer_blocks.5.attn.to_out,single_transformer_blocks.6.attn.to_out,single_transformer_blocks.7.attn.to_out,single_transformer_blocks.8.attn.to_out,single_transformer_blocks.9.attn.to_out,single_transformer_blocks.10.attn.to_out,single_transformer_blocks.11.attn.to_out,single_transformer_blocks.12.attn.to_out,single_transformer_blocks.13.attn.to_out,single_transformer_blocks.14.attn.to_out,single_transformer_blocks.15.attn.to_out,single_transformer_blocks.16.attn.to_out,single_transformer_blocks.17.attn.to_out,single_transformer_blocks.18.attn.to_out,single_transformer_blocks.19.attn.to_out,single_transformer_blocks.20.attn.to_out,single_transformer_blocks.21.attn.to_out,single_transformer_blocks.22.attn.to_out,single_transformer_blocks.23.attn.to_out" \
  --lora_rank 32 \
  --use_gradient_checkpointing \
  --lora_checkpoint "${SHENSHENG_WEIGHTS_LEGACY}/v2.151e20.safetensors" \
  --face_id_mode firered --face_id_weight 0.1 --face_sigma_cap 0.9 \
  --arcface_ckpt_path /mnt/data/image-edit/models/arcface/weights/arcface-r100-glint360k.pth \
  --insightface_root "${SHENSHENG_MODELS}/insightface" \

