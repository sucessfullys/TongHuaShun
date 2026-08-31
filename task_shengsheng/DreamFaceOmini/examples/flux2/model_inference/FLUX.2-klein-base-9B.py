from diffsynth.pipelines.flux2_image import Flux2ImagePipeline, ModelConfig
import torch
from PIL import Image


pipe = Flux2ImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="black-forest-labs/FLUX.2-klein-9B", origin_file_pattern="text_encoder/*.safetensors"),
        ModelConfig(model_id="black-forest-labs/FLUX.2-klein-9B", origin_file_pattern="transformer/*.safetensors"),
        ModelConfig(model_id="black-forest-labs/FLUX.2-klein-9B", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    tokenizer_config=ModelConfig(model_id="black-forest-labs/FLUX.2-klein-9B", origin_file_pattern="tokenizer/"),
)
# prompt = "Masterpiece, best quality. Anime-style portrait of a woman in a blue dress, underwater, surrounded by colorful bubbles."
# image = pipe(prompt, seed=0, rand_device="cuda", num_inference_steps=50, cfg_scale=4)
# image.save("image_FLUX.2-klein-base-9B.jpg")

image = Image.open("/mnt/data/image-edit/datasets/shensheng/code/dev/DiffSynth-Studio-new/examples/flux2/model_inference/cluster_33_2.jpg")
prompt = "The figure from the reference stands confidently with arms crossed over a broad chest, chin lifted while gazing intensely into the distance, embodying a pose of rugged authority and calm determination. This central character is rendered in a hyperrealistic oil painting style, where thick, visible brushstrokes mimic the texture of canvas while preserving the subject's exact facial structure and identity. Soft, golden-hour sunlight filters through dense forest foliage, casting dappled shadows across the detailed uniform and highlighting the contours of the face. The composition balances the dramatic stance with the lush, natural backdrop, ensuring the artistic medium enhances rather than obscures the human form's powerful presence and specific emotional expression."
image = pipe(prompt, edit_image=[image], seed=1, rand_device="cuda", num_inference_steps=4, cfg_scale=1)
image.save("image_edit_FLUX.2-klein-base-9B.jpg")
