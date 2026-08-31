from diffsynth.pipelines.flux2_image import Flux2ImagePipeline, ModelConfig
import torch


pipe = Flux2ImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="black-forest-labs/FLUX.2-klein-4B", origin_file_pattern="text_encoder/*.safetensors"),
        ModelConfig(model_id="black-forest-labs/FLUX.2-klein-4B", origin_file_pattern="transformer/*.safetensors"),
        ModelConfig(model_id="black-forest-labs/FLUX.2-klein-4B", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    tokenizer_config=ModelConfig(model_id="black-forest-labs/FLUX.2-klein-4B", origin_file_pattern="tokenizer/"),
)
prompt = "A studio product shot focuses on the lower half of a person wearing black athletic joggers. The pants feature a drawstring waist, zippered side pockets, and a prominent vertical white text graphic reading \"EASI HORSE\" running down the left leg. The person is standing in a neutral pose against a plain, light-grey studio background. The lighting is bright and artificial, creating soft shadows that highlight the texture and fit of the synthetic fabric."
image = pipe(prompt, seed=0, rand_device="cuda", num_inference_steps=4)
image.save("image_FLUX.2-klein-4B.jpg")

# prompt = "change the color of the clothes to red"
# image = pipe(prompt, edit_image=[image], seed=1, rand_device="cuda", num_inference_steps=4)
# image.save("image_edit_FLUX.2-klein-4B.jpg")
