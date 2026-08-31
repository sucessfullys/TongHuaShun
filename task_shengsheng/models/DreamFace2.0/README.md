---
language:
- en
license: other
license_name: flux-non-commercial-license
tags:
- image-generation
- image-editing
- identity-preservation
- portrait
- face-consistency
- flux
- diffusion-single-file
pipeline_tag: image-to-image
library_name: diffusers
base_model: black-forest-labs/FLUX.2-klein-9B
---

# DreamFace2.0

DreamFace2.0 is a portrait identity-consistency image editing model based on
`black-forest-labs/FLUX.2-klein-9B`. It is built by merging a DreamFace identity
LoRA into the FLUX.2 Klein 9B transformer backbone, while keeping the original
Diffusers model layout and the single-file backbone layout.

The model is designed for portrait and character editing workflows where the
generated image should follow the edit prompt while preserving the subject's
facial identity and overall human appearance.

## Examples

Each example below shows the reference image and the edited result side by side.

![DreamFace identity editing example 1](./dreamface_identity_1.webp)

![DreamFace identity editing example 2](./dreamface_identity_2.webp)

![DreamFace identity editing example 3](./dreamface_identity_3.webp)

## Key Features

1. Portrait identity preservation for image editing.
2. Supports single-reference and multi-reference FLUX.2 Klein workflows.
3. Compatible with Diffusers folder loading via `Flux2KleinPipeline`.
4. Includes a merged single-file backbone for ComfyUI-style loading.
5. Keeps the original FLUX.2 Klein 9B tokenizer, text encoder, VAE, scheduler,
   and pipeline configuration.

## Model Layout

This repository follows the same structure as `black-forest-labs/FLUX.2-klein-9B`:

```text
DreamFace2.0/
├── model_index.json
├── scheduler/
├── tokenizer/
├── text_encoder/
├── transformer/
├── vae/
└── flux-2-klein-9b.safetensors
```

The `transformer/` directory contains the merged Diffusers transformer weights.
The root `flux-2-klein-9b.safetensors` file contains the merged single-file
backbone for tools that load the FLUX.2 Klein backbone separately.

## Using With Diffusers

Install or upgrade Diffusers to a version that supports FLUX.2 Klein:

```bash
pip install -U diffusers transformers accelerate safetensors
```

Load the full model directory:

```python
import torch
from PIL import Image
from diffusers import Flux2KleinPipeline

model_path = "/mnt/image-edit/datasets/duanyufa/task_shengsheng/models/DreamFace2.0"

pipe = Flux2KleinPipeline.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
)
pipe.to("cuda")

reference = Image.open("reference.png").convert("RGB")
prompt = "A natural portrait photo of the same person, high quality, detailed face"

image = pipe(
    prompt=prompt,
    image=reference,
    height=1152,
    width=896,
    guidance_scale=1.0,
    num_inference_steps=4,
    generator=torch.Generator(device="cuda").manual_seed(42),
).images[0]

image.save("dreamface_output.png")
```

For lower VRAM usage, replace `pipe.to("cuda")` with:

```python
pipe.enable_model_cpu_offload()
```

## Loading Additional LoRA Weights

DreamFace2.0 can be used as a new base model. Additional FLUX.2 Klein LoRA
weights can still be loaded on top:

```python
pipe.load_lora_weights("/path/to/new_lora.safetensors", adapter_name="extra_lora")
```

The additional LoRA must target the same FLUX.2 Klein transformer architecture.

## Using With ComfyUI

Use the merged single-file backbone:

```text
flux-2-klein-9b.safetensors
```

The accompanying `text_encoder/`, `tokenizer/`, `vae/`, and `scheduler/`
directories are kept from the original FLUX.2 Klein 9B release so the model can
also be loaded as a complete package by tools that support Diffusers-style model
folders.

## Training and Merge Details

DreamFace2.0 was produced from:

```text
Base model: black-forest-labs/FLUX.2-klein-9B
Merged LoRA: dreamface_nft_vlm_gt_identity_gemma4 checkpoint-80
LoRA rank: 32
LoRA alpha: 32
Merge scale: 1.0
```

The merge metadata is stored in:

```text
merge_metadata.json
```

## Intended Use

DreamFace2.0 is intended for research and creative portrait editing workflows,
including identity-preserving transformations, stylized portrait generation, and
multi-reference human image editing.

## Limitations

- Identity preservation is not guaranteed for every input, pose, age, lighting,
  occlusion, or prompt.
- The model may fail on very small faces, heavy blur, extreme profile views, or
  conflicting multi-reference inputs.
- Prompt following and identity consistency can trade off against each other.
- Text rendering, factual content, and fine-grained physical details may be
  inaccurate.
- The model may inherit biases and limitations from the base FLUX.2 Klein 9B
  model and from the DreamFace training data.

## Responsible Use

Do not use this model to create misleading, abusive, non-consensual, illegal, or
harmful content. Deployments should include appropriate user consent, safety
filtering, and review mechanisms, especially for face and identity-related use
cases.

## License

This model is based on `black-forest-labs/FLUX.2-klein-9B` and inherits the
FLUX Non-Commercial License. Use of the model and its derivatives must comply
with the original license and acceptable use policy.
