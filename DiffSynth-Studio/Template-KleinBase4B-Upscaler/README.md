---
license: apache-2.0
---
# Templates-Super-Resolution (FLUX.2-klein-base-4B)

This model is part of the open-source Diffusion Templates series by [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio). Specifically designed for image super-resolution, it takes low-resolution input images and redraws them with rich high-definition details while preserving the original composition and semantics.

* Open-source code: [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio)
* Technical report: [arXiv](https://arxiv.org/abs/2604.24351)
* Project page: [GitHub](https://modelscope.github.io/diffusion-templates-web/)
* Documentation: [English Version](https://diffsynth-studio-doc.readthedocs.io/en/latest/Diffusion_Templates/Introducing_Diffusion_Templates.html)、[中文版](https://diffsynth-studio-doc.readthedocs.io/zh-cn/latest/Diffusion_Templates/Introducing_Diffusion_Templates.html)
* Online demo: [ModelScope](https://modelscope.cn/studios/DiffSynth-Studio/Diffusion-Templates)
* Models: [ModelScope](https://modelscope.cn/collections/DiffSynth-Studio/KleinBase4B-Templates)、[ModelScope International](https://modelscope.ai/collections/DiffSynth-Studio/KleinBase4B-Templates)、[HuggingFace](https://huggingface.co/collections/DiffSynth-Studio/kleinbase4b-templates)
* Datasets: [ModelScope](https://modelscope.cn/collections/DiffSynth-Studio/ImagePulseV2)、[ModelScope International](https://modelscope.cn/collections/DiffSynth-Studio/ImagePulseV2)、[HuggingFace](https://huggingface.co/collections/DiffSynth-Studio/imagepulsev2)

## Results

> **Prompt:** A cat is sitting on a stone.

| Input (100px) | Output | Input (512px) | Output |
|:---:|:---:|:---:|:---:|
| ![](./assets/cat_lowers_100.jpg) | ![](./assets/cat_Upscaler_1.png) | ![](./assets/cat_lowers_512.jpg) | ![](./assets/cat_Upscaler_2.png) |

---

> **Prompt:** An anime girl under a cherry blossom tree, looking at the sky.

| Input (100px) | Output | Input (512px) | Output |
|:---:|:---:|:---:|:---:|
| ![](./assets/girl_lowers_100.jpg) | ![](./assets/girl_Upscaler_1.png) | ![](./assets/girl_lowers_512.jpg) | ![](./assets/girl_Upscaler_2.png) |

---

> **Prompt:** A hamburger with fries on a plate.

| Input (100px) | Output | Input (512px) | Output |
|:---:|:---:|:---:|:---:|
| ![](./assets/food_lowers_100.jpg) | ![](./assets/food_Upscaler_1.png) | ![](./assets/food_lowers_512.jpg) | ![](./assets/food_Upscaler_2.png) |

## Inference Code

* Install [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio)

```
git clone https://github.com/modelscope/DiffSynth-Studio.git
cd DiffSynth-Studio
pip install -e .
```

* Direct inference (requires 40G GPU memory)

```python
from diffsynth.diffusion.template import TemplatePipeline
from diffsynth.pipelines.flux2_image import Flux2ImagePipeline, ModelConfig
import torch
from modelscope import dataset_snapshot_download
from PIL import Image
```

```python
pipe = Flux2ImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="black-forest-labs/FLUX.2-klein-base-4B", origin_file_pattern="transformer/*.safetensors"),
        ModelConfig(model_id="black-forest-labs/FLUX.2-klein-4B", origin_file_pattern="text_encoder/*.safetensors"),
        ModelConfig(model_id="black-forest-labs/FLUX.2-klein-4B", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    tokenizer_config=ModelConfig(model_id="black-forest-labs/FLUX.2-klein-4B", origin_file_pattern="tokenizer/"),
)
template = TemplatePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[ModelConfig(model_id="DiffSynth-Studio/Template-KleinBase4B-Upscaler")],
)
dataset_snapshot_download(
    "DiffSynth-Studio/examples_in_diffsynth",
    allow_file_pattern=["templates/*"],
    local_dir="data/examples",
)
image = template(
    pipe,
    prompt="A cat is sitting on a stone.",
    seed=0, cfg_scale=4, num_inference_steps=50,
    template_inputs=[{
        "image": Image.open("data/examples/templates/image_lowres_512.jpg"),
        "prompt": "A cat is sitting on a stone.",
    }],
    negative_template_inputs=[{
        "image": Image.open("data/examples/templates/image_lowres_512.jpg"),
        "prompt": "",
    }],
)
image.save("image_Upscaler_1.png")
image = template(
    pipe,
    prompt="A cat is sitting on a stone.",
    seed=0, cfg_scale=4, num_inference_steps=50,
    template_inputs=[{
        "image": Image.open("data/examples/templates/image_lowres_100.jpg"),
        "prompt": "A cat is sitting on a stone.",
    }],
    negative_template_inputs=[{
        "image": Image.open("data/examples/templates/image_lowres_100.jpg"),
        "prompt": "",
    }],
)
image.save("image_Upscaler_2.png")
```

* Enable lazy loading and memory management, requires 24G GPU memory

```python
from diffsynth.diffusion.template import TemplatePipeline
from diffsynth.pipelines.flux2_image import Flux2ImagePipeline, ModelConfig
import torch
from modelscope import dataset_snapshot_download
from PIL import Image

vram_config = {
    "offload_dtype": "disk",
    "offload_device": "disk",
    "onload_dtype": torch.float8_e4m3fn,
    "onload_device": "cpu",
    "preparing_dtype": torch.float8_e4m3fn,
    "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}
pipe = Flux2ImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="black-forest-labs/FLUX.2-klein-base-4B", origin_file_pattern="transformer/*.safetensors", **vram_config),
        ModelConfig(model_id="black-forest-labs/FLUX.2-klein-4B", origin_file_pattern="text_encoder/*.safetensors", **vram_config),
        ModelConfig(model_id="black-forest-labs/FLUX.2-klein-4B", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    tokenizer_config=ModelConfig(model_id="black-forest-labs/FLUX.2-klein-4B", origin_file_pattern="tokenizer/"),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 0.5,
)
template = TemplatePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[ModelConfig(model_id="DiffSynth-Studio/Template-KleinBase4B-Upscaler")],
    lazy_loading=True,
)
dataset_snapshot_download(
    "DiffSynth-Studio/examples_in_diffsynth",
    allow_file_pattern=["templates/*"],
    local_dir="data/examples",
)
image = template(
    pipe,
    prompt="A cat is sitting on a stone.",
    seed=0, cfg_scale=4, num_inference_steps=50,
    template_inputs = [{
        "image": Image.open("data/examples/templates/image_lowres_512.jpg"),
        "prompt": "A cat is sitting on a stone.",
    }],
    negative_template_inputs = [{
        "image": Image.open("data/examples/templates/image_lowres_512.jpg"),
        "prompt": "",
    }],
)
image.save("image_Upscaler_1.png")
image = template(
    pipe,
    prompt="A cat is sitting on a stone.",
    seed=0, cfg_scale=4, num_inference_steps=50,
    template_inputs = [{
        "image": Image.open("data/examples/templates/image_lowres_100.jpg"),
        "prompt": "A cat is sitting on a stone.",
    }],
    negative_template_inputs = [{
        "image": Image.open("data/examples/templates/image_lowres_100.jpg"),
        "prompt": "",
    }],
)
image.save("image_Upscaler_2.png")

```

## Training Script

After installing DiffSynth-Studio, use the following script to start training. For more information, please refer to the [DiffSynth-Studio Documentation](https://diffsynth-studio-doc.readthedocs.io/zh-cn/latest/).

```shell
modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "flux2/Template-KleinBase4B-Upscaler/*" --local_dir ./data/diffsynth_example_dataset

accelerate launch examples/flux2/model_training/train.py \
  --dataset_base_path data/diffsynth_example_dataset/flux2/Template-KleinBase4B-Upscaler \
  --dataset_metadata_path data/diffsynth_example_dataset/flux2/Template-KleinBase4B-Upscaler/metadata.jsonl \
  --extra_inputs "template_inputs" \
  --max_pixels 1048576 \
  --dataset_repeat 50 \
  --model_id_with_origin_paths "black-forest-labs/FLUX.2-klein-4B:text_encoder/*.safetensors,black-forest-labs/FLUX.2-klein-base-4B:transformer/*.safetensors,black-forest-labs/FLUX.2-klein-4B:vae/diffusion_pytorch_model.safetensors" \
  --template_model_id_or_path "DiffSynth-Studio/Template-KleinBase4B-Upscaler:" \
  --tokenizer_path "black-forest-labs/FLUX.2-klein-4B:tokenizer/" \
  --learning_rate 1e-4 \
  --num_epochs 2 \
  --remove_prefix_in_ckpt "pipe.template_model." \
  --output_path "./models/train/Template-KleinBase4B-Upscaler_full" \
  --trainable_models "template_model" \
  --use_gradient_checkpointing \
  --find_unused_parameters
```