---
license: cc-by-4.0
tags:
- image-to-image
- computer-vision
- photo-restoration
- synthetic-data
- image-processing
pretty_name: OpenPhoto Restore Dataset
size_categories:
- 1K<n<10K
task_categories:
- image-to-image
task_ids:
- image-inpainting
- image-colorization
dataset_info:
  features:
  - name: pristine_image
    dtype: image
  - name: damaged_image
    dtype: image
  splits:
  - name: train
    num_bytes: 4008073440.6
    num_examples: 4500
  - name: test
    num_bytes: 435635569.4
    num_examples: 500
  download_size: 4419767995
  dataset_size: 4443709010.0
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: test
    path: data/test-*
---

# Synthetic Photo Restoration Dataset

## Dataset Summary

This dataset was created to address the scarcity of large-scale, permissively licensed, and fully open-source datasets for photo restoration.
To our knowledge, it is one of the first photo restoration dataset to be generated using a fully reproducible, open-source pipeline  that combines both texture-based and procedural damage simulation.

The dataset consists of pairs of pristine, high-quality modern photographs and a corresponding "damaged" version that has been processed through a sophisticated, multi-stage synthetic damage pipeline.
It is designed for training image-to-image models for historical photo restoration and colorization.

## Dataset Structure

The dataset is provided as a `DatasetDict` containing a `train` and `test` split.

### Data Splits

| Split | Size          |
|-------|---------------|
| `train` | 4500  |
| `test`  | 500   |

### Data Fields

*   `pristine_image`: A Image object containing the original, high-quality photograph, rescaled to 1024x1024.
*   `damaged_image`: A Image object containing the synthetically damaged version.

## Data Generation Pipeline

### Source Data

The pristine images were sourced from **Unsplash**, chosen for its high photographic quality and permissive license.
The image URL and tags were retrieved from the **Unsplash Lite Dataset**.
The images were filtered by tags to primarily include portraits and photos of people to align with the common use case of restoring family photos.

### Synthetic Damage Pipeline

A multi-stage, probablistic pipeline was developed to simulate realistic photographic decay.
The order of operations was designed to mimic the physical and chemical processes that affect a real photograph over time.

**Stage 1: Physical Damage**
This stage simulates physical interactions with the photograph, such as tears, cracks, and surface scratches.

*   **Cracks:** High resolution, ground truth crack segmentation data from **SUT-Crack Dataset** is used as crack texture masks. These masks were then randomly augmented (flipped, rotated), feathered, and composited onto the pristine image.
*   **Scratches & Dust:** A high-quality film-damage video overlay from **Enchanted Media** was used as a source for unique scratch and dust patterns, which were then blended onto the image.

**Stage 2: Chemical & Age Damage**
This stage simulates the print aging over time.

*   **Color Shift:** The image is converted grayscale, and a sepia color filter is probablistically applied.
*   **Fading:** Contrast and brightness were randomly reduced using an algorithmic approach to simulate chemical fading.

**Stage 3: Final Texture / Film Stock Simulation**
This stage simulates the film stock that the photo was printed on.

*   **Film Grain:** Gaussian noise is applied to simulate the underlying texture of film stock, avoiding the need for external texture files.

## Usage

The dataset can be loaded using the `datasets` library:

```python
from datasets import load_dataset

data = load_dataset("joshuachin/openphoto-restore-dataset")
train_dataset = data['train']
test_dataset = data['test']
```

## Licensing Information

The dataset is released under the **Creative Commons Attribution 4.0 International License (CC BY 4.0)**.
You are free to share and adapt this dataset for any purpose, including commercial, as long as you provide appropriate credit to both this dataset and the original sources listed in the acknowledgments.

The Unsplash images are licensed under the [Unsplash License](https://unsplash.com/license).

The SUT-Crack dataset is licensed under [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/).

The Enchanted Media video is provided under the following terms of use:
"You are free to use these video assets within any project, including commercial broadcast and monetized YouTube videos, but distributing or reselling these without permission from Enchanted Media is strictly forbidden."

## Citation

If you use this dataset in your research or project, please cite it as follows:

```bibtex
@misc{chin2025openphoto,
  author    = {Joshua Chin},
  title     = {OpenPhoto Restore Dataset},
  month     = {September}
  year      = {2025},
   howpublished = {\url{https://huggingface.co/datasets/joshuachin/openphoto-restore-dataset}}
}
```

## Acknowledgments

This project would not have been possible without the generous open-source contributions of the following individuals and organizations.
We encourage users of this dataset to also acknowledge their foundational work.

*   **[Unsplash](https://unsplash.com/)** for providing the high-quality source images, retrieved using the **[Unsplash Lite Dataset](https://unsplash.com/data)**.
*   **[SUT-Crack](https://data.mendeley.com/datasets/gsbmknrhkv/6)** for providing crack texture masks.
*   **[Enchanted Media](https://www.enchanted.media/downloads/free-dust-and-scratches-overlay-video/)** for providing film damage textures.
*   **[PikFix](https://github.com/DerrickXuNu/Pik-Fix)** for inspiring the data generation methodology.

## Source Citations

For academic use, please cite the original sources of the data components used to create this dataset.

**For the crack textures (SUT-Crack):**
```bibtex
@misc{sut_crack_2023,
  author    = {Sabouri, Mohammadreza and Sepidbar, Alireza},
  title     = {SUT-Crack},
  year      = {2023},
  publisher = {Mendeley Data},
  version   = {V6},
  doi       = {10.17632/gsbmknrhkv.6},
  url       = {https://data.mendeley.com/datasets/gsbmknrhkv/6}
}
```