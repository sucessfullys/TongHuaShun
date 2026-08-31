# Image Quality Metrics

Evaluate enhanced/restored images with PSNR, SSIM, LPIPS, MUSIQ, and MANIQA.

## Install

In the environment you use for evaluation:

```bash
pip install pyiqa
```

`pyiqa` will download LPIPS/MUSIQ/MANIQA weights automatically the first time the metrics are created. To download/cache weights before a long run:

```bash
cd /mnt/image-edit/datasets/duanyufa/DiffSynth-Studio
python metric/eval_image_quality_metrics.py --download_only --device cuda:0
```

If the machine cannot access the internet, run the command once on a machine with internet and copy the model cache into the same user cache directory on this machine. `pyiqa` commonly stores weights under the user cache, such as `~/.cache/torch` and package-specific cache folders.

## Run

```bash
cd /mnt/image-edit/datasets/duanyufa/DiffSynth-Studio

python metric/eval_image_quality_metrics.py \
  --pred_dir /path/to/enhanced_images \
  --gt_dir /path/to/gt_images \
  --output metric/results.csv \
  --txt_output metric/results.txt \
  --device cuda:0
```

By default, images are matched by the same filename. Use `--recursive` to match by relative path under each folder.

The TXT file contains the five mean metrics:

- PSNR
- SSIM
- LPIPS
- MUSIQ
- MANIQA

PSNR, SSIM, and LPIPS require `--gt_dir`. MUSIQ and MANIQA are no-reference metrics and only use `--pred_dir`. For LPIPS, lower is better. For the other four metrics, higher is generally better.
