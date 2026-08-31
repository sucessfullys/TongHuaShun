from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import fitz
from PIL import Image, ImageStat


@dataclass(frozen=True)
class Shape:
    w: int
    h: int
    weight: float

    @property
    def ratio(self) -> float:
        return max(self.w / self.h, self.h / self.w)

    @property
    def name(self) -> str:
        return f'{self.w}x{self.h}'


@dataclass(frozen=True)
class Config:
    name: str
    dpi: int
    target: int
    shapes: tuple[Shape, ...]


CONFIGS = [
    Config(
        name='images_1024mix_dpi300_10k',
        dpi=300,
        target=10000,
        shapes=(
            Shape(1024, 1024, 0.70),
            Shape(1280, 960, 0.08),
            Shape(960, 1280, 0.08),
            Shape(1536, 1024, 0.07),
            Shape(1024, 1536, 0.07),
        ),
    ),
    Config(
        name='images_2048mix_dpi300_10k',
        dpi=300,
        target=10000,
        shapes=(
            Shape(2048, 2048, 0.70),
            Shape(2048, 1536, 0.08),
            Shape(1536, 2048, 0.08),
            Shape(1792, 1280, 0.07),
            Shape(1280, 1792, 0.07),
        ),
    ),
    Config(
        name='images_2048mix_dpi400_10k',
        dpi=400,
        target=10000,
        shapes=(
            Shape(2048, 2048, 0.70),
            Shape(2048, 1536, 0.08),
            Shape(1536, 2048, 0.08),
            Shape(1792, 1280, 0.07),
            Shape(1280, 1792, 0.07),
        ),
    ),
]


def render_page(page: fitz.Page, dpi: int) -> Image.Image:
    zoom = dpi / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    return Image.frombytes('RGB', (pix.width, pix.height), pix.samples)


def is_content_patch(img: Image.Image, white_threshold: int, min_nonwhite_ratio: float) -> bool:
    gray = img.convert('L')
    hist = gray.histogram()
    total = gray.width * gray.height
    white_like = sum(hist[white_threshold:])
    nonwhite_ratio = 1.0 - white_like / max(total, 1)
    if nonwhite_ratio < min_nonwhite_ratio:
        return False
    stat = ImageStat.Stat(gray)
    return (stat.stddev[0] if stat.stddev else 0) > 8


def box_iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / max(area_a + area_b - inter, 1)


def candidate_boxes(width: int, height: int, shape: Shape, rng: random.Random) -> list[tuple[int, int, int, int]]:
    pw, ph = shape.w, shape.h
    if width < pw or height < ph:
        return []
    sx = max(1, int(pw * 0.50))
    sy = max(1, int(ph * 0.50))
    xs = list(range(0, max(width - pw, 0) + 1, sx))
    ys = list(range(0, max(height - ph, 0) + 1, sy))
    if not xs or xs[-1] != width - pw:
        xs.append(width - pw)
    if not ys or ys[-1] != height - ph:
        ys.append(height - ph)
    boxes = [(x, y, x + pw, y + ph) for y in ys for x in xs]
    rng.shuffle(boxes)
    return boxes


def shape_quotas(cfg: Config) -> dict[str, int]:
    quotas = {s.name: int(cfg.target * s.weight) for s in cfg.shapes}
    rest = cfg.target - sum(quotas.values())
    quotas[cfg.shapes[0].name] += rest
    return quotas


def choose_shape(cfg: Config, quotas: dict[str, int], rng: random.Random) -> Shape:
    available = [s for s in cfg.shapes if quotas.get(s.name, 0) > 0]
    if not available:
        return cfg.shapes[0]
    weights = [quotas[s.name] for s in available]
    return rng.choices(available, weights=weights, k=1)[0]


def iter_pages(pdfs: list[Path]) -> Iterable[tuple[int, Path, int]]:
    for pdf_idx, pdf in enumerate(pdfs):
        try:
            doc = fitz.open(pdf)
            n = len(doc)
            doc.close()
        except Exception:
            continue
        for page_idx in range(n):
            yield pdf_idx, pdf, page_idx


def generate_config(base: Path, pdfs: list[Path], cfg: Config, seed: int, overwrite: bool):
    out_dir = base / cfg.name
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / 'pdf_patch_manifest.jsonl'
    summary_path = out_dir / 'summary.json'

    if overwrite:
        for p in out_dir.glob('*.png'):
            p.unlink()
        if manifest_path.exists():
            manifest_path.unlink()

    existing = sorted(out_dir.glob('*.png'))
    count = len(existing)
    rng = random.Random(seed)
    quotas = shape_quotas(cfg)
    shape_counts = {s.name: 0 for s in cfg.shapes}
    pages_used = set()
    pdfs_used = set()
    page_boxes: dict[str, list[tuple[int, int, int, int]]] = {}
    page_order = list(iter_pages(pdfs))
    rng.shuffle(page_order)

    print(f'[{cfg.name}] existing={count} pages={len(page_order)} target={cfg.target}', flush=True)
    pass_id = 0
    with manifest_path.open('a') as mf:
        while count < cfg.target:
            wrote_this_pass = 0
            pass_id += 1
            for pdf_idx, pdf_path, page_idx in page_order:
                if count >= cfg.target:
                    break
                page_key = f'{pdf_idx}:{page_idx}'
                try:
                    doc = fitz.open(pdf_path)
                    page_img = render_page(doc[page_idx], cfg.dpi)
                    doc.close()
                except Exception as e:
                    print(f'[{cfg.name}] render_failed pdf={pdf_path.name} page={page_idx} err={e}', flush=True)
                    continue

                tried_shapes = []
                wrote_page = False
                for _ in range(len(cfg.shapes)):
                    shape = choose_shape(cfg, quotas, rng)
                    if shape.name in tried_shapes:
                        break
                    tried_shapes.append(shape.name)
                    boxes = candidate_boxes(page_img.width, page_img.height, shape, rng)
                    if not boxes:
                        quotas[shape.name] = max(0, quotas.get(shape.name, 0) - 1)
                        continue
                    old_boxes = page_boxes.setdefault(page_key, [])
                    for box in boxes:
                        if any(box_iou(box, old) > 0.60 for old in old_boxes):
                            continue
                        x1, y1, x2, y2 = box
                        patch = page_img.crop(box)
                        if not is_content_patch(patch, 245, 0.03):
                            continue
                        name = f'{cfg.name}_{count:06d}_pdf{pdf_idx:03d}_p{page_idx:03d}_{shape.name}_x{x1}_y{y1}.png'
                        dst = out_dir / name
                        patch.save(dst, format='PNG', optimize=False)
                        row = {
                            'image': str(dst),
                            'pdf': str(pdf_path),
                            'pdf_index': pdf_idx,
                            'page': page_idx,
                            'dpi': cfg.dpi,
                            'x': x1,
                            'y': y1,
                            'width': shape.w,
                            'height': shape.h,
                            'aspect_ratio': round(shape.ratio, 4),
                            'pass': pass_id,
                            'source_page_width': page_img.width,
                            'source_page_height': page_img.height,
                        }
                        mf.write(json.dumps(row, ensure_ascii=False) + '\n')
                        count += 1
                        wrote_this_pass += 1
                        quotas[shape.name] = max(0, quotas.get(shape.name, 0) - 1)
                        shape_counts[shape.name] += 1
                        old_boxes.append(box)
                        pages_used.add(page_key)
                        pdfs_used.add(pdf_idx)
                        wrote_page = True
                        break
                    if wrote_page:
                        break

                if count % 500 == 0 and wrote_page:
                    print(f'[{cfg.name}] count={count} pass={pass_id} pages_used={len(pages_used)} pdfs_used={len(pdfs_used)}', flush=True)

            print(f'[{cfg.name}] pass={pass_id} wrote={wrote_this_pass} total={count}', flush=True)
            if wrote_this_pass == 0:
                raise RuntimeError(f'No more valid patches for {cfg.name}; total={count}')

    summary = {
        'name': cfg.name,
        'target': cfg.target,
        'actual': count,
        'dpi': cfg.dpi,
        'shapes': [{'width': s.w, 'height': s.h, 'weight': s.weight, 'ratio': s.ratio} for s in cfg.shapes],
        'shape_counts': shape_counts,
        'pdf_count_total': len(pdfs),
        'pdf_count_used': len(pdfs_used),
        'page_count_total': len(page_order),
        'page_count_used': len(pages_used),
        'max_iou_per_page_threshold': 0.60,
        'out_dir': str(out_dir),
        'manifest': str(manifest_path),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f'[{cfg.name}] DONE {json.dumps(summary, ensure_ascii=False)}', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-dir', default='/mnt/image-edit/datasets/duanyufa/SR_Dataset/文档图片')
    parser.add_argument('--pdf-dir', default='/mnt/image-edit/datasets/duanyufa/SR_Dataset/文档图片/PDF')
    parser.add_argument('--seed', type=int, default=20260809)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--only', choices=[c.name for c in CONFIGS], default=None)
    args = parser.parse_args()

    base = Path(args.base_dir)
    pdfs = sorted(Path(args.pdf_dir).glob('*.pdf'))
    if not pdfs:
        raise SystemExit(f'No PDFs found in {args.pdf_dir}')
    print(f'pdf_count={len(pdfs)}', flush=True)

    configs = [c for c in CONFIGS if args.only is None or c.name == args.only]
    for i, cfg in enumerate(configs):
        generate_config(base, pdfs, cfg, args.seed + i * 1000, args.overwrite)


if __name__ == '__main__':
    main()
