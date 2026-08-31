from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import fitz
from PIL import Image, ImageStat


def is_content_patch(img: Image.Image, white_threshold: int, min_nonwhite_ratio: float) -> bool:
    gray = img.convert('L')
    hist = gray.histogram()
    total = gray.width * gray.height
    white_like = sum(hist[white_threshold:])
    nonwhite_ratio = 1.0 - white_like / max(total, 1)
    if nonwhite_ratio < min_nonwhite_ratio:
        return False
    stat = ImageStat.Stat(gray)
    # Avoid nearly flat blank pages with compression artifacts.
    return (stat.stddev[0] if stat.stddev else 0) > 8


def render_page(page: fitz.Page, dpi: int) -> Image.Image:
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    return Image.frombytes('RGB', (pix.width, pix.height), pix.samples)


def crop_candidates(width: int, height: int, patch: int, rng: random.Random, attempts: int):
    if width < patch or height < patch:
        return []
    coords = []
    # A few deterministic grid anchors plus random anchors.
    xs = sorted(set([0, max(0, (width - patch) // 2), max(0, width - patch)]))
    ys = sorted(set([0, max(0, (height - patch) // 2), max(0, height - patch)]))
    for y in ys:
        for x in xs:
            coords.append((x, y))
    for _ in range(attempts):
        coords.append((rng.randint(0, width - patch), rng.randint(0, height - patch)))
    return coords


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdf-dir', default='/mnt/image-edit/datasets/duanyufa/SR_Dataset/文档图片/PDF')
    parser.add_argument('--out-dir', default='/mnt/image-edit/datasets/duanyufa/SR_Dataset/文档图片/images')
    parser.add_argument('--target-count', type=int, default=20000)
    parser.add_argument('--patch-size', type=int, default=1024)
    parser.add_argument('--dpi', type=int, default=300)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--attempts-per-page', type=int, default=24)
    parser.add_argument('--max-per-page', type=int, default=8)
    parser.add_argument('--white-threshold', type=int, default=245)
    parser.add_argument('--min-nonwhite-ratio', type=float, default=0.03)
    args = parser.parse_args()

    pdf_dir = Path(args.pdf_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / 'pdf_patch_manifest.jsonl'
    rng = random.Random(args.seed)

    pdfs = sorted(pdf_dir.glob('*.pdf'))
    if not pdfs:
        raise SystemExit(f'No PDF files found in {pdf_dir}')

    count = len([p for p in out_dir.glob('*.png') if p.is_file()])
    print(f'existing_png={count}')

    with manifest_path.open('a') as mf:
        for pdf_idx, pdf_path in enumerate(pdfs):
            if count >= args.target_count:
                break
            print(f'PDF {pdf_idx+1}/{len(pdfs)} {pdf_path.name}', flush=True)
            try:
                doc = fitz.open(pdf_path)
            except Exception as e:
                print(f'  open failed: {e}', flush=True)
                continue
            for page_idx in range(len(doc)):
                if count >= args.target_count:
                    break
                try:
                    page_img = render_page(doc[page_idx], args.dpi)
                except Exception as e:
                    print(f'  render failed page={page_idx}: {e}', flush=True)
                    continue

                if page_img.width < args.patch_size or page_img.height < args.patch_size:
                    scale = max(args.patch_size / page_img.width, args.patch_size / page_img.height)
                    new_size = (int(page_img.width * scale + 0.5), int(page_img.height * scale + 0.5))
                    page_img = page_img.resize(new_size, Image.Resampling.LANCZOS)

                coords = crop_candidates(page_img.width, page_img.height, args.patch_size, rng, args.attempts_per_page)
                rng.shuffle(coords)
                page_written = 0
                for x, y in coords:
                    if count >= args.target_count or page_written >= args.max_per_page:
                        break
                    patch = page_img.crop((x, y, x + args.patch_size, y + args.patch_size))
                    if not is_content_patch(patch, args.white_threshold, args.min_nonwhite_ratio):
                        continue
                    name = f'docpatch_{count:06d}_pdf{pdf_idx:02d}_p{page_idx:03d}_x{x}_y{y}.png'
                    dst = out_dir / name
                    patch.save(dst, format='PNG', optimize=False)
                    row = {
                        'image': str(dst),
                        'pdf': str(pdf_path),
                        'page': page_idx,
                        'x': x,
                        'y': y,
                        'patch_size': args.patch_size,
                        'dpi': args.dpi,
                        'source_page_width': page_img.width,
                        'source_page_height': page_img.height,
                    }
                    mf.write(json.dumps(row, ensure_ascii=False) + '\n')
                    count += 1
                    page_written += 1
                print(f'  page {page_idx+1}/{len(doc)} wrote={page_written} total={count}', flush=True)
            doc.close()

    print('DONE')
    print(f'total_png={count}')
    print(f'out_dir={out_dir}')
    print(f'manifest={manifest_path}')


if __name__ == '__main__':
    main()
