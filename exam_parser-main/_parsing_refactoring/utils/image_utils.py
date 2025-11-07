from __future__ import annotations

import os, shutil
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

try:
    from PIL import Image
except Exception:
    Image = None


@dataclass
class ImageConfig:
    """이미지 입출력 경로 및 동작 옵션"""
    src_root: str            # 원본 이미지들이 흩어져 있는 루트 (ex. "images")
    dst_img_root: str        # 문제 본문 이미지 저장 루트 (ex. "images/image")
    dst_tbl_root: str        # 표 이미지 저장 루트 (ex. "images/table")
    dst_form_root: str       # 수식 이미지 저장 루트 (ex. "images/formula")
    move: bool = False       # 복사 대신 이동할지
    overwrite: bool = True   # 같은 파일명 덮어쓰기 여부


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _get_image_size(img_path: str) -> Dict[str, Optional[int]]:
    ch = h = w = None
    if Image is not None and img_path and os.path.exists(img_path):
        try:
            with Image.open(img_path) as im:
                w, h = im.size
                mode = im.mode.upper()
                ch = {"1": 1, "L": 1, "P": 1, "LA": 2, "RGB": 3, "RGBA": 4, "CMYK": 4, "YCbCr": 3}.get(mode, None)
        except Exception:
            pass
    return {"channel": ch, "height": h, "width": w}


def _resolve_image_src_path(src_hint: str, cfg: ImageConfig) -> str:
    """src_hint 절대/상대경로 또는 파일명만 들어와도 src_root 아래에서 찾아냄."""
    if not src_hint:
        return ""
    p = src_hint
    if os.path.exists(p):
        return p
    base = os.path.basename(p)
    cand = os.path.join(cfg.src_root, base)
    if os.path.exists(cand):
        return cand
    # src_root 하위 전체에서 탐색 (가볍지 않으니 필요 시 캐시 가능)
    for root, _, files in os.walk(cfg.src_root):
        if base in files:
            return os.path.join(root, base)
    return ""


def _unique_dest_with_base(dest_dir: str, base_no_ext: str, ext: str, overwrite: bool) -> str:
    ensure_dir(dest_dir)
    out = os.path.join(dest_dir, base_no_ext + ext)
    if overwrite:
        return out
    k = 1
    while os.path.exists(out):
        out = os.path.join(dest_dir, f"{base_no_ext}_{k}{ext}")
        k += 1
    return out


def _copy_or_move(src: str, dst: str, move: bool) -> None:
    ensure_dir(os.path.dirname(dst))
    try:
        if os.path.samefile(src, dst):
            return
    except Exception:
        pass
    if move:
        shutil.move(src, dst)
    else:
        shutil.copy2(src, dst)


def classify_and_store_image(
    src_hint: str,
    stem: str,
    tag: str,
    kind: str,         # "img" | "tbl" | "form"
    cfg: ImageConfig,
) -> Tuple[str, Dict[str, Optional[int]], str]:
    """
    반환값: (저장된상대경로, 이미지사이즈dict, 원본절대경로)
    - 파일이 없으면 ("", {}, "")
    - kind:
        * "img"  -> cfg.dst_img_root / <stem> / <tag>.<ext>
        * "tbl"  -> cfg.dst_tbl_root / <stem> / <tag>.<ext>
        * "form" -> cfg.dst_form_root / <stem> / <tag>.<ext>
    """
    src_abs = _resolve_image_src_path(src_hint, cfg)
    if not src_abs:
        return "", {}, ""

    ext = os.path.splitext(src_abs)[1].lower()
    if ext not in (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tif", ".tiff", ".webp"):
        ext = ".jpg"

    if kind == "img":
        dst_root = cfg.dst_img_root
    elif kind == "tbl":
        dst_root = cfg.dst_tbl_root
    elif kind == "form":
        dst_root = cfg.dst_form_root
    else:
        # 알 수 없는 kind면 기본 이미지 루트로
        dst_root = cfg.dst_img_root

    dest_dir = os.path.join(dst_root, stem)
    dest_path = _unique_dest_with_base(dest_dir, tag, ext, overwrite=cfg.overwrite)

    _copy_or_move(src_abs, dest_path, move=cfg.move)

    rel_path = os.path.normpath(dest_path).replace("\\", "/")
    size = _get_image_size(dest_path)
    return rel_path, size, os.path.normpath(src_abs).replace("\\", "/")