# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Any, Dict, List
from utils.text_utils import clean_text

def _bbox_overlaps_horiz(b1: List[int], b2: List[int]) -> bool:
    x11, _, x12, _ = b1
    x21, _, x22, _ = b2
    return max(0, min(x12, x22) - max(x11, x21)) > 0

def _guess_image_title(blocks: List[Dict[str, Any]], idx: int) -> str:
    b = blocks[idx]
    page = b.get("page_idx")
    bbox = b.get("bbox") or [0, 0, 0, 0]
    x1, y1, x2, y2 = bbox if len(bbox) == 4 else (0, 0, 0, 0)
    caps = (b.get("image_caption") or []) + (b.get("image_footnote") or [])
    cap = b.get("caption")
    if cap:
        if isinstance(cap, str):
            caps.append(cap)
        else:
            try:
                caps.extend(list(cap))
            except Exception:
                pass
    for c in caps:
        c = clean_text(c)
        if c: return c
    TH = 60
    for j in range(idx - 1, -1, -1):
        pb = blocks[j]
        if pb.get("type") != "text": continue
        if pb.get("page_idx") != page: break
        bb = pb.get("bbox") or [0, 0, 0, 0]
        _, by1, _, by2 = bb if len(bb) == 4 else (0, 0, 0, 0)
        if by2 <= y1 and (y1 - by2) <= TH and _bbox_overlaps_horiz(bbox, bb):
            txt = clean_text(pb.get("text", ""))
            import re
            if re.search(r"(그림|Fig\.?|도\s*\d+)", txt, re.I):
                return txt
    return ""
