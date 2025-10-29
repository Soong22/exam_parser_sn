# -*- coding: utf-8 -*-
import json, re
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterable, Tuple

# ===== 경로(하드코딩) =====
SRC_DIR = Path(r"exam_parser-main\01_middle_process\data\plain")     # cleaned JSON들 (para_blocks 포함)
DST_DIR = Path(r"exam_parser-main\01_middle_process/data/content_list")  # 결과 저장

DST_DIR.mkdir(parents=True, exist_ok=True)

# ===== 유틸 & 규칙 =====
SENT_END_RE = re.compile(r"(다\.)|([\.!?][”\")\]]?)\s*$")

def get_image_path_any(d: Dict[str, Any]) -> str:
    return d.get("img_path") or d.get("image_path") or d.get("path") or ""

def norm_type(t: Optional[str]) -> str:
    return (t or "").lower().strip()

def clean_spaces(s: str) -> str:
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\s+([,.\)\]”])", r"\1", s)
    s = re.sub(r"([\(\[“])\s+", r"\1", s)
    return s.strip()

def union_bbox(b1: Optional[List[float]], b2: Optional[List[float]]) -> Optional[List[float]]:
    if not b1: return b2
    if not b2: return b1
    return [min(b1[0], b2[0]), min(b1[1], b2[1]), max(b1[2], b2[2]), max(b1[3], b2[3])]

def get_bbox_any(d: Dict[str, Any]) -> Optional[List[float]]:
    b = d.get("bbox") or d.get("bb") or d.get("rect")
    if isinstance(b, list) and len(b) == 4:
        return b
    return None

def get_span_text(sp: Dict[str, Any]) -> str:
    for k in ("content", "text", "value", "str"):
        v = sp.get(k)
        if isinstance(v, str):
            return v
    # 수식 필드가 텍스트로만 오는 경우도 보정
    for k in ("latex", "math", "formula", "asciimath"):
        v = sp.get(k)
        if isinstance(v, str):
            return v
    return ""

def iter_child_blocks(block: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    for sub in (block.get("blocks") or []):
        yield sub
        yield from iter_child_blocks(sub)

def iter_spans_from_block_deep(block: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    # 현재층
    if isinstance(block.get("lines"), list) and block["lines"]:
        for line in block["lines"]:
            for sp in (line.get("spans") or []):
                yield sp
    for sp in (block.get("spans") or []):
        yield sp
    # 하위 블록들
    for sub in (block.get("blocks") or []):
        yield from iter_spans_from_block_deep(sub)

def join_text_in(block: Dict[str, Any]) -> str:
    """lines/spans 안의 텍스트들을 공백 1칸으로 합치기 (캡션 텍스트 등 추출용)"""
    parts: List[str] = []
    for sp in iter_spans_from_block_deep(block):
        if norm_type(sp.get("type")) == "inline_equation":
            parts.append(get_span_text(sp))
        else:
            t = get_span_text(sp)
            if t:
                parts.append(t)
    return clean_spaces(" ".join(parts))

# ===== 판정 =====
def is_inline_equation_span(sp: Dict[str, Any]) -> bool:
    t = norm_type(sp.get("type"))
    if "inline" in t and ("equation" in t or "math" in t or "formula" in t):
        return True
    # content에 수식 문자열만 들어간 경우도 허용
    for k in ("latex", "math", "formula", "asciimath"):
        if isinstance(sp.get(k), str):
            return True
    return False

def is_interline_equation_block(blk: Dict[str, Any]) -> bool:
    t = norm_type(blk.get("type"))
    if not t: return False
    if "interline" in t and ("equation" in t or "math" in t or "formula" in t): return True
    if "display"   in t and ("equation" in t or "math" in t or "formula" in t): return True
    if ("equation" in t or "math" in t or "formula" in t) and ("inline" not in t): return True
    return False

def is_table_block(blk: Dict[str, Any]) -> bool:
    t = norm_type(blk.get("type"))
    if isinstance(blk.get("table_body"), (str, list)) or blk.get("html"):
        return True
    if "table" in t and ("caption" not in t and "footnote" not in t):
        return True
    # 컨테이너만 있고 바디는 하위에 있는 경우도 존재
    for sp in iter_spans_from_block_deep(blk):
        if norm_type(sp.get("type")) == "table":
            return True
    return False

def is_image_block(blk: Dict[str, Any]) -> bool:
    t = norm_type(blk.get("type"))
    if "image" in t or "figure" in t or "img" in t or "picture" in t:
        return True
    if blk.get("img_path") or blk.get("image_path") or blk.get("path"):
        return True
    for sp in iter_spans_from_block_deep(blk):
        st = norm_type(sp.get("type"))
        if ("image" in st) or ("figure" in st) or ("img" in st) or ("picture" in st):
            return True
    return False

# ===== 페이로드 추출 =====
def extract_table_payload(blk: Dict[str, Any]) -> Dict[str, Any]:
    # 본문
    table_body = blk.get("table_body") or blk.get("html") or blk.get("content") or ""
    if not table_body:
        for sub in iter_child_blocks(blk):
            table_body = sub.get("table_body") or sub.get("html") or sub.get("content")
            if table_body:
                break
    if not table_body:
        for sp in iter_spans_from_block_deep(blk):
            if norm_type(sp.get("type")) == "table":
                table_body = sp.get("html") or sp.get("content") or ""
                if table_body:
                    break

    # ⬇️ 이미지 경로 주워오기 (블록 → 하위블록 → 스팬)
    img_path = get_image_path_any(blk)
    if not img_path:
        for sub in iter_child_blocks(blk):
            img_path = get_image_path_any(sub)
            if img_path:
                break
    if not img_path:
        for sp in iter_spans_from_block_deep(blk):
            st = norm_type(sp.get("type"))
            if "image" in st or "figure" in st or "img" in st or "picture" in st or "table" in st:
                img_path = get_image_path_any(sp) or img_path
                if img_path:
                    break

    # 캡션/각주
    captions: List[str] = []
    footnotes: List[str] = []
    for sub in (blk.get("blocks") or []):
        tt = norm_type(sub.get("type"))
        if "caption" in tt:
            cap = join_text_in(sub)
            if cap: captions.append(cap)
        if "footnote" in tt:
            fn = join_text_in(sub)
            if fn: footnotes.append(fn)

    return {
        "table_body": table_body or "",
        "table_caption": captions,
        "table_footnote": footnotes,
        "image_path": img_path,                # ⬅️ 추가
        "original_type": blk.get("type"),
    }


def extract_image_payload(blk: Dict[str, Any]) -> Dict[str, Any]:
    img_path = blk.get("img_path") or blk.get("image_path") or blk.get("path") or ""
    alt = blk.get("alt") or blk.get("desc") or blk.get("description") or ""
    caption = blk.get("caption") or blk.get("image_caption") or blk.get("figure_caption")

    if not img_path:
        for sp in iter_spans_from_block_deep(blk):
            st = norm_type(sp.get("type"))
            if ("image" in st) or ("figure" in st) or ("img" in st) or ("picture" in st):
                img_path = sp.get("img_path") or sp.get("image_path") or sp.get("path") or img_path
                alt = sp.get("alt") or sp.get("desc") or sp.get("description") or alt
                if caption is None:
                    caption = sp.get("caption")
                if img_path:
                    break

    # 캡션이 컨테이너 내부의 caption 블록으로 따로 있는 경우 합치기
    if caption is None:
        caps: List[str] = []
        for sub in (blk.get("blocks") or []):
            if "caption" in norm_type(sub.get("type")):
                c = join_text_in(sub)
                if c: caps.append(c)
        caption = caps if caps else None

    return {
        "image_path": img_path or "",
        "alt": alt or "",
        "caption": caption if isinstance(caption, (list, str)) else None,
        "original_type": blk.get("type"),
    }

def extract_equation_text(blk: Dict[str, Any]) -> str:
    for k in ("content", "text", "latex", "math", "formula", "asciimath", "value"):
        v = blk.get(k)
        if isinstance(v, str) and v.strip():
            return v
    parts: List[str] = []
    for sp in iter_spans_from_block_deep(blk):
        s = get_span_text(sp)
        if s: parts.append(s)
    return clean_spaces(" ".join(parts)) if parts else ""

# ===== 문장 flush =====
def flush_sentence(acc: List[Dict[str, Any]], buf_text: str, buf_bbox: Optional[List[float]], page_idx: int):
    t = clean_spaces(buf_text)
    if t:
        if not (acc and acc[-1]["type"] == "text" and acc[-1]["text"] == t and acc[-1]["page_idx"] == page_idx):
            acc.append({
                "type": "text",
                "text": t,
                "bbox": buf_bbox if buf_bbox else [0,0,0,0],
                "page_idx": page_idx
            })
    return "", None

# ===== 블록 → 아이템 =====
def fold_block_to_items(block: Dict[str, Any], page_idx: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    # 표/이미지/행간수식 컨테이너/블록 우선 처리
    if is_table_block(block):
        payload = extract_table_payload(block)
        out.append({
            "type": "table",
            "bbox": get_bbox_any(block) or [0,0,0,0],
            "page_idx": page_idx,
            **payload
        })
        return out

    if is_image_block(block):
        payload = extract_image_payload(block)
        out.append({
            "type": "image",
            "bbox": get_bbox_any(block) or [0,0,0,0],
            "page_idx": page_idx,
            **payload
        })
        return out

    if is_interline_equation_block(block):
        eq_text = extract_equation_text(block)
        out.append({
            "type": "interline_equation",
            "content": eq_text,
            "bbox": get_bbox_any(block) or [0,0,0,0],
            "page_idx": page_idx,
            "original_type": block.get("type")
        })
        return out

    # 일반 텍스트 + 인라인 수식 (재귀 spans)
    buf_text = ""
    buf_bbox: Optional[List[float]] = None

    for sp in iter_spans_from_block_deep(block):
        sptype = norm_type(sp.get("type"))
        txt = get_span_text(sp)
        bbox = get_bbox_any(sp)

        # 스팬이 이미지/표로 오는 특수 케이스도 안전 처리
        if ("image" in sptype) or ("figure" in sptype) or ("img" in sptype) or ("picture" in sptype):
            if buf_text:
                buf_text, buf_bbox = flush_sentence(out, buf_text, buf_bbox, page_idx)
            out.append({
                "type": "image",
                "bbox": bbox if bbox else [0,0,0,0],
                "page_idx": page_idx,
                "image_path": sp.get("img_path") or sp.get("image_path") or sp.get("path") or "",
                "alt": sp.get("alt") or sp.get("desc") or sp.get("description") or "",
                "caption": sp.get("caption") or None,
                "original_type": sp.get("type")
            })
            continue

        if "table" in sptype:
            if buf_text:
                buf_text, buf_bbox = flush_sentence(out, buf_text, buf_bbox, page_idx)
            out.append({
                "type": "table",
                "bbox": bbox if bbox else [0,0,0,0],
                "page_idx": page_idx,
                "table_body": sp.get("table_body") or sp.get("html") or sp.get("content") or "",
                "table_caption": [],
                "table_footnote": [],
                "image_path": get_image_path_any(sp),
                "original_type": sp.get("type")
            })
            continue

        if is_inline_equation_span(sp):
            if buf_text:
                buf_text, buf_bbox = flush_sentence(out, buf_text, buf_bbox, page_idx)
            if txt:
                out.append({
                    "type": "inline_equation",
                    "content": txt,
                    "bbox": bbox if bbox else [0,0,0,0],
                    "page_idx": page_idx
                })
            continue

        # 일반 텍스트
        if txt:
            buf_text = clean_spaces((buf_text + " " + txt) if buf_text else txt)
            # 텍스트 bbox만 누적
            buf_bbox = union_bbox(buf_bbox, bbox)

        # 문장 경계면 flush
        if buf_text and SENT_END_RE.search(buf_text):
            buf_text, buf_bbox = flush_sentence(out, buf_text, buf_bbox, page_idx)

    if buf_text:
        buf_text, buf_bbox = flush_sentence(out, buf_text, buf_bbox, page_idx)

    return out

def page_to_items(page: Dict[str, Any]) -> List[Dict[str, Any]]:
    seq: List[Dict[str, Any]] = []
    page_idx = page.get("page_idx", 0)
    # cleaned 기준: para_blocks 안에 모든 상위 블록이 들어있음
    for blk in (page.get("para_blocks") or []):
        seq.extend(fold_block_to_items(blk, page_idx))
    return seq

def convert_file_to_flat_content_list(src: Path, dst: Path):
    data = json.loads(src.read_text(encoding="utf-8"))
    flat: List[Dict[str, Any]] = []
    for page in (data.get("pdf_info") or []):
        flat.extend(page_to_items(page))
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(flat, ensure_ascii=False, indent=2), encoding="utf-8")

def main():
    files = list(SRC_DIR.rglob("*.json"))
    print(f"🔍 {len(files)}개 파일 처리")
    for src in files:
        rel = src.relative_to(SRC_DIR)
        out_name = rel.stem.replace("_plain", "") + "_content_list.json"
        dst = DST_DIR / rel.with_name(out_name)
        convert_file_to_flat_content_list(src, dst)
        print(f"✅ {rel} → {out_name}")

if __name__ == "__main__":
    main()
