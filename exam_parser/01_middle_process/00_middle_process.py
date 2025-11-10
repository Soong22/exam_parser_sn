# -*- coding: utf-8 -*-
import json, re
from pathlib import Path
from typing import Any, Dict, List, Optional, Iterable, Tuple

# ============================================================
# 하드코딩 경로 
# ============================================================
INPUT_MIDDLE_DIR     = Path(r"exam_parser-main\01_middle_process\data\middle\2023\0111-2023-국어영역-국어영역-문제_middle.json")
OUTPUT_CLEANED_DIR   = Path(r"exam_parser-main\01_middle_process\data\cleand")
OUTPUT_CLEANED_DIR.mkdir(parents=True, exist_ok=True)

# 폴더 검색 시에만 적용할 기본 필터(파일명에 '문제' 포함)
ONLY_PROBLEM_FILES = True   # 단일 파일 입력일 때는 이 옵션과 무관하게 그 파일을 처리

# ============================================================
# 설정
# ============================================================
# ❗ 복구 없이 통째로 제거할 키
REMOVE_KEYS = {"discarded_blocks", "preproc_blocks"}

# 문장 경계 휴리스틱
SENT_END_RE = re.compile(r"(다\.)|([\.!?][”\")\]]?)\s*$")

# ============================================================
# 공용 유틸
# ============================================================
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

def ensure_list_field(d: Dict[str, Any], key: str) -> List[Any]:
    v = d.get(key)
    if not isinstance(v, list):
        v = []
        d[key] = v
    return v

# ============================================================
# 페이지/블록 접근 (스키마 내성)
# ============================================================
def has_table_span_deep(blk: Dict[str, Any]) -> bool:
    for sp in iter_spans_from_block_deep(blk):
        if norm_type(sp.get("type")) == "table" and (sp.get("html") or sp.get("content")):
            return True
    return False

def has_image_span_deep(blk: Dict[str, Any]) -> bool:
    for sp in iter_spans_from_block_deep(blk):
        sptype = norm_type(sp.get("type"))
        if ("image" in sptype) or ("figure" in sptype) or ("img" in sptype) or ("picture" in sptype):
            if sp.get("img_path") or sp.get("image_path") or sp.get("path"):
                return True
    return False

def get_pages(data: Any) -> List[Dict[str, Any]]:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if isinstance(data.get("pdf_info"), list):
            return data["pdf_info"]
        if isinstance(data.get("pages"), list):
            return data["pages"]
    return [data]

def get_blocks(page: Dict[str, Any]) -> List[Dict[str, Any]]:
    """preproc_blocks 는 병합하지 않음(중복 방지)"""
    merged = []
    order_keys = ("para_blocks", "blocks", "items")  # ❗ preproc_blocks 제외
    for list_order, key in enumerate(order_keys):
        v = page.get(key)
        if isinstance(v, list) and v:
            for i, blk in enumerate(v):
                idx = blk.get("index")
                bb = blk.get("bbox") or blk.get("bb") or blk.get("rect")
                y_hint = bb[1] if isinstance(bb, list) and len(bb) == 4 else 10**9
                merged.append((
                    idx if isinstance(idx, (int, float)) else 10**9,
                    list_order, i, y_hint, blk
                ))
    # index → 리스트우선순위 → 로컬순번 → y좌표 순으로 안정 정렬
    merged.sort(key=lambda t: (t[0], t[1], t[2], t[3]))
    return [t[4] for t in merged]

def iter_spans_from_block(block: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    if isinstance(block.get("lines"), list) and block["lines"]:
        for line in block["lines"]:
            for sp in (line.get("spans") or []):
                yield sp
    for sp in (block.get("spans") or []):
        yield sp

def looks_like_page(d: Dict[str, Any]) -> bool:
    if not isinstance(d, dict):
        return False
    if "page_idx" in d or "pageIndex" in d:
        return True
    if ("width" in d and "height" in d) and any(k in d for k in ("para_blocks", "blocks", "items")):
        return True
    return False

# ============================================================
# 제거 전처리: 복구 없이 지정 키만 제거
# ============================================================
def process_node_salvage(obj: Any) -> Any:
    """'discarded_blocks'/'preproc_blocks'는 내용을 옮기지 않고 통째로 제거"""
    if isinstance(obj, dict):
        # 자식 먼저 처리
        for k in list(obj.keys()):
            obj[k] = process_node_salvage(obj[k])
        # 해당 키 삭제
        for rk in list(obj.keys()):
            if rk in REMOVE_KEYS:
                obj.pop(rk, None)
        return obj
    elif isinstance(obj, list):
        return [process_node_salvage(it) for it in obj]
    return obj

# ============================================================
# 텍스트/수식/이미지/표 판정 & 추출
# ============================================================
def get_span_text(sp: Dict[str, Any]) -> str:
    for k in ("content", "text", "value", "str"):
        v = sp.get(k)
        if isinstance(v, str):
            return v
    for k in ("latex", "math", "formula", "asciimath"):
        v = sp.get(k)
        if isinstance(v, str):
            return v
    return ""

def is_inline_equation_span(sp: Dict[str, Any]) -> bool:
    t = norm_type(sp.get("type"))
    if "inline" in t and ("equation" in t or "math" in t or "formula" in t):
        return True
    for k in ("latex", "math", "formula", "asciimath"):
        if isinstance(sp.get(k), str):
            return True
    return False

def is_interline_equation_block(blk: Dict[str, Any]) -> bool:
    t = norm_type(blk.get("type"))
    if not t:
        return False
    if "interline" in t and ("equation" in t or "math" in t or "formula" in t):
        return True
    if "display" in t and ("equation" in t or "math" in t or "formula" in t):
        return True
    if ("equation" in t or "math" in t or "formula" in t) and ("inline" not in t):
        return True
    return False

def extract_equation_text(blk: Dict[str, Any]) -> str:
    for k in ("content", "text", "latex", "math", "formula", "asciimath", "value"):
        v = blk.get(k)
        if isinstance(v, str) and v.strip():
            return v
    parts: List[str] = []
    for sp in iter_spans_from_block(blk):
        s = get_span_text(sp)
        if s:
            parts.append(s)
    return clean_spaces(" ".join(parts)) if parts else ""

def is_table_block(blk: Dict[str, Any]) -> bool:
    t = norm_type(blk.get("type"))
    if isinstance(blk.get("table_body"), (str, list)) or blk.get("html"):
        return True
    # 컨테이너/바디(캡션, 푸트노트 제외)
    if "table" in t and ("caption" not in t and "footnote" not in t):
        return True
    # ✅ 깊은 곳에 table span(html)이 있으면 표로 간주
    if has_table_span_deep(blk):
        return True
    return False

def is_image_block(blk: Dict[str, Any]) -> bool:
    t = norm_type(blk.get("type"))
    if "image" in t or "figure" in t or "img" in t or "picture" in t:
        return True
    if blk.get("img_path") or blk.get("image_path") or blk.get("path"):
        return True
    # ✅ 깊은 곳에 image span(img_path)이 있으면 이미지로 간주
    if has_image_span_deep(blk):
        return True
    return False

def extract_table_payload(blk: Dict[str, Any]) -> Dict[str, Any]:
    table_body = blk.get("table_body") or blk.get("html") or blk.get("content") or ""
    caption = blk.get("table_caption") or blk.get("caption") or []
    footnote = blk.get("table_footnote") or blk.get("footnote") or []

    # 하위 blocks에서 먼저 찾기
    if not table_body:
        for sub in iter_child_blocks(blk):
            table_body = sub.get("table_body") or sub.get("html") or sub.get("content")
            if table_body:
                caption = sub.get("table_caption") or sub.get("caption") or caption
                footnote = sub.get("table_footnote") or sub.get("footnote") or footnote
                break

    # 그래도 없으면 spans 깊게 뒤져서 type=table의 html 가져오기
    if not table_body:
        for sp in iter_spans_from_block_deep(blk):
            if norm_type(sp.get("type")) == "table":
                table_body = sp.get("html") or sp.get("content") or ""
                break

    return {
        "table_body": table_body or "",
        "table_caption": caption if isinstance(caption, (list, str)) else [],
        "table_footnote": footnote if isinstance(footnote, (list, str)) else [],
        "original_type": blk.get("type"),
    }

def extract_image_payload(blk: Dict[str, Any]) -> Dict[str, Any]:
    # 1) 현재 블록에서 먼저 시도
    img_path = blk.get("img_path") or blk.get("image_path") or blk.get("path") or ""
    alt = blk.get("alt") or blk.get("desc") or blk.get("description") or ""
    caption = blk.get("caption") or blk.get("image_caption") or blk.get("figure_caption")

    # 2) 없으면 하위 blocks / spans에서 재귀로 찾기
    if not img_path:
        for sp in iter_spans_from_block_deep(blk):
            sptype = norm_type(sp.get("type"))
            if ("image" in sptype) or ("figure" in sptype) or ("img" in sptype) or ("picture" in sptype):
                img_path = sp.get("img_path") or sp.get("image_path") or sp.get("path") or img_path
                alt = sp.get("alt") or sp.get("desc") or sp.get("description") or alt
                caption = sp.get("caption") or caption
                if img_path:
                    break

    return {
        "image_path": img_path or "",
        "alt": alt,
        "caption": caption if isinstance(caption, (list, str)) else None,
        "original_type": blk.get("type"),
    }

# ============================================================
# flush 도우미 (문장 단위 텍스트 축적)
# ============================================================
def flush_sentence(acc: List[Dict[str, Any]], buf_text: str, buf_bbox: Optional[List[float]], page_idx: int) -> Tuple[str, Optional[List[float]]]:
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

# ============================================================
# 블록 → 아이템 시퀀스 (표/이미지/행간수식은 블록 레벨, 나머지는 스팬)
# ============================================================
def iter_child_blocks(block: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    """block 내부의 blocks 트리를 재귀 순회"""
    for sub in (block.get("blocks") or []):
        yield sub
        yield from iter_child_blocks(sub)

def iter_spans_from_block_deep(block: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    """lines/spans + 하위 blocks까지 재귀로 spans 순회"""
    # 1층
    if isinstance(block.get("lines"), list) and block["lines"]:
        for line in block["lines"]:
            for sp in (line.get("spans") or []):
                yield sp
    for sp in (block.get("spans") or []):
        yield sp
    # 하위
    for sub in (block.get("blocks") or []):
        yield from iter_spans_from_block_deep(sub)

def fold_block_to_items(block: Dict[str, Any], page_idx: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    # 1) 표
    if is_table_block(block):
        payload = extract_table_payload(block)
        # 최종 안전장치: table_body가 여전히 비면 deep에서 한 번 더 확인
        if not payload.get("table_body"):
            if has_table_span_deep(block):
                for sp in iter_spans_from_block_deep(block):
                    if norm_type(sp.get("type")) == "table":
                        html = sp.get("html") or sp.get("content")
                        if html:
                            payload["table_body"] = html
                            break
        out.append({
            "type": "table",
            "bbox": get_bbox_any(block) or [0,0,0,0],
            "page_idx": page_idx,
            **payload
        })
        return out

    # 2) 이미지
    if is_image_block(block):
        payload = extract_image_payload(block)
        if not payload.get("image_path"):
            if has_image_span_deep(block):
                for sp in iter_spans_from_block_deep(block):
                    sptype = norm_type(sp.get("type"))
                    if ("image" in sptype) or ("figure" in sptype) or ("img" in sptype) or ("picture" in sptype):
                        img = sp.get("img_path") or sp.get("image_path") or sp.get("path")
                        if img:
                            payload["image_path"] = img
                            payload.setdefault("alt", sp.get("alt") or sp.get("desc") or sp.get("description") or "")
                            break
        out.append({
            "type": "image",
            "bbox": get_bbox_any(block) or [0,0,0,0],
            "page_idx": page_idx,
            **payload
        })
        return out

    # 3) 행간(디스플레이) 수식
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

    # 4) 텍스트/인라인 수식: 재귀 스팬 사용
    buf_text = ""
    buf_bbox: Optional[List[float]] = None

    for sp in iter_spans_from_block_deep(block):
        sptype = norm_type(sp.get("type"))
        txt = get_span_text(sp)
        bbox = get_bbox_any(sp)

        if "image" in sptype or "figure" in sptype or "img" in sptype or "picture" in sptype:
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
                "image_path": sp.get("img_path") or sp.get("image_path") or sp.get("path") or "",
                "table_caption": sp.get("table_caption") or [],
                "table_footnote": sp.get("table_footnote") or [],
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

        if txt:
            buf_text = clean_spaces((buf_text + " " + txt) if buf_text else txt)
            buf_bbox = union_bbox(buf_bbox, bbox)

        if buf_text and SENT_END_RE.search(buf_text):
            buf_text, buf_bbox = flush_sentence(out, buf_text, buf_bbox, page_idx)

    if buf_text:
        buf_text, buf_bbox = flush_sentence(out, buf_text, buf_bbox, page_idx)

    return out

def page_to_items(page: Dict[str, Any]) -> List[Dict[str, Any]]:
    seq: List[Dict[str, Any]] = []
    page_idx = page.get("page_idx", page.get("index", 0)) or 0
    for blk in get_blocks(page):
        seq.extend(fold_block_to_items(blk, page_idx))
    return seq

# ============================================================
# 파이프라인: 1) 제거 전처리 → 2) 평탄화
# ============================================================
def clean_one_json(src: Path, dst_cleaned: Path) -> Dict[str, Any]:
    data = json.loads(src.read_text(encoding="utf-8"))
    cleaned = process_node_salvage(data)  # 복구 없음, 지정 키만 삭제
    dst_cleaned.parent.mkdir(parents=True, exist_ok=True)
    dst_cleaned.write_text(json.dumps(cleaned, ensure_ascii=False, indent=2), encoding="utf-8")
    return cleaned

def _unique_path(dst_dir: Path, filename: str) -> Path:
    """dst_dir 아래에 filename으로 저장하되, 충돌 시 _1, _2 ... 접미사 부여."""
    p = dst_dir / filename
    if not p.exists():
        return p
    stem, ext = Path(filename).stem, Path(filename).suffix
    k = 1
    while True:
        cand = dst_dir / f"{stem}_{k}{ext}"
        if not cand.exists():
            return cand
        k += 1

def _list_target_files(inp: Path, only_problem: bool = True) -> List[Path]:
    """
    입력 경로가 파일이면 그 파일만, 폴더면 재귀 스캔.
    only_problem=True 면(폴더 스캔 시) 파일명에 '문제' 포함만 선택.
    """
    if inp.is_file():
        # 단일 파일은 무조건 대상
        return [inp]
    # 폴더 스캔
    patt = "*.json"
    files = list(inp.rglob(patt))
    if only_problem:
        files = [p for p in files if "문제" in p.stem or "문제" in str(p.parent)]
    return files

def run_pipeline():
    files = _list_target_files(INPUT_MIDDLE_DIR, only_problem=ONLY_PROBLEM_FILES)
    print(f"🔍 대상 파일: {len(files)}개")
    if not files:
        print("⚠️ 대상 파일이 없습니다. 경로 또는 필터(ONLY_PROBLEM_FILES)를 확인하세요.")
        return

    for src in files:
        # 출력은 모두 OUTPUT_CLEANED_DIR 한 폴더로 모으고,
        # 파일명 충돌 시 _1, _2 ... 자동 부여
        dst_flat = _unique_path(OUTPUT_CLEANED_DIR, src.name)
        try:
            clean_one_json(src, dst_flat)
            print(f"✅ {src} → {dst_flat.name} 생성 완료")
        except Exception as e:
            print(f"⚠️ 오류 ({src}): {e}")

# ============================================================
# 엔트리포인트
# ============================================================
if __name__ == "__main__":
    run_pipeline()
