#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import json, re, unicodedata

# =========================
# 경로/설정 (하드코딩)
# =========================
# 입력: 폴더 또는 단일 파일 경로
SRC_PATH = Path(r"exam_parser-main\01_middle_process\data\cleand\0111-2023-국어영역-국어영역-문제_middle.json")
# 출력: 폴더 또는 단일 파일 경로
DST_PATH = Path(r"exam_parser-main\01_middle_process\data\plain")

# 폴더 모드에서 매칭할 파일 패턴(재귀 탐색)
GLOB_PATTERN = "*.json"
RECURSIVE = True

# 변환 동작 파라미터
CONTAINERS = ["para_blocks"]     # 컨테이너 키
OUT_TYPE   = "text_converted"    # 변환 후 type
OUT_SUFFIX = "_converted"        # 출력 파일명 접미사 (예: foo.json -> foo_converted.json)
STRICT_MODE = True               # 보수적 규칙 적용
RM_PARENS_POLICY = "forbid"      # \mathrm 내부 괄호 처리: forbid | keep | strip
PRINT_LIMIT = 10                 # 변환 예시 출력 개수(0이면 조용히 동작)

# =========================
# 보수적 플레인화 규칙
# =========================
HANGUL = re.compile(r"[\uac00-\ud7a3]")
FORBIDDEN_CHARS = re.compile(r"[._,]")
TEXTCIRCLED_OCCURS = re.compile(r"\\textcircled\s*\{")
TEXTCIRCLED_NUM_ONLY = re.compile(r"^\s*\\textcircled\s*\{([0-9]{1,2})\}\s*$")
BIGCIRC_ONLY = re.compile(r"^\s*\\bigcirc\s*$")

def make_rm_allowed(paren_policy: str) -> re.Pattern:
    paren = r"|[()]" if paren_policy in ("keep", "strip") else ""
    token_group = rf"(?:[A-Za-z+\-\s~]|\\sim{paren})+"
    return re.compile(rf"^\s*\\mathrm\s*\{{{token_group}\}}\s*$")

TRANSLATE_MAP = str.maketrans({
    '\uFF3C': '\\', '\u2216': '\\', '\u29F5': '\\', '\uFE68': '\\',
    '｛': '{', '｝': '}', '（':'(', '）':')', '［':'[', '］':']',
})
ZERO_WIDTH = re.compile(r'[\u200B\u200C\u200D\uFEFF\u2060]')
NBSP_OR_SPACES = re.compile(r'[\u00A0\u2000-\u200A\u202F\u205F\u3000]')

def strip_math_delims(s: str) -> str:
    if s is None: return ""
    s = str(s).strip()
    if s.startswith("$") and s.endswith("$"):   return s[1:-1].strip()
    if s.startswith(r"\(") and s.endswith(r"\)"): return s[2:-2].strip()
    if s.startswith(r"\[") and s.endswith(r"\]"): return s[2:-2].strip()
    return s

def pre_normalize_light(s: str) -> str:
    if s is None: return ""
    s = str(s).translate(TRANSLATE_MAP)
    s = unicodedata.normalize('NFKC', s)
    s = ZERO_WIDTH.sub('', s)
    s = NBSP_OR_SPACES.sub(' ', s)
    s = re.sub(r"[ \t]+", " ", s).strip()
    return s

def to_circled_number_1_10(num_str: str) -> Optional[str]:
    try:
        n = int(num_str)
    except ValueError:
        return None
    return chr(0x2460 + (n - 1)) if 1 <= n <= 10 else None

def plainify_conservative(src: str, *, rm_parens_policy: str="forbid") -> Tuple[bool, Optional[str], str]:
    if src is None or str(src).strip() == "":
        return False, None, "empty"
    s = strip_math_delims(str(src))
    s = pre_normalize_light(s)
    if HANGUL.search(s):
        return False, None, "hangul_present"
    if len(TEXTCIRCLED_OCCURS.findall(s)) >= 2:
        return False, None, "textcircled_multiple"
    if FORBIDDEN_CHARS.search(s):
        return False, None, "forbidden_chars(._,)"
    if re.fullmatch(r"(?:\\sim|~)", s):
        return True, "~", "ok_sim_alone"
    m_num = TEXTCIRCLED_NUM_ONLY.match(s)
    if m_num:
        circ = to_circled_number_1_10(m_num.group(1))
        return (True, circ, "ok_textcircled_1_10") if circ else (False, None, "textcircled_out_of_range")
    if BIGCIRC_ONLY.match(s):
        return True, "○", "ok_bigcirc"
    RM_ALLOWED = make_rm_allowed(rm_parens_policy)
    if RM_ALLOWED.match(s):
        inner = s[s.find("{")+1 : s.rfind("}")]
        inner = re.sub(r"\s+", " ", inner).strip()
        inner = re.sub(r"\s*\\sim\s*", "~", inner)
        if rm_parens_policy == "strip":
            inner = inner.replace("(", "").replace(")", "")
        return True, inner, "ok_mathrm"
    return False, None, "not_in_allowlist"

def convert_inline_equation_strict(content: str, rm_parens_policy: str="forbid"):
    ok, text, reason = plainify_conservative(content, rm_parens_policy=rm_parens_policy)
    return (text if ok else None), ok, reason

# =========================
# 유틸
# =========================
def iter_nodes(obj: Any, path: Optional[List[Union[str,int]]] = None):
    if path is None: path = []
    if isinstance(obj, dict):
        yield obj, path
        for k, v in obj.items():
            new_path = path + [k]
            if isinstance(v, (dict, list)):
                yield from iter_nodes(v, new_path)
    elif isinstance(obj, list):
        yield obj, path
        for i, v in enumerate(obj):
            new_path = path + [i]
            if isinstance(v, (dict, list)):
                yield from iter_nodes(v, new_path)

def in_selected_containers(path: List[Union[str,int]], containers: List[str]) -> bool:
    return any(isinstance(p, str) and p in containers for p in path) if containers else True

def transform_out_name(src_name: str, out_suffix: str) -> str:
    """
    입력 파일명에서 `_middle`, `_plain`, `_converted` 같은 중간 접미사는 제거하고
    원하는 out_suffix를 붙여서 반환.
    """
    p = Path(src_name)
    stem = p.stem
    # 중간 산출물 접미사 제거
    for tail in ("_middle", "_plain", "_converted"):
        if stem.endswith(tail):
            stem = stem[: -len(tail)]
            break
    if out_suffix:
        stem = f"{stem}{out_suffix}"
    return f"{stem}{p.suffix}"

def compute_out_path(src_file: Path, src_root: Path, out_root: Path, out_suffix: str) -> Path:
    rel = src_file.relative_to(src_root)
    out_rel = rel.with_name(transform_out_name(rel.name, out_suffix))
    return out_root / out_rel

def ensure_parent(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

def make_preview(s: Optional[str], limit: int = 80) -> str:
    if s is None: return ""
    t = str(s).replace("\n", "\\n")
    return t if len(t) <= limit else (t[:limit-1] + "…")

# =========================
# 핵심 처리
# =========================
def process_file(src: Path, dst: Path):
    with src.open("r", encoding="utf-8") as f:
        data = json.load(f)

    stats = {"inline_total":0, "inline_in_containers":0, "converted_to_plain":0,
             "skipped_outside_containers":0, "skipped_reason_counts":{}}

    printed = 0
    for node, path in iter_nodes(data):
        if not (isinstance(node, dict) and node.get("type") == "inline_equation"):
            continue
        stats["inline_total"] += 1
        if not in_selected_containers(path, CONTAINERS):
            stats["skipped_outside_containers"] += 1
            continue

        stats["inline_in_containers"] += 1
        before = node.get("content", "")

        if STRICT_MODE:
            after_text, ok, reason = convert_inline_equation_strict(before, rm_parens_policy=RM_PARENS_POLICY)
        else:
            ok, reason, after_text = False, "strict_disabled", None

        if ok:
            node["content"] = after_text
            node["type"] = OUT_TYPE
            meta = node.get("meta") or {}
            meta["converted_from"] = "inline_equation"
            node["meta"] = meta
            stats["converted_to_plain"] += 1

            if PRINT_LIMIT > 0 and printed < PRINT_LIMIT:
                printed += 1
                print(f"[converted] {make_preview(before, 80)}  ->  {make_preview(after_text, 80)}")
        else:
            stats["skipped_reason_counts"][reason] = stats["skipped_reason_counts"].get(reason, 0) + 1

    ensure_parent(dst)
    with dst.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\n=== {src.name} ===")
    print(f" total_inline={stats['inline_total']}, in_containers={stats['inline_in_containers']}, "
          f"converted={stats['converted_to_plain']}, outside={stats['skipped_outside_containers']}, "
          f"reasons={stats['skipped_reason_counts']}")

def iter_input_files(root: Path, glob_pattern: str, recursive: bool = True) -> List[Path]:
    it = root.rglob(glob_pattern) if recursive else root.glob(glob_pattern)
    return [p for p in it if p.is_file()]

def process_dir(src_dir: Path, dst_dir: Path):
    files = iter_input_files(src_dir, GLOB_PATTERN, RECURSIVE)
    if not files:
        print(f"[WARN] No files under '{src_dir}' with pattern '{GLOB_PATTERN}'.")
        return
    print(f"[INFO] {len(files)} files found (pattern='{GLOB_PATTERN}', recursive={RECURSIVE}).")

    for i, src in enumerate(sorted(files), 1):
        dst = compute_out_path(src, src_root=src_dir, out_root=dst_dir, out_suffix=OUT_SUFFIX)
        print(f"\n[RUN {i}/{len(files)}] {src} -> {dst}")
        process_file(src, dst)

# =========================
# 실행부
# =========================
def main():
    src, dst = SRC_PATH, DST_PATH
    src_is_dir = src.is_dir()
    # 확장자 없으면 폴더 취급(예: ...\plain)
    dst_is_dir = dst.is_dir() or (not dst.suffix)

    if src_is_dir:
        if not dst_is_dir:
            print("[ERROR] SRC가 폴더면 DST도 폴더여야 합니다.")
            return
        dst.mkdir(parents=True, exist_ok=True)
        process_dir(src, dst)
    else:
        # 파일 -> 파일(그대로) 또는 파일 -> 폴더(자동 이름 생성)
        if dst_is_dir:
            # src.parent를 root로 하여 상대경로 유지 + OUT_SUFFIX 적용
            dst = compute_out_path(src, src_root=src.parent, out_root=dst, out_suffix=OUT_SUFFIX)
            ensure_parent(dst)
        else:
            ensure_parent(dst)
        print(f"[RUN] {src} -> {dst}")
        process_file(src, dst)

if __name__ == "__main__":
    main()
