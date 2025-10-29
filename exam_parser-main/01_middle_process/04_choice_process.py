# -*- coding: utf-8 -*-
from __future__ import annotations

import json, re, unicodedata
from pathlib import Path
from typing import Any, Dict, List, Tuple

# =========================
# 하드코딩 경로
# =========================
IN_DIR  = Path(r"exam_parser-main\01_middle_process\data\content_list")     # 입력 루트
OUT_DIR = Path(r"exam_parser-main\01_middle_process\data\choice_process") # 출력 루트 (원하는 경로로 변경)

# =========================
# 매핑 규칙
# 7 -> ㄱ,  L/l -> ㄴ,  c -> ㄷ,  2 -> ㄹ
# =========================
TOKEN_MAP = {
    "7": "ㄱ",
    "L": "ㄴ", "l": "ㄴ",
    "c": "ㄷ",
    "2": "ㄹ",
    "ㄱ": "ㄱ", "ㄴ": "ㄴ", "ㄷ": "ㄷ", "ㄹ": "ㄹ",
}

# 원숫자(①-⑳, ⓪)
RE_CIRCLED_NUM = r"[\u2460-\u2473\u24ea]"
# 코드/한글 토큰
RE_CODE_TOKEN = r"[7Ll2cㄱㄴㄷㄹ]"
# 토큰 구분자(쉼표/전각쉼표/가운뎃점/중점)
RE_SEP = r"[,，·ㆍ]"
# <보기> 패턴(대괄호/꺾쇠 허용)
RE_BOGI = r"[<\[]?\s*보기\s*[>\]]?"

# 원숫자 + 토큰목록 패턴
PATTERN = re.compile(
    rf"({RE_CIRCLED_NUM})"                       # 1: 원숫자
    rf"(\s*[:：\.\)\]]?\s*)"                     # 2: 선택 구분기호와 공백
    rf"("                                        # 3: 토큰 시퀀스
    rf"{RE_CODE_TOKEN}"
    rf"(?:\s*{RE_SEP}\s*{RE_CODE_TOKEN})*"
    rf")"
)

# <보기> 바로 뒤 '7.' → 'ㄱ.' (콜론/공백 허용)
PAT_BOGI_7_DOT = re.compile(rf"({RE_BOGI})\s*[:：]?\s*(7)(\.)")

FORMULA_TYPES = {"formula", "inline_equation"}

# 이웃 탐색 반경(앞뒤 한 칸)
NEIGHBOR_RADIUS = 1
_digit_re = re.compile(r"^\d+$")

# ---------- 유틸 ----------

def _is_all_numeric_tokens(seq: str) -> bool:
    parts = re.split(RE_SEP, seq)
    any_token = False
    for p in parts:
        t = unicodedata.normalize("NFKC", p).strip()
        if not t:
            continue
        any_token = True
        if not _digit_re.fullmatch(t):
            return False
    return any_token  # 토큰이 있고 전부 숫자면 True

def _string_has_nonnumeric_pattern(s: str) -> bool:
    """
    문자열 안에 '원숫자+토큰들'이 있고, 그중 하나라도 '비숫자'(ㄱ/ㄴ/ㄷ/ㄹ/L/c)를 포함하면 True
    """
    for m in PATTERN.finditer(s):
        seq = m.group(3)
        if not _is_all_numeric_tokens(seq):
            return True
    return False

def _apply_bogi_head_rule(s: str) -> str:
    # <보기> 바로 뒤 7. -> ㄱ. (콜론/공백 허용)
    def _repl(m: re.Match) -> str:
        bogi = m.group(1)
        dot = m.group(3)  # '.'
        return f"{bogi} ㄱ{dot}"
    return PAT_BOGI_7_DOT.sub(_repl, s)

def _normalize_token(tok: str) -> str:
    t = unicodedata.normalize("NFKC", tok).strip()
    return TOKEN_MAP.get(t, tok)

def _normalize_item_seq(seq: str, *, force_numeric_convert: bool) -> str:
    """
    force_numeric_convert=False이면 '전부 숫자' 시퀀스는 보존.
    True이면 (컨텍스트상 선택지로 판단) 숫자만 있어도 7/2을 ㄱ/ㄹ로 치환.
    """
    if not force_numeric_convert and _is_all_numeric_tokens(seq):
        return seq
    parts = re.split(RE_SEP, seq)
    mapped = [_normalize_token(p) for p in parts]
    return ", ".join(m.strip() for m in mapped if m.strip())

def _normalize_in_text(s: str, *, force_numeric_convert: bool) -> str:
    # 1) <보기> 규칙
    s = _apply_bogi_head_rule(s)

    # 2) 원숫자+토큰 목록 변환
    def _repl(m: re.Match) -> str:
        circled = m.group(1)
        glue = m.group(2)
        items = m.group(3)
        new_items = _normalize_item_seq(items, force_numeric_convert=force_numeric_convert)
        return f"{circled}{glue}{new_items}"

    return PATTERN.sub(_repl, s)

# ---------- 컨텍스트 스캔 ----------

def _element_has_nonnumeric(obj: Any, parent_type: str | None) -> bool:
    """
    요소 내부에 '비숫자 토큰이 포함된 원숫자+토큰목록'이 하나라도 있으면 True.
    formula/inline_equation은 스캔 제외.
    """
    if isinstance(obj, dict):
        t = obj.get("type") or parent_type
        if t in FORMULA_TYPES:
            return False
        for v in obj.values():
            if isinstance(v, str):
                if _string_has_nonnumeric_pattern(v):
                    return True
            else:
                if _element_has_nonnumeric(v, t):
                    return True
        return False
    elif isinstance(obj, list):
        return any(_element_has_nonnumeric(v, parent_type) for v in obj)
    elif isinstance(obj, str):
        return _string_has_nonnumeric_pattern(obj)
    return False

# ---------- 변환(이웃 기반) ----------

def _walk_with_context_in_list(lst: List[Any], parent_type: str | None, inherited_force: bool) -> List[Any]:
    """
    같은 리스트 안의 이웃(앞/뒤 NEIGHBOR_RADIUS) 중 '비숫자 토큰 포함'이 하나라도 있으면
    해당 요소는 force_numeric_convert=True로 변환.
    이웃에도 전혀 없으면(=전부 숫자-only 컨텍스트) 변환하지 않음.
    """
    n = len(lst)
    # 1) 각 요소가 '비숫자 토큰 포함'인지 사전 스캔
    flags = [ _element_has_nonnumeric(e, parent_type) for e in lst ]

    # 2) 이웃 컨텍스트 계산 + 변환
    out: List[Any] = []
    for i, e in enumerate(lst):
        neighbor_nonnum = any(
            0 <= j < n and flags[j]
            for j in range(i - NEIGHBOR_RADIUS, i + NEIGHBOR_RADIUS + 1)
        )
        force = inherited_force or neighbor_nonnum
        out.append(_walk(e, parent_type, force_numeric_convert=force))
    return out

def _walk(obj: Any, parent_type: str | None = None, *, force_numeric_convert: bool = False) -> Any:
    """
    JSON 전체 순회. 문자열을 변환하되,
    - formula/inline_equation은 건드리지 않음
    - 리스트는 이웃 컨텍스트를 고려하여 처리
    """
    if isinstance(obj, dict):
        t = obj.get("type") or parent_type
        if t in FORMULA_TYPES:
            return obj  # 수식 블록 통째로 보존
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            if isinstance(v, str):
                out[k] = _normalize_in_text(v, force_numeric_convert=force_numeric_convert)
            else:
                out[k] = _walk(v, t, force_numeric_convert=force_numeric_convert)
        return out

    elif isinstance(obj, list):
        # 리스트 단위로 이웃 컨텍스트 스캔/적용
        return _walk_with_context_in_list(obj, parent_type, inherited_force=force_numeric_convert)

    else:
        if isinstance(obj, str):
            return _normalize_in_text(obj, force_numeric_convert=force_numeric_convert)
        return obj

# ---------- 입출력 ----------

def _mapped_name(name: str) -> str:
    """파일명에서 '_content_list' 제거 + '_process' 접미사 적용"""
    name2 = name.replace("_content_list", "")

    # if name2.endswith(".jsonl"):
    #     return name2[:-6] + "_process.jsonl"
    # elif name2.endswith(".json"):
    #     return name2[:-5] + "_process.json"
    # else:
    #     return name2 + "_process"
    return name2

def _out_path_for(p: Path) -> Path:
    """
    IN_DIR 기준 상대 경로를 유지하여 OUT_DIR에 저장.
    파일명은 '_content_list' 제거 + '_process' 접미사로 변환.
    """
    try:
        rel = p.relative_to(IN_DIR)  # IN_DIR 하위일 때만
        out_dir = OUT_DIR / rel.parent
        out_name = _mapped_name(rel.name)
    except ValueError:
        # 혹시 IN_DIR 밖이면 평면 저장
        out_dir = OUT_DIR
        out_name = _mapped_name(p.name)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / out_name

def _process_json_file(path: Path) -> None:
    outp = _out_path_for(path)
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as fin, outp.open("w", encoding="utf-8") as fout:
            for line in fin:
                line = line.rstrip("\n")
                if not line.strip():
                    fout.write("\n")
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    fout.write(line + "\n")
                    continue
                obj2 = _walk(obj)
                fout.write(json.dumps(obj2, ensure_ascii=False) + "\n")
    else:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        data2 = _walk(data)
        with outp.open("w", encoding="utf-8") as f:
            json.dump(data2, f, ensure_ascii=False, indent=2)
    print(f"[OK] {path} -> {outp}")

def main() -> None:
    if not IN_DIR.exists():
        print(f"[ERR] 입력 폴더 없음: {IN_DIR}")
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    targets: List[Path] = []
    for ext in ("*.json", "*.jsonl"):
        targets.extend(IN_DIR.rglob(ext))
    if not targets:
        print(f"[WARN] 처리할 JSON이 없습니다: {IN_DIR}")
        return

    for p in sorted(targets):
        try:
            _process_json_file(p)
        except Exception as e:
            print(f"[FAIL] {p}: {e}")

if __name__ == "__main__":
    main()
