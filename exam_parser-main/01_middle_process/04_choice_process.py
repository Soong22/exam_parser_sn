# -*- coding: utf-8 -*-
from __future__ import annotations

import json, re, unicodedata
from pathlib import Path
from typing import Any, Dict, List, Tuple

# =========================
# 하드코딩 경로
# =========================
IN_DIR  = Path(r"exam_parser-main\01_middle_process\data\layout")     # 입력 루트(파일 또는 폴더)
OUT_DIR = Path(r"exam_parser-main\01_middle_process\data\choice_process_mathpix")   # 출력 루트 (폴더)

# =========================
# 매핑 규칙
# 7 -> ㄱ,  L/l -> ㄴ,  c/C -> ㄷ,  2 -> ㄹ
# =========================
TOKEN_MAP = {
    "7": "ㄱ",
    "L": "ㄴ", "l": "ㄴ",
    "c": "ㄷ", "C": "ㄷ",
    "2": "ㄹ",
    "ㄱ": "ㄱ", "ㄴ": "ㄴ", "ㄷ": "ㄷ", "ㄹ": "ㄹ",
    "드": "ㄷ",
}

# 원숫자(①-⑳, ⓪)
RE_CIRCLED_NUM = r"[\u2460-\u2473\u24ea]"
# 코드/한글 토큰 (대문자 C 허용)
RE_CODE_TOKEN = r"[7Ll2cCㄱㄴㄷㄹ]"
# 토큰 구분자(콤마/전각콤마/가운뎃점/중점/세미콜론)
RE_SEP = r"[,，·ㆍ;]"
# <보기> 패턴(대괄호/꺾쇠 허용) — '보 기'처럼 내부 공백 허용
RE_BOGI = r"[<\[]?\s*보\s*기\s*[>\]]?"
RE_BOGI_RX = re.compile(RE_BOGI)  # 세그먼트 분할용

# 원숫자 + 토큰목록 패턴
PATTERN = re.compile(
    rf"({RE_CIRCLED_NUM})"                       # 1: 원숫자
    rf"(\s*[:：\.\)\]]?\s*)"                     # 2: 선택 구분기호와 공백
    rf"("                                        # 3: 토큰 시퀀스
    rf"{RE_CODE_TOKEN}"
    rf"(?:\s*{RE_SEP}\s*{RE_CODE_TOKEN})*"
    rf")"
)

# 줄머리 열거 토큰(7/L/l/c/C/2/ㄱㄴㄷㄹ/드) + '.'를 문장/블록 시작이나 공백 뒤에서 치환
PAT_HEAD_ENUM = re.compile(
    r"(?P<prefix>(^|[\s\(\[\{<]))"          # 문장/블록 시작 또는 공백류/괄호 뒤
    r"(?P<tok>7|L|l|c|C|2|ㄱ|ㄴ|ㄷ|ㄹ|드)"    # 토큰
    r"\s*\."                                # 점
)

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
    문자열 안에 '원숫자+토큰들'이 있고, 그중 하나라도 '비숫자'(ㄱ/ㄴ/ㄷ/ㄹ/L/c/C/2 등)를 포함하면 True
    """
    for m in PATTERN.finditer(s):
        seq = m.group(3)
        if not _is_all_numeric_tokens(seq):
            return True
    return False

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

# ======== '연속 등장일 때만 치환' 로직 ========

def _count_head_markers(s: str) -> int:
    """문자열 내 줄머리 열거 토큰(… '.') 출현 개수"""
    return len(list(PAT_HEAD_ENUM.finditer(s or "")))

def _has_inline_token_list_sequence_in_match(seq: str) -> bool:
    """원형숫자 뒤 토큰 시퀀스가 2개 이상일 때 True"""
    parts = [p.strip() for p in re.split(RE_SEP, seq or "") if p.strip()]
    return len(parts) >= 2

def _has_any_inline_token_list_sequence(s: str) -> bool:
    """문자열 전체에 '원숫자+토큰목록' 중 2개 이상 토큰인 경우가 하나라도 있으면 True"""
    for m in PATTERN.finditer(s or ""):
        if _has_inline_token_list_sequence_in_match(m.group(3)):
            return True
    return False

def _apply_head_enum_rule(s: str) -> str:
    def _repl(m):
        prefix = m.group("prefix")
        tok = m.group("tok")
        mapped = TOKEN_MAP.get(tok, tok)
        if mapped.endswith("."):
            return f"{prefix}{mapped}"
        return f"{prefix}{mapped}."
    return PAT_HEAD_ENUM.sub(_repl, s)

# ---- 세그먼트(= <보기> 뒤 구간) 전용 정규화 ----
_RE_SEGMENT_LEADING_7DOT = re.compile(r"^\s*7\s*\.")

def _normalize_only_segment(
    seg: str,
    *,
    force_numeric_convert: bool,
    allow_head_enums: bool | None = None,
    allow_inline_lists: bool | None = None,
) -> str:
    """
    <보기> 태그 '뒤'의 한 구간만 변환.
    - allow_* 가 None이면 세그먼트 내부에서 자체 판정
    - allow_* 를 넘기면 그 기준으로만 적용 (리스트-세그먼트 전역 판단 결과 주입용)
    """
    if not seg:
        return seg

    text = seg

    # 연속판단(세그먼트 내부 또는 주입)
    _head_count = _count_head_markers(text)
    local_allow_head_enums = (_head_count >= 2)
    local_allow_inline_lists = _has_any_inline_token_list_sequence(text)

    if allow_head_enums is None:
        allow_head_enums = local_allow_head_enums
    if allow_inline_lists is None:
        allow_inline_lists = local_allow_inline_lists

    # 세그먼트 선두 '7.' → 'ㄱ.' (헤드 연속 조건 충족 시 1회)
    if allow_head_enums and _RE_SEGMENT_LEADING_7DOT.match(text):
        text = _RE_SEGMENT_LEADING_7DOT.sub("ㄱ.", text, count=1)

    # 줄머리 열거 치환 (연속일 때만)
    if allow_head_enums:
        text = _apply_head_enum_rule(text)

    # 원형숫자(①②…)+토큰 목록 변환 — 해당 매치의 목록 길이가 2개 이상일 때만 변환
    def _repl(m: re.Match) -> str:
        circled = m.group(1)
        glue = m.group(2)
        items = m.group(3)

        if not allow_inline_lists or not _has_inline_token_list_sequence_in_match(items):
            return m.group(0)

        new_items = _normalize_item_seq(items, force_numeric_convert=force_numeric_convert)
        return f"{circled}{glue}{new_items}"

    text = PATTERN.sub(_repl, text)
    return text

def _normalize_in_text(s: str, *, force_numeric_convert: bool) -> str:
    """
    전체 문자열에서 <보기> 태그 '뒤' 구간에만 변환을 적용.
    <보기>가 없으면 원문 그대로 반환.
    <보기>가 여러 번 있으면 각각의 뒤 구간에만 개별 적용.
    """
    if not s:
        return s

    out_parts: List[str] = []
    last_idx = 0
    for m in RE_BOGI_RX.finditer(s):
        # 태그 이전 구간 -> 그대로 보존 (태그 자체도 포함해 보존)
        out_parts.append(s[last_idx:m.end()])
        # 태그 뒤 구간(다음 <보기> 혹은 문자열 끝 전까지)을 변환
        next_m = RE_BOGI_RX.search(s, m.end())
        seg_end = next_m.start() if next_m else len(s)
        seg = s[m.end():seg_end]
        seg_norm = _normalize_only_segment(seg, force_numeric_convert=force_numeric_convert)
        out_parts.append(seg_norm)
        last_idx = seg_end

    if last_idx == 0:
        # <보기>가 전혀 없으면 변환하지 않음
        return s

    # 마지막 나머지(다음 <보기> 이후로 이어지는 꼬리)는 이미 처리됨. 조립해서 반환.
    return "".join(out_parts)

# ---------- 컨텍스트 스캔 ----------

def _element_has_nonnumeric(obj: Any, parent_type: str | None) -> bool:
    """
    요소 내부에 '비숫자 토큰이 포함된 원숫자+토큰목록'이 하나라도 있으면 True.
    formula/inline_equation은 스캔 제외.
    (참고: 이 함수는 이웃 강제 변환 여부 판단용이며, <보기> 뒤 제한과는 무관)
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

# ---------- 변환(세그먼트 & 이웃 기반) ----------

def _walk_with_context_in_list(lst: List[Any], parent_type: str | None, inherited_force: bool) -> List[Any]:
    """
    리스트 내부에서:
    1) dict/text 형태 블록이 섞여 있고 '<보기>'가 보이면, 그 뒤 연속 텍스트 블록을 '세그먼트'로 묶어
       세그먼트 전체 기준(헤드 합산, 목록 존재)으로 허용 여부를 판단하고,
       각 블록의 'text'에 _normalize_only_segment를 주입 플래그(allow_*)와 함께 적용.
    2) 그렇지 않은 일반 리스트는 기존의 '이웃 컨텍스트' 규칙으로 처리.
    """
    # (A) 텍스트 블록 존재 여부 확인
    has_text_blocks = any(isinstance(e, dict) and (e.get("type") == "text") and isinstance(e.get("text"), str) for e in lst)
    if not has_text_blocks:
        # 기존 이웃 규칙
        n = len(lst)
        flags = [_element_has_nonnumeric(e, parent_type) for e in lst]
        out: List[Any] = []
        for i, e in enumerate(lst):
            neighbor_nonnum = any(0 <= j < n and flags[j] for j in range(i-NEIGHBOR_RADIUS, i+NEIGHBOR_RADIUS+1))
            force = inherited_force or neighbor_nonnum
            out.append(_walk(e, parent_type, force_numeric_convert=force))
        return out

    # (B) 텍스트 블록 세그먼트 처리
    out = list(lst)
    i = 0
    n = len(lst)

    def _elem_text(idx: int) -> str:
        e = lst[idx]
        if isinstance(e, dict) and e.get("type") == "text" and isinstance(e.get("text"), str):
            return e["text"]
        return ""

    while i < n:
        txt = _elem_text(i)
        if txt and RE_BOGI_RX.search(txt):
            # 세그먼트 시작: i (보기도 포함, 실제 변환은 i+1부터)
            j = i + 1
            seg_indices: List[int] = []
            while j < n:
                # 다음 <보기>를 만나면 종료
                t2 = _elem_text(j)
                if t2 and RE_BOGI_RX.search(t2):
                    break
                # text 타입만 세그먼트로 포함
                if not (isinstance(lst[j], dict) and lst[j].get("type") == "text" and isinstance(lst[j].get("text"), str)):
                    break
                seg_indices.append(j)
                j += 1

            if seg_indices:
                # 세그먼트 전체 텍스트 합쳐 허용 여부 계산
                combined = "\n".join(_elem_text(k) for k in seg_indices if _elem_text(k))
                allow_head = (_count_head_markers(combined) >= 2)
                allow_list = _has_any_inline_token_list_sequence(combined)

                # 세그먼트 내 각 블록 텍스트 정규화 (주입 플래그 사용)
                for k in seg_indices:
                    e = lst[k]
                    if isinstance(e, dict):
                        t = e.get("text")
                        if isinstance(t, str):
                            new_t = _normalize_only_segment(
                                t,
                                force_numeric_convert=inherited_force,
                                allow_head_enums=allow_head,
                                allow_inline_lists=allow_list,
                            )
                            new_e = dict(e)
                            new_e["text"] = new_t
                            out[k] = _walk(new_e, parent_type, force_numeric_convert=inherited_force)
                        else:
                            out[k] = _walk(e, parent_type, force_numeric_convert=inherited_force)
                    else:
                        out[k] = _walk(e, parent_type, force_numeric_convert=inherited_force)

            # 보기 블록(i)은 그대로 통과
            out[i] = _walk(lst[i], parent_type, force_numeric_convert=inherited_force)
            i = j  # 다음 탐색 위치로 점프
        else:
            # 일반 요소: 기존 규칙 적용
            out[i] = _walk(lst[i], parent_type, force_numeric_convert=inherited_force)
            i += 1

    return out

def _walk(obj: Any, parent_type: str | None = None, *, force_numeric_convert: bool = False) -> Any:
    """
    JSON 전체 순회. 문자열을 변환하되,
    - formula/inline_equation은 건드리지 않음
    - 리스트는 이웃 컨텍스트 및 <보기> 세그먼트 로직을 고려하여 처리
    - 문자열 변환은 _normalize_in_text가 <보기> 뒤 구간에만 적용
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
        # 리스트 단위로 <보기> 세그먼트 처리 + 이웃 컨텍스트 처리
        return _walk_with_context_in_list(obj, parent_type, inherited_force=force_numeric_convert)

    else:
        if isinstance(obj, str):
            return _normalize_in_text(obj, force_numeric_convert=force_numeric_convert)
        return obj

# ---------- 입출력 ----------

def _mapped_name(name: str) -> str:
    """파일명에서 '_content_list' 제거 (확장자 유지)"""
    return name.replace("_content_list", "")

def _out_path_for(p: Path, src_root: Path) -> Path:
    """
    src_root 기준 상대 경로를 유지하여 OUT_DIR에 저장.
    파일명은 '_content_list' 제거.
    (단일 파일 입력일 경우 src_root는 그 파일의 부모 폴더)
    """
    try:
        rel = p.relative_to(src_root)
        out_dir = OUT_DIR / rel.parent
        out_name = _mapped_name(rel.name)
    except ValueError:
        # 혹시 src_root 밖이면 평면 저장
        out_dir = OUT_DIR
        out_name = _mapped_name(p.name)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / out_name

def _process_json_file(path: Path, src_root: Path) -> None:
    outp = _out_path_for(path, src_root)
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
        print(f"[ERR] 입력 경로 없음: {IN_DIR}")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 입력이 파일인지 폴더인지 판별
    if IN_DIR.is_file():
        src_root = IN_DIR.parent
        targets = [IN_DIR]
    else:
        src_root = IN_DIR
        targets: List[Path] = []
        for ext in ("*.json", "*.jsonl"):
            targets.extend(IN_DIR.rglob(ext))

    if not targets:
        print(f"[WARN] 처리할 JSON이 없습니다: {IN_DIR}")
        return

    for p in sorted(targets):
        try:
            _process_json_file(p, src_root)
        except Exception as e:
            print(f"[FAIL] {p}: {e}")

if __name__ == "__main__":
    main()
