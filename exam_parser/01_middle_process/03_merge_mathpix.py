# -*- coding: utf-8 -*-
import json, unicodedata, re
from pathlib import Path
from collections import OrderedDict
from typing import Any, Dict, List, Tuple, Optional
from difflib import SequenceMatcher

# =========================
# 하드코딩 경로 설정
# =========================
INPUT_DIR  = Path(r"exam_parser-main\01_middle_process\data\content_list")  # 원본 JSON 폴더
PATCH_DIR  = Path(r"exam_parser-main\01_middle_process\data\수능기출_2025_mathpix_converted\2025_only_mathpix")  # 패치 JSON 폴더(유사 파일명 허용)
OUTPUT_DIR = Path(r"exam_parser-main\01_middle_process\data\merge_mathpix")               # 결과 저장 폴더
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 퍼지 매칭 임계값(이상일 때 매칭으로 인정)
SIM_THRESHOLD = 0.70

# 파일명 정규화 시 지울 접미사/토큰들 (확장)
_SUFFIX_CLEAN_RE = re.compile(
    r"(?:_content_list|_choice_plain|_layout|_middle|_mathpix_converted|_converted|_result|_res|_merge|_merged)"
    r"|(?:[-_](?:문제|정답|해설))",
    re.I
)

# 메타 추출용
_ID_RE    = re.compile(r"^\s*(\d{4,6})")
_YEAR_RE  = re.compile(r"(19|20)\d{2}")
_TYPE_RE  = re.compile(r"(?:^|[-_（(])([가나다])(?:[-_)）]|$)")
_TOKEN_SPLIT_RE = re.compile(r"[^\w\u3131-\u318E\uAC00-\uD7A3]+")

PREFERRED_KEY_ORDER = ("type", "text", "content", "bbox", "page_idx", "_source_ids", "_order_index", "id")
FORMULA_TYPES = {"formula", "inline_equation"}

def _reorder_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    od = OrderedDict()
    for k in PREFERRED_KEY_ORDER:
        if k in d:
            od[k] = d[k]
    for k, v in d.items():
        if k not in od:
            od[k] = v
    return od

# =========================
# 공통 유틸
# =========================
def _load_any(path: Path):
    with open(path, "r", encoding="utf-8-sig") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        return obj, None, obj
    if isinstance(obj, dict):
        for k in ("items", "content_list", "blocks"):
            if isinstance(obj.get(k), list):
                return obj[k], k, obj
        return [obj], None, obj
    raise TypeError(f"Unsupported JSON structure in {path}")

def _save_any(path: Path, items: List[Dict[str, Any]], wrapper_key: Optional[str], original_obj: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    items = [_reorder_dict(it) for it in items]
    if wrapper_key is None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
    else:
        original_obj[wrapper_key] = items
        with open(path, "w", encoding="utf-8") as f:
            json.dump(original_obj, f, ensure_ascii=False, indent=2)

def _bbox_key(d: Dict[str, Any]) -> Tuple:
    bbox = tuple(d.get("bbox") or [])
    pg = d.get("page_idx", None)
    return (bbox, pg) if pg is not None else (bbox,)

# ----- 파일명 정규화/메타 -----
def _stem_norm(p: Path) -> str:
    name = _SUFFIX_CLEAN_RE.sub("", p.stem)
    s = unicodedata.normalize("NFKC", name).lower()
    return re.sub(r"[^\w\u3131-\u318E\uAC00-\uD7A3]+", "", s)

def _extract_id(name: str) -> Optional[str]:
    m = _ID_RE.search(name)
    return m.group(1) if m else None

def _extract_year(name: str) -> Optional[str]:
    m = _YEAR_RE.search(name)
    return m.group(0) if m else None

def _extract_type(name: str) -> Optional[str]:
    m = _TYPE_RE.search(name)
    return m.group(1) if m else None

_STOP_TOKENS = {
    "공개경쟁채용시험", "후보자선발시험", "국가고시", "영역", "과학기술", "행정",
    "가", "나", "다", "문제", "정답", "해설"
}

def _extract_tokens(name: str) -> set:
    s = unicodedata.normalize("NFKC", name)
    raw = [t for t in _TOKEN_SPLIT_RE.split(s) if t]
    toks = set()
    for t in raw:
        if len(t) < 2:
            continue
        if t in _STOP_TOKENS:
            continue
        toks.add(t.lower())
    return toks

def _fname_similarity(a: Path, b: Path) -> float:
    a_name = a.stem
    b_name = b.stem
    sa, sb = _stem_norm(a), _stem_norm(b)
    base = SequenceMatcher(None, sa, sb).ratio()
    contain_bonus = 0.15 if (sa and sb and (sa in sb or sb in sa)) else 0.0
    ida, idb = _extract_id(a_name), _extract_id(b_name)
    year_a, year_b = _extract_year(a_name), _extract_year(b_name)
    type_a, type_b = _extract_type(a_name), _extract_type(b_name)
    id_bonus   = 0.35 if (ida and idb and ida == idb) else 0.0
    year_bonus = 0.05 if (year_a and year_b and year_a == year_b) else 0.0
    type_bonus = 0.05 if (type_a and type_b and type_a == type_b) else 0.0
    toks_a = _extract_tokens(a_name)
    toks_b = _extract_tokens(b_name)
    inter = toks_a & toks_b
    tok_bonus = min(0.15, 0.05 * len(inter)) if inter else 0.0
    score = base + contain_bonus + id_bonus + year_bonus + type_bonus + tok_bonus
    return min(1.0, score)

def _candidate_pool_for(orig_path: Path, patch_files: List[Path]) -> List[Path]:
    ida = _extract_id(orig_path.stem)
    if ida:
        same_id = [pf for pf in patch_files if _extract_id(pf.stem) == ida]
        if same_id:
            return same_id
    return patch_files

def pick_patch_for(orig_path: Path, patch_files: List[Path], thr: float = SIM_THRESHOLD) -> Optional[Path]:
    pool = _candidate_pool_for(orig_path, patch_files)
    best, best_sc = None, -1.0
    for pf in pool:
        sc = _fname_similarity(orig_path, pf)
        if sc > best_sc:
            best_sc, best = sc, pf
    if best and best_sc >= thr:
        print(f"   ↳ match: {best.name} (sim={best_sc:.3f})")
        return best
    print(f"   ↳ no good match (best={best.name if best else 'NONE'}, sim={best_sc:.3f})")
    return None

# =========================
# image_path basename 처리
# =========================
def _basename_only(p: str) -> str:
    return re.sub(r"^.*[\\/]", "", p)

def _normalize_image_paths_inplace(obj: Any) -> None:
    if isinstance(obj, dict):
        if "image_path" in obj:
            val = obj["image_path"]
            if isinstance(val, str):
                obj["image_path"] = _basename_only(val)
            elif isinstance(val, list):
                obj["image_path"] = [_basename_only(v) if isinstance(v, str) else v for v in val]
        for v in obj.values():
            _normalize_image_paths_inplace(v)
    elif isinstance(obj, list):
        for v in obj:
            _normalize_image_paths_inplace(v)

# =========================
# (1)→① 변환 유틸
# =========================
CIRCLED_DIGIT_MAP = {
    "0": "⓪", "1": "①", "2": "②", "3": "③", "4": "④",
    "5": "⑤", "6": "⑥", "7": "⑦", "8": "⑧", "9": "⑨",
}
PAREN_ONE_DIGIT_RE = re.compile(r"[（(]\s*([0-9])\s*[)）]")

def _circled_digits_in_text(s: str) -> str:
    return PAREN_ONE_DIGIT_RE.sub(lambda m: CIRCLED_DIGIT_MAP.get(m.group(1), m.group(0)), s)

def _circled_digits_inplace(obj: Any, parent_type: Optional[str] = None) -> None:
    if isinstance(obj, dict):
        t = obj.get("type") or parent_type
        for k, v in obj.items():
            if isinstance(v, str):
                if t in FORMULA_TYPES:
                    continue
                obj[k] = _circled_digits_in_text(v)
            else:
                _circled_digits_inplace(v, t)
    elif isinstance(obj, list):
        for v in obj:
            _circled_digits_inplace(v, parent_type)
            
# =========================
# 경계 중복 글자 제거 유틸
# =========================
def _get_text_key_and_value(d: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """블럭 dict에서 (key, value) 반환. 우선순위: text > content."""
    if not isinstance(d, dict):
        return None, None
    v = d.get("text")
    if isinstance(v, str):
        return "text", v
    v = d.get("content")
    if isinstance(v, str):
        return "content", v
    return None, None

def _first_non_ws_char(s: str) -> Tuple[Optional[int], Optional[str]]:
    for i, ch in enumerate(s):
        if not ch.isspace():
            return i, ch
    return None, None

def _last_non_ws_char(s: str) -> Tuple[Optional[int], Optional[str]]:
    for i in range(len(s) - 1, -1, -1):
        if not s[i].isspace():
            return i, s[i]
    return None, None

def _trim_adjacent_duplicate_boundary_in_list(lst: List[Any]) -> None:
    """
    같은 리스트 내에서, mathpix=True 블럭(a) 바로 뒤 블럭(b)의
    첫 비공백 문자와 a의 마지막 비공백 문자가 같으면
    a의 끝 글자 1개를 제거한다.
    """
    n = len(lst)
    for i in range(n - 1):
        a, b = lst[i], lst[i + 1]
        if not (isinstance(a, dict) and isinstance(b, dict)):
            continue
        if a.get("mathpix") is not True:
            continue

        k1, s1 = _get_text_key_and_value(a)
        k2, s2 = _get_text_key_and_value(b)
        if not (k1 and isinstance(s1, str) and k2 and isinstance(s2, str)):
            continue

        i_last, ch_last = _last_non_ws_char(s1)
        i_first, ch_first = _first_non_ws_char(s2)
        if i_last is None or i_first is None:
            continue

        if ch_last == ch_first:
            # 앞(=mathpix True) 블럭의 마지막 글자 제거
            a[k1] = s1[:i_last] + s1[i_last+1:]

# =========================
# ==== 출력 파일명 규칙 ====
# =========================
_OUT_SUFFIX_RE = re.compile(r"_content_list$", re.I)
def _out_path_for(orig_path: Path) -> Path:
    """abc_content_list.json -> abc_mathpix.json 로 변환"""
    new_stem = _OUT_SUFFIX_RE.sub("", orig_path.stem)
    return OUTPUT_DIR / f"{new_stem}_mathpix{orig_path.suffix}"

# =========================
# 핵심 로직
# =========================
def remap_patch_keys(patch_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    remapped: List[Dict[str, Any]] = []
    for it in patch_items:
        t = it.get("type")
        if t == "text" and "content" in it:
            new_it = dict(it)
            if "text" not in new_it:
                new_it["text"] = new_it["content"]
            new_it.pop("content", None)
            remapped.append(_reorder_dict(new_it))
        else:
            remapped.append(_reorder_dict(dict(it)))
    _normalize_image_paths_inplace(remapped)
    return remapped

def build_patch_groups(patch_items: List[Dict[str, Any]]) -> "OrderedDict[Tuple, List[Dict]]":
    groups: "OrderedDict[Tuple, List[Dict]]" = OrderedDict()
    for it in patch_items:
        key = _bbox_key(it)
        if not key or not key[0]:
            continue
        groups.setdefault(key, []).append(it)
    return groups

def replace_inline_equations_by_bbox(original_items: List[Dict[str, Any]],
                                     patch_groups: "OrderedDict[Tuple, List[Dict]]") -> int:
    for it in original_items:
        if "mathpix" not in it:
            it["mathpix"] = False

    replaced_count = 0
    i = 0
    n = len(original_items)
    patch_map = dict(patch_groups)

    while i < n:
        it = original_items[i]
        if it.get("type") != "inline_equation":
            i += 1
            continue
        key = _bbox_key(it)
        if not key or not key[0]:
            i += 1
            continue
        group = patch_map.get(key)
        if not group:
            i += 1
            continue

        new_group = []
        for g in group:
            ng = dict(g)
            ng["mathpix"] = True
            new_group.append(_reorder_dict(ng))

        original_items[i:i+1] = new_group
        n = len(original_items)
        replaced_count += 1
        patch_map.pop(key, None)

        page_str = f", page_idx={key[1]}" if (len(key) > 1 and key[1] is not None) else ""
        print(f"      - 적용완료: bbox={key[0]}{page_str}  (+{len(new_group)} items)")

        i += len(new_group)

    return replaced_count

# =========================
# 폴더 단위 처리
# =========================
def mathpix_api_folder_fuzzy():
    if not INPUT_DIR.exists():
        print(f"[ERR] INPUT_DIR not found: {INPUT_DIR}")
        return
    if not PATCH_DIR.exists():
        print(f"[ERR] PATCH_DIR not found: {PATCH_DIR}")
        return
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    base_files  = sorted(INPUT_DIR.glob("*.json"))
    patch_files = sorted(PATCH_DIR.glob("*.json"))

    if not base_files:
        print(f"[WARN] No JSON files in {INPUT_DIR}")
        return
    if not patch_files:
        print(f"[WARN] No JSON files in {PATCH_DIR}")
        return

    total_files, total_replaced, total_true = 0, 0, 0
    for orig_path in base_files:
        print(f"\n[PAIR] {orig_path.name}")
        patch_path = pick_patch_for(orig_path, patch_files, thr=SIM_THRESHOLD)
        if not patch_path:
            print(f"[SKIP] No similar patch for {orig_path.name}")
            continue

        try:
            orig_items, orig_key, orig_obj = _load_any(orig_path)
            patch_items, _, _ = _load_any(patch_path)

            patch_items = remap_patch_keys(patch_items)
            patch_groups = build_patch_groups(patch_items)
            replaced = replace_inline_equations_by_bbox(orig_items, patch_groups)
            
            # mathpix=true 블럭과 다음 블럭 경계 중복 글자 제거
            _trim_adjacent_duplicate_boundary_in_list(orig_items)

            # 저장 직전 후처리
            _normalize_image_paths_inplace(orig_items)   # image_path → basename
            _circled_digits_inplace(orig_items)          # (1)(2)… → ①②… (수식 블록 제외)

            true_cnt = sum(1 for it in orig_items if isinstance(it, dict) and it.get("mathpix") is True)

            out_path = _out_path_for(orig_path)          # ★ 여기서 새 파일명 규칙 사용
            _save_any(out_path, orig_items, orig_key, orig_obj)

            print(f"[OK] {orig_path.name} -> {out_path.name} | replaced={replaced} | mathpix_true_items={true_cnt}")
            total_files += 1
            total_replaced += replaced
            total_true += true_cnt
        except Exception as e:
            print(f"[ERR] {orig_path.name}: {e}")

    print(f"\n=== SUMMARY ===\nmathpix_apied files : {total_files}\nReplaced blocks : {total_replaced}\nmathpix true items : {total_true}\nOutput dir      : {OUTPUT_DIR}")

# =========================
# 실행
# =========================
if __name__ == "__main__":
    mathpix_api_folder_fuzzy()
