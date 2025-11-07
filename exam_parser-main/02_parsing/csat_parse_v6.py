from __future__ import annotations

import json, re, os, unicodedata, glob, sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from table_utils import sanitize_table_html
from image_utils import ImageConfig, classify_and_store_image

# =============================================================================
# 1) 상수·정규식
# =============================================================================

_DOT_DATE_TAIL_GUARD = (
    r"(?!"
    r"\s*\d{1,2}\s*[.。．]\s*"
    r"(?:\d{2,4}\s*[.。．]\s*)?"
    r"(?:$|[^\w가-힣]|[년월일])"
    r")"
)

RE_CIRCLED_NUM = r"[\u2460-\u2473\u24ea\u24f5-\u24fe]"
RE_MARKER_TOKEN = re.compile(r"\(?\s*(" + RE_CIRCLED_NUM + r")\s*\)?")

RE_Q_HEADER = re.compile(
    rf"^(?!\s*{RE_CIRCLED_NUM})\s*(?:문\s*)?([1-9]\d{{0,2}})\s*(?:\)|[.。．])\s*{_DOT_DATE_TAIL_GUARD}",
    re.UNICODE,
)
RE_LEAD_QNUM = re.compile(
    rf"^\s*(?:문\s*)?\d{{1,3}}\s*(?:\)\s*|[.。．]\s*{_DOT_DATE_TAIL_GUARD})",
    re.UNICODE,
)

RE_BOGI_TAG_LINE = re.compile(r"(?m)^[ \t]*[<\[]?\s*보\s*기\s*[>\]]?")
RE_BOGI_TAG_AT_START = re.compile(r"^[ \t]*[<\[]?\s*보\s*기\s*[>\]]?\s*")
RE_BOGI_TAG = re.compile(r"[<\[]?\s*보\s*기\s*[>\]]?", re.UNICODE)

_PH_RE = re.compile(r"<<(?:IMG|FORM|TBL)_\d+>>")

RE_VIEW_ENUM_START = re.compile(r"(?:^|\n|\s)(?:ㄱ|ㄴ|ㄷ|ㄹ|ㅁ|ㅂ|ㅅ|ㅇ|ㅈ|ㅊ|ㅋ|ㅌ|ㅍ|ㅎ)\s*[)\.]\s*")
RE_CHOICE_INLINE = re.compile(r"[\u2460-\u2473]")
RE_INLINE_TEX = re.compile(r"(\\\(.+?\\\))", re.S)
RE_HTML_TABLE = re.compile(r"<\s*table\b.*?</\s*table\s*>", re.I | re.S)
RE_ZERO_WIDTH = re.compile("[\u200B\u200C\u200D\u200E\u200F\u2060\u2066\u2067\u2068\u2069\ufeff\u00ad\u2028\u2029]")
RE_VIEW_UNIT = re.compile(r"(?<!\S)([㉠-㉯]|[ㄱ-ㅎ])\s*[)\.]?\s*")

UC2NUM = {chr(0x2460 + i): str(i + 1) for i in range(20)}
UC2NUM.update({"\u24ea": "0"})
UC2NUM.update({chr(0x24F5 + i): str(i + 1) for i in range(10)})

NUM2UC = {"0": "\u24ea", **{str(i): chr(0x2460 + (i - 1)) for i in range(1, 21)}}

IMAGES_SRC_DIR = r"images"
IMAGES_IMG_DIR = os.path.join(IMAGES_SRC_DIR, "image")
IMAGES_TBL_DIR = os.path.join(IMAGES_SRC_DIR, "table")
IMAGES_FORM_DIR = os.path.join(IMAGES_SRC_DIR, "formula")
IMAGES_INDEX_DIR = os.path.join(IMAGES_SRC_DIR, "_index")
IMAGE_MOVE = False
IMAGE_OVERWRITE = True

LAYOUT_DIR = r"exam_parser-main\01_middle_process\data\2025_layout"
FORMAT_DIR = r"exam_parser-main\02_parsing\data\00_final\00_2025"
ANSWER_DIR = r"exam_parser-main\02_parsing\data\정답\수능정답파일.jsonl"

# =============================================================================
# 2) 경로/글로벌 유틸
# =============================================================================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _normalize_compat_jamo(s: str) -> str:
    if not s:
        return s
    s2 = unicodedata.normalize("NFKC", s)
    JAMO_CHO_SRC = "ᄀᄁᄂᄃᄄᄅᄆᄇᄈᄉᄊᄋᄌᄍᄎᄏᄐᄑᄒ"
    JAMO_CHO_DST = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
    trans = str.maketrans({s: d for s, d in zip(JAMO_CHO_SRC, JAMO_CHO_DST)})
    return s2.translate(trans)

def normalize_text(s: str) -> str:
    if not s:
        return ""
    s = _normalize_compat_jamo(s)
    s = RE_ZERO_WIDTH.sub("", s)
    return s

def strip_leading_qnum(s: str) -> str:
    return RE_LEAD_QNUM.sub("", s, count=1)

def clean_text(s: str) -> str:
    if not s:
        return ""
    s = normalize_text(s).replace("\u00A0", " ").replace("$", "")
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def clean_text_keep_newlines(s: str) -> str:
    if not s:
        return ""
    s = normalize_text(s).replace("\u00A0", " ").replace("$", "")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"[ \t\r]*\n[ \t\r]*", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def _abs_norm(p: str) -> str:
    p2 = os.path.abspath(os.path.expanduser(os.path.expandvars(p or "")))
    return unicodedata.normalize("NFC", p2)

def _to_nfd(p: str) -> str:
    return unicodedata.normalize("NFD", p or "")

def safe_glob(dir_path: str, pattern: str, recursive: bool = True) -> List[str]:
    out: List[str] = []
    if not dir_path:
        return out
    d_nfc = _abs_norm(dir_path)
    d_nfd = _to_nfd(d_nfc) if sys.platform == "darwin" else d_nfc
    pats = [pattern]
    if pattern == "*-정답.jsonl":
        pats.append("*정답*.jsonl")
    for root in {d_nfc, d_nfd}:
        for pat in pats:
            glob_pat = os.path.join(root, "**", pat) if recursive else os.path.join(root, pat)
            out.extend(glob.glob(glob_pat, recursive=recursive))
    uniq = sorted({_abs_norm(p) for p in out})
    return uniq

# =============================================================================
# 3) 수식/이미지/테이블 플레이스홀더
# =============================================================================

def _strip_math_delims(s: str) -> str:
    s = s.strip()
    if s.startswith(r"\(") and s.endswith(r"\)"):
        return s[2:-2].strip()
    return s

FORMULA_BLOCK_TYPES = {"formula", "inline_equation"}

def _latex_from_block(b: Dict[str, Any]) -> str:
    return str(b.get("content") or b.get("latex") or b.get("text") or "")

def insert_formula_placeholders(text: str, cur: Dict[str, Any]) -> str:
    if not text:
        return text
    def repl(m: re.Match) -> str:
        latex_raw = m.group(1)
        forms = cur.setdefault("_forms", [])
        idx = len(forms)
        forms.append({"latex": latex_raw})
        return f"<<FORM_{idx}>>"
    return RE_INLINE_TEX.sub(repl, text)

def insert_table_placeholders(text: str, cur: Dict[str, Any]) -> str:
    if not text:
        return text
    out = []
    pos = 0
    while True:
        m = RE_HTML_TABLE.search(text, pos)
        if not m:
            out.append(text[pos:]); break
        out.append(text[pos:m.start()])
        html_src = m.group(0)
        tables = cur.setdefault("_tables", [])
        idx = len(tables)
        tables.append({"html": sanitize_table_html(html_src), "file_name": "", "bbox": [], "page_idx": None, "title": ""})
        out.append(f"<<TBL_{idx}>>")
        pos = m.end()
    return "".join(out)

# =============================================================================
# 4) 이미지 캡션/선택지 도우미
# =============================================================================

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
            if re.search(r"(그림|Fig\.?|도\s*\d+)", txt, re.I):
                return txt
    return ""

def explode_inline_choice_blocks(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for b in blocks:
        t = b.get("text", "")
        if not isinstance(t, str) or not t.strip():
            out.append(b); continue
        parts = RE_MARKER_TOKEN.split(t)
        if len(parts) >= 5:
            i = 1
            while i < len(parts):
                token = parts[i]; tail = parts[i+1] if i+1 < len(parts) else ""
                i += 2
                nb = dict(b); nb["text"] = (token or "") + (tail or "")
                out.append(nb)
        else:
            out.append(b)
    return out

def _rebalance_choice_assets(pairs: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    if not pairs: return pairs
    bodies = [b or "" for _, b in pairs]
    tokens = _PH_RE.findall("\n".join(bodies))
    if not tokens: return pairs
    only_assets = [(_PH_RE.sub("", b).strip() == "") for b in bodies]
    if all(only_assets):
        it = iter(tokens)
        new_pairs = []
        for lab, body in pairs:
            tok = next(it, None)
            new_pairs.append((lab, tok if tok is not None else body))
        return new_pairs
    return pairs

_CONNECTOR_ONLY_RE = re.compile(r"^[\s\-\~–—,·.\u00B7／/\\()\[\]{}<>:;|]*$")

def _is_marker_decor_line(s: str) -> bool:
    if not s or not s.strip(): return False
    u = normalize_text(s)
    circ_cnt = len(RE_MARKER_TOKEN.findall(u))
    if circ_cnt < 2: return False
    tmp = RE_MARKER_TOKEN.sub("", u)
    return _CONNECTOR_ONLY_RE.match(tmp.strip() or "") is not None

def _find_tail_start_idx(marks: List[Tuple[int,int,str,Optional[int],str]]) -> Optional[int]:
    n = len(marks)
    if n < 2: return None
    for i in range(n - 2, -1, -1):
        if marks[i][3] == 1:
            return i
    return n - 2

def slice_choices_by_markers_tailonly(big_text: str) -> Tuple[List[Tuple[str, str]], str]:
    if not big_text: return [], ""
    raw_keep = RE_ZERO_WIDTH.sub("", big_text or "")
    marks = _gather_markers_with_kind(raw_keep)
    if len(marks) < 2:
        pref = re.sub(r"[ \t]+\n", "\n", big_text or "")
        pref = re.sub(r"\n{3,}", "\n\n", pref).strip()
        return [], pref
    start_idx = _find_tail_start_idx(marks)
    if start_idx is None:
        pref = re.sub(r"[ \t]+\n", "\n", big_text or "")
        pref = re.sub(r"\n{3,}", "\n\n", pref).strip()
        return [], pref
    prefix = re.sub(r"[ \t]+\n", "\n", raw_keep[:marks[start_idx][0]]).strip()
    out: List[Tuple[str, str]] = []
    n = len(marks)
    for k in range(start_idx, n):
        s_k = marks[k][1]
        e_k = marks[k + 1][0] if (k + 1) < n else len(raw_keep)
        lab = marks[k][2]
        chunk = (raw_keep[s_k:e_k] or "").strip()
        out.append((lab, chunk))
    out = _rebalance_choice_assets(out)
    return out, prefix

def _peel_decor_edges(s: str) -> Tuple[str, str]:
    if not s: return "", ""
    lines = s.splitlines()
    i, j = 0, len(lines)-1
    head, tail = [], []
    while i <= j and _is_marker_decor_line(lines[i]): head.append(lines[i]); i += 1
    while j >= i and _is_marker_decor_line(lines[j]): tail.append(lines[j]); j -= 1
    decor = "\n".join(head + list(reversed(tail))).strip()
    rest  = "\n".join(lines[i:j+1]).strip()
    decor = re.sub(r"[ \t]+\n", "\n", decor).strip()
    rest  = re.sub(r"[ \t]+\n", "\n", rest).strip()
    return decor, rest

def _gather_markers_with_kind(text: str) -> List[Tuple[int, int, str, Optional[int], str]]:
    out: List[Tuple[int, int, str, Optional[int], str]] = []
    raw_keep = text or ""
    for m in RE_MARKER_TOKEN.finditer(raw_keep):
        ch = m.group(1)
        num = UC2NUM.get(ch)
        num_i = int(num) if num and num.isdigit() else None
        out.append((m.start(), m.end(), ch, num_i, "circled"))
    out.sort(key=lambda x: x[0])
    return out

def slice_choices_by_markers(big_text: str) -> Tuple[List[Tuple[str, str]], str]:
    raw_keep = RE_ZERO_WIDTH.sub("", big_text or "")
    raw_norm = normalize_text(big_text)
    marks: List[Tuple[int, int, str]] = []
    for m in RE_MARKER_TOKEN.finditer(raw_keep):
        ch = m.group(1)
        marks.append((m.start(), m.end(), ch))
    marks.sort(key=lambda x: x[0])
    if marks:
        prefix = clean_text_keep_newlines(raw_keep[:marks[0][0]])
        out: List[Tuple[str, str]] = []
        for i, (s, e, lab) in enumerate(marks):
            start = e
            end = marks[i+1][0] if i+1 < len(marks) else len(raw_keep)
            chunk = clean_text(raw_keep[start:end])
            if lab: out.append((lab, chunk))
        out = _rebalance_choice_assets(out)
        return out, prefix
    return [], clean_text_keep_newlines(raw_norm)

def _has_view_enumeration_with_content(s: str) -> bool:
    if not s or not s.strip(): return False
    u = normalize_text(s)
    matches = list(RE_VIEW_UNIT.finditer(u))
    if len(matches) < 2: return False
    substantive = 0
    for idx, m in enumerate(matches):
        start = m.end()
        end   = matches[idx+1].start() if idx+1 < len(matches) else len(u)
        chunk = clean_text_keep_newlines(u[start:end]).strip()
        if chunk and not _CONNECTOR_ONLY_RE.match(chunk):
            substantive += 1
    return substantive >= 1

def route_prefix_generically(prefix: str) -> Tuple[str, str]:
    if not prefix or not prefix.strip(): return "", ""
    decor, core = _peel_decor_edges(prefix)
    if not core.strip(): return decor, ""
    m = RE_BOGI_TAG_LINE.search(core)
    if m:
        before = core[:m.start()].rstrip()
        after  = core[m.start():].lstrip()
        to_body = "\n".join(x for x in [decor, re.sub(r"[ \t]+\n", "\n", before).strip()] if x).strip()
        aft = re.sub(r"[ \t]+\n", "\n", after)
        m2 = RE_VIEW_UNIT.search(aft)
        if m2:
            head = aft[:m2.start()].strip()
            packed = _pack_view_with_newlines(aft[m2.start():])
            to_view = (head + ("\n" if head and packed else "") + packed).strip()
        else:
            to_view = aft.strip()
        return to_body, to_view
    if _has_view_enumeration_with_content(core):
        to_view = _pack_view_with_newlines(re.sub(r"[ \t]+\n", "\n", core))
        return decor, to_view
    core2 = re.sub(r"[ \t]+\n", "\n", core).strip()
    to_body = (decor + ("\n" if decor and core2 else "") + core2).strip()
    return to_body, ""

def _pack_view_with_newlines(vtxt: str) -> str:
    s = (vtxt or "").strip()
    if not s: return s
    items = []
    matches = list(RE_VIEW_UNIT.finditer(s))
    if not matches: return s
    for idx, m in enumerate(matches):
        lab = m.group(1)
        content_start = m.end()
        content_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(s)
        chunk = s[content_start:content_end].strip()
        if 'ㄱ' <= lab <= 'ㅎ': items.append(f"{lab}. {chunk}".strip())
        else: items.append(f"{lab} {chunk}".strip())
    out = "\n".join(it for it in items if it)
    out = re.sub(r"\n+", "\n", out).strip()
    return out

# =============================================================================
# 5) Question 모델
# =============================================================================

@dataclass
class Question:
    number: str
    body: str = ""
    view: str = ""
    choices: Dict[str, str] = field(default_factory=dict)
    _images: List[Dict[str, Any]] = field(default_factory=list)
    _forms: List[Dict[str, Any]] = field(default_factory=list)
    _tables: List[Dict[str, Any]] = field(default_factory=list)
    _chunks: List[str] = field(default_factory=list)
    # 라운드 인덱스
    _round: int = 1
    _idx_in_round: int = 1
    _idx_global: int = 1

# =============================================================================
# 6) 본문/보기 분리
# =============================================================================

def _split_question_and_view(text: str) -> Tuple[str, str]:
    if not text: return "", ""
    s = RE_ZERO_WIDTH.sub("", text or "")
    m_bogi = RE_BOGI_TAG_LINE.search(s)
    start_idx = m_bogi.start() if m_bogi else None
    m_bul = RE_VIEW_ENUM_START.search(s)
    if m_bul and (start_idx is None or m_bul.start() < start_idx):
        start_idx = m_bul.start()
    if start_idx is None:
        return s.strip(), ""
    qtxt = s[:start_idx].rstrip()
    vtxt = s[start_idx:].lstrip()
    m_choice = RE_CHOICE_INLINE.search(vtxt)
    if m_choice:
        vtxt = vtxt[: m_choice.start()].rstrip()
    aft = re.sub(r"[ \t]+\n", "\n", vtxt)
    m2 = RE_VIEW_UNIT.search(aft)
    if m2:
        head = aft[:m2.start()].strip()
        packed = _pack_view_with_newlines(aft[m2.start():])
        v_out = (head + ("\n" if head and packed else "") + packed).strip()
    else:
        v_out = aft.strip()
    return qtxt, v_out

# =============================================================================
# 7) 추출기
# =============================================================================

def extract(doc: Any) -> List[Question]:
    if isinstance(doc, dict):
        blocks = doc.get("items") or doc.get("blocks") or []
        if not blocks and "pages" in doc:
            blocks = [b for p in doc["pages"] for b in p.get("blocks", [])]
    else:
        blocks = doc
    blocks = explode_inline_choice_blocks(blocks)
    res: List[Question] = []
    cur: Optional[Question] = None

    def push():
        nonlocal cur
        if not cur: return
        if not cur.choices: cur.choices = {}
        res.append(cur); cur = None

    i, n = 0, len(blocks)
    while i < n:
        b = blocks[i]
        btype = b.get("type")
        text = (b.get("text") or "") if btype == "text" else ""
        if btype == "text":
            head = normalize_text(text).strip()
            m = RE_Q_HEADER.match(head)
            if m:
                new_num = m.group(1)
                if cur is not None and str(cur.number) == new_num:
                    body0 = strip_leading_qnum(text)
                    body0 = insert_formula_placeholders(body0, cur.__dict__)
                    body0 = insert_table_placeholders(body0, cur.__dict__)
                    qtxt, vtxt = _split_question_and_view(body0)
                    qtxt = (qtxt or "").strip()
                    vtxt = re.sub(r"[ \t]+\n", "\n", vtxt or "")
                    vtxt = re.sub(r"\n{3,}", "\n\n", vtxt).strip()
                    if qtxt: cur.body = (cur.body + ("\n" if cur.body else "") + qtxt).strip()
                    if vtxt: cur.view = (cur.view + ("\n" if cur.view and vtxt else "") + vtxt).strip()
                    i += 1; continue
                push()
                cur = Question(number=new_num)
                body0 = strip_leading_qnum(text)
                body0 = insert_formula_placeholders(body0, cur.__dict__)
                body0 = insert_table_placeholders(body0, cur.__dict__)
                qtxt, vtxt = _split_question_and_view(body0)
                cur.body = (qtxt or "").strip()
                if vtxt:
                    vv = re.sub(r"[ \t]+\n", "\n", vtxt)
                    vv = re.sub(r"\n{3,}", "\n\n", vv).strip()
                    cur.view = vv
                i += 1; continue

        if cur is not None:
            j = i
            big_parts: List[str] = []
            while j < n:
                nb = blocks[j]
                nbtype = nb.get("type")
                if nbtype == "text":
                    t = nb.get("text", "")
                    if RE_Q_HEADER.match(normalize_text(t).strip()): break
                    t = insert_formula_placeholders(t, cur.__dict__)
                    t = insert_table_placeholders(t, cur.__dict__)
                    big_parts.append(t)
                elif nbtype in FORMULA_BLOCK_TYPES:
                    latex_txt = _latex_from_block(nb)
                    idx_for_form = len(cur._forms)
                    cur._forms.append({
                        "latex": latex_txt,
                        "image_path": nb.get("image_path") or nb.get("img_path") or "",
                        "bbox": nb.get("bbox") or [],
                        "page_idx": nb.get("page_idx"),
                    })
                    big_parts.append(f"<<FORM_{idx_for_form}>>")
                elif nbtype == "image":
                    idx_for_img = len(cur._images)
                    cur._images.append({
                        "file_name": nb.get("file_name") or nb.get("image_path") or nb.get("img_path") or nb.get("src") or "",
                        "bbox": nb.get("bbox") or [],
                        "page_idx": nb.get("page_idx"),
                        "title": _guess_image_title(blocks, j),
                    })
                    big_parts.append(f"\n<<IMG_{idx_for_img}>>\n")
                elif (nbtype == "table"
                      or nb.get("table_body")
                      or (nb.get("html") and "<table" in str(nb.get("html")).lower())
                      or (nb.get("text") and "<table" in str(nb.get("text")).lower())):
                    html_src = nb.get("table_body") or nb.get("html") or nb.get("text") or ""
                    if html_src:
                        start_pos = 0
                        while True:
                            m_tbl = RE_HTML_TABLE.search(html_src, start_pos)
                            if not m_tbl: break
                            idx_for_tbl = len(cur._tables)
                            caption = " ".join(nb.get("table_caption") or [])
                            title = clean_text(caption or nb.get("title") or nb.get("caption") or "")
                            cur._tables.append({
                                "html": sanitize_table_html(m_tbl.group(0)),
                                "file_name": nb.get("file_name") or nb.get("image_path") or nb.get("table_img_path") or nb.get("img_path") or "",
                                "bbox": nb.get("bbox") or [],
                                "page_idx": nb.get("page_idx"),
                                "title": title,
                            })
                            big_parts.append(f"\n<<TBL_{idx_for_tbl}>>\n")
                            start_pos = m_tbl.end()
                j += 1

            big = "\n".join(big_parts).strip()
            choices, prefix = slice_choices_by_markers_tailonly(big)
            if choices:
                if prefix:
                    to_body, to_view = route_prefix_generically(prefix)
                    if to_body: cur.body = (cur.body + ("\n" if cur.body else "") + to_body).strip()
                    if to_view: cur.view = (cur.view + ("\n" if cur.view else "") + to_view).strip()
                for lab, body in choices:
                    cur.choices[str(lab)] = body
                cur._chunks.append(big)
                i = j; continue
            else:
                if prefix:
                    to_body, to_view = route_prefix_generically(prefix)
                    if to_body: cur.body = (cur.body + ("\n" if cur.body else "") + to_body).strip()
                    if to_view: cur.view = (cur.view + ("\n" if cur.view else "") + to_view).strip()
                if big:
                    big2 = re.sub(r"[ \t]+\n", "\n", big)
                    big2 = re.sub(r"\n{3,}", "\n\n", big2).strip()
                    cur.body += ("\n" if cur.body else "") + big2
                i = j; continue
        i += 1

    push()
    return res

# =============================================================================
# 8) 파일/정답 로딩 & 메타 파싱
# =============================================================================

def load_json_or_jsonl(path: str):
    if path.lower().endswith(".jsonl"):
        items = []
        with open(path, "r", encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                items.append(json.loads(line))
        return items
    else:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

def _strip_known_suffixes(base: str) -> str:
    return re.sub(r"(?:_content_list|_layout|_middle)$", "", base, flags=re.I)

def _parse_meta_from_filename(path_like: str) -> Dict[str, str]:
    name = os.path.basename(path_like)
    name = os.path.splitext(name)[0]
    name = _strip_known_suffixes(name)
    parts = name.split("-")
    meta = {"data_id": "", "year": "", "exam": "", "subject": "", "type": "", "data_title": ""}

    if len(parts) >= 6 and parts[-1].startswith("문제"):
        if re.fullmatch(r"\d{4}", parts[0]): meta["data_id"] = parts[0]
        if re.fullmatch(r"\d{4}", parts[1]): meta["year"] = parts[1]
        meta["type"] = parts[-2]
        meta["subject"] = parts[-3]
        meta["exam"] = "-".join(parts[2:-3]).strip()
    else:
        m = re.match(r"(?P<id>\d{4}).*?(?P<year>\d{4}).*?-문제$", name)
        meta["data_id"] = (m and m.group("id")) or (re.match(r"\d{4}", name).group(0) if re.match(r"\d{4}", name) else "")
        meta["year"] = (m and m.group("year")) or ""
        if "-문제" in name:
            head = name[: name.rfind("-문제")]
            segs = head.split("-")
            if len(segs) >= 4:
                meta["type"] = segs[-1]
                meta["subject"] = segs[-2]
                meta["exam"] = "-".join(segs[2:-2]).strip()

    title_parts = [p for p in [meta["year"], meta["exam"], meta["subject"]] if p]
    meta["data_title"] = "-".join(title_parts)
    return meta

_SUBJECT_CATEGORY_MAP = {"교육": ("시험",)}
_FLAT_SUB_MAIN = sorted(
    [(sub, main) for main, subs in _SUBJECT_CATEGORY_MAP.items() for sub in subs if sub],
    key=lambda x: len(x[0]), reverse=True,
)

def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    s = s.replace("영역", "")
    s = re.sub(r"\s+", "", s)
    return s.lower()

def _guess_category(subject: str) -> Tuple[str, str]:
    if not subject: return ("교육", "대학수학능력시험")
    s = _norm(subject)
    for sub, main in _FLAT_SUB_MAIN:
        if _norm(sub) in s: return (main, sub)
    for main in _SUBJECT_CATEGORY_MAP.keys():
        if _norm(main) in s: return (main, "")
    return ("교육", "대학수학능력시험")

def load_answer_bundle(path: str):
    def _k(s):
        return (s or "").replace(" ", "")

    ans_map: Dict[str, Any] = {}
    exp_map: Dict[str, str] = {}
    meta_top: Dict[str, Any] = {}
    perq_map: Dict[str, Dict[str, Any]] = {}

    if not path or not os.path.exists(path):
        return ans_map, exp_map, meta_top, perq_map

    def _ingest_obj(obj: Dict[str, Any]):
        for k in ["기출연도", "기출 연도", "시험명", "영역", "과목", "시험유형", "세부과목"]:
            if k in obj and obj[k] not in (None, ""):
                meta_top[_k(k)] = obj[k]
        if "source_url" in obj:
            val = obj["source_url"]
            if isinstance(val, str) and val.strip() and not meta_top.get("source_url"):
                meta_top["source_url"] = val.strip()

        raw_ans = obj.get("문제번호_정답") or obj.get("문제번호-정답") or obj.get("문제번호:정답")
        if isinstance(raw_ans, dict):
            for qk, v in raw_ans.items():
                sk = str(qk)
                if isinstance(v, dict):
                    if "정답" in v:
                        ans_map[sk] = v.get("정답")
                    sub = {}
                    for kk in ["난이도", "배점", "정답률", "문제유형"]:
                        if kk in v and v[kk] not in (None, ""):
                            sub[kk] = v[kk]
                    if sub:
                        perq_map.setdefault(sk, {}).update(sub)
                else:
                    ans_map[sk] = v
        elif isinstance(raw_ans, list):
            for item in raw_ans:
                if isinstance(item, dict):
                    for qk, v in item.items():
                        sk = str(qk)
                        if isinstance(v, dict):
                            if "정답" in v:
                                ans_map[sk] = v.get("정답")
                            sub = {}
                            for kk in ["난이도", "배점", "정답률", "문제유형"]:
                                if kk in v and v[kk] not in (None, ""):
                                    sub[kk] = v[kk]
                            if sub:
                                perq_map.setdefault(sk, {}).update(sub)
                        else:
                            ans_map[sk] = v

        raw_exp = obj.get("문제번호_해설") or obj.get("문제번호-해설") or obj.get("문제번호:해설")
        if isinstance(raw_exp, dict):
            for k, v in raw_exp.items():
                if v is not None and str(v).strip():
                    exp_map[str(k)] = str(v)
        elif isinstance(raw_exp, list):
            for item in raw_exp:
                if isinstance(item, dict):
                    for k, v in item.items():
                        if v is not None and str(v).strip():
                            exp_map[str(k)] = str(v)

    lower = path.lower()
    if lower.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if isinstance(obj, dict):
                    _ingest_obj(obj)
    else:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return ans_map, exp_map, meta_top, perq_map

        if isinstance(data, dict):
            _ingest_obj(data)
        elif isinstance(data, list):
            for obj in data:
                if isinstance(obj, dict):
                    _ingest_obj(obj)

    return ans_map, exp_map, meta_top, perq_map

def collect_answer_bundles_auto(dir_hint: str, layout_dir: str) -> Dict[str, Dict[str, Any]]:
    mapping: Dict[str, Dict[str, Any]] = {}
    roots: List[str] = []
    if dir_hint and os.path.isdir(_abs_norm(dir_hint)):
        roots.append(dir_hint)
    ld = _abs_norm(layout_dir)
    ld_parent = os.path.dirname(ld)
    ld_grand = os.path.dirname(ld_parent)
    for r in [ld_parent, ld_grand, os.getcwd()]:
        if r and os.path.isdir(r):
            roots.append(r)

    seen_files: List[str] = []
    for root in roots:
        for pat in ["*-정답.jsonl", "*정답*.jsonl"]:
            found = safe_glob(root, pat, recursive=True)
            seen_files.extend(found)

    files = sorted({f for f in seen_files if f.lower().endswith(".jsonl")})

    for ap in files:
        base = os.path.basename(ap)
        m = re.match(r"^(\d{4})", base)
        if not m:
            continue
        stem = m.group(1)
        a_map, e_map = load_answer_bundle(ap)
        prev = mapping.get(stem)
        if prev:
            if base.endswith("-정답.jsonl") and not os.path.basename(prev["path"]).endswith("-정답.jsonl"):
                mapping[stem] = {"ans": a_map, "exp": e_map, "path": ap}
        else:
            mapping[stem] = {"ans": a_map, "exp": e_map, "path": ap}

    print(f"🔎 정답 파일 탐색 루트: {[ _abs_norm(r) for r in roots ]}")
    print(f"🔎 발견 정답 파일 수: {len(files)}")
    return mapping

def find_answers_by_stem(answer_path: str) -> Dict[str, Dict[str, Any]]:
    """
    ANSWER_DIR가 단일 JSONL인 경우, STEM(앞 4자리 data_id)별로:
      - 통합 맵(ans/exp/perq)
      - 시험유형별 맵(ans_by_type/exp_by_type/perq_by_type)
      - ★ 추가: 과목/세부과목별 맵(by_subject) + (시험유형별×과목) 맵(by_type_subject)
    디렉터리인 경우엔 기존 통합 맵만 제공 (fallback).
    """
    mapping: Dict[str, Dict[str, Any]] = {}
    if not answer_path:
        return mapping

    ap_norm = _abs_norm(answer_path)
    # 1) 파일인 경우: 파일 본문(jsonl/json) 재스캔 → STEM & 시험유형/과목으로 그룹핑
    if os.path.isfile(ap_norm):
        # ★ 추가: stems 구조에 by_subject / by_type_subject 필드 준비
        stems: Dict[str, Dict[str, Any]] = {}
        def _ingest_line_obj(obj: Dict[str, Any]):
            stem = str(obj.get("data_id", ""))[:4]
            if not (stem.isdigit() and len(stem) == 4):
                return
            exam_type = str(obj.get("시험유형", "") or "").strip()  # "홀" / "짝" / 또는 빈문자
            subj = str(obj.get("과목", "") or "").strip()
            subj2 = str(obj.get("세부과목", "") or "").strip()
            subj_key = f"{subj}|{subj2}" if (subj or subj2) else ""

            entry = stems.setdefault(stem, {
                "ans": {},
                "exp": {},
                "meta": {},
                "perq": {},
                "ans_by_type": {},
                "exp_by_type": {},
                "perq_by_type": {},
                # ★ 추가: 과목/세부과목별
                "ans_by_subject": {},
                "exp_by_subject": {},
                "perq_by_subject": {},
                # ★ 추가: (시험유형 → 과목/세부과목 → 맵)
                "ans_by_type_subject": {},
                "exp_by_type_subject": {},
                "perq_by_type_subject": {},
                "path": ap_norm
            })

            # 메타(마지막 값이 덮어씀 가능)
            for k in ["기출연도","시험명","영역","과목","시험유형","세부과목","source_url"]:
                if k in obj and obj[k] not in (None, ""):
                    entry["meta"][k] = obj[k]

            # 타입 맵 준비
            if exam_type:
                entry["ans_by_type"].setdefault(exam_type, {})
                entry["exp_by_type"].setdefault(exam_type, {})
                entry["perq_by_type"].setdefault(exam_type, {})
                # ★ 추가
                if subj_key:
                    entry["ans_by_type_subject"].setdefault(exam_type, {}).setdefault(subj_key, {})
                    entry["exp_by_type_subject"].setdefault(exam_type, {}).setdefault(subj_key, {})
                    entry["perq_by_type_subject"].setdefault(exam_type, {}).setdefault(subj_key, {})

            # 정답/부가정보
            raw_ans = obj.get("문제번호_정답")
            if isinstance(raw_ans, dict):
                for qk, v in raw_ans.items():
                    sk = str(qk)
                    # 통합 맵
                    if isinstance(v, dict) and "정답" in v:
                        entry["ans"][sk] = v.get("정답")
                        sub = {}
                        for kk in ["난이도","배점","정답률","문제유형"]:
                            if kk in v and v[kk] not in (None, ""):
                                sub[kk] = v[kk]
                        if sub:
                            entry["perq"].setdefault(sk, {}).update(sub)
                    else:
                        entry["ans"][sk] = v

                    # 타입별 맵
                    if exam_type:
                        if isinstance(v, dict) and "정답" in v:
                            entry["ans_by_type"][exam_type][sk] = v.get("정답")
                            sub_t = {}
                            for kk in ["난이도","배점","정답률","문제유형"]:
                                if kk in v and v[kk] not in (None, ""):
                                    sub_t[kk] = v[kk]
                            if sub_t:
                                entry["perq_by_type"][exam_type].setdefault(sk, {}).update(sub_t)
                        else:
                            entry["ans_by_type"][exam_type][sk] = v

                        # ★ 추가: (시험유형 × 과목)
                        if subj_key:
                            if isinstance(v, dict) and "정답" in v:
                                entry["ans_by_type_subject"][exam_type][subj_key][sk] = v.get("정답")
                                if sub_t:
                                    entry["perq_by_type_subject"][exam_type][subj_key].setdefault(sk, {}).update(sub_t)
                            else:
                                entry["ans_by_type_subject"][exam_type][subj_key][sk] = v

                    # ★ 추가: 과목/세부과목별(시험유형 무시, 전부 병합)
                    if subj_key:
                        if isinstance(v, dict) and "정답" in v:
                            entry["ans_by_subject"].setdefault(subj_key, {})[sk] = v.get("정답")
                            sub_s = {}
                            for kk in ["난이도","배점","정답률","문제유형"]:
                                if kk in v and v[kk] not in (None, ""):
                                    sub_s[kk] = v[kk]
                            if sub_s:
                                entry["perq_by_subject"].setdefault(subj_key, {}).setdefault(sk, {}).update(sub_s)
                        else:
                            entry["ans_by_subject"].setdefault(subj_key, {})[sk] = v

            # 해설
            raw_exp = obj.get("문제번호_해설")
            if isinstance(raw_exp, dict):
                for k, v in raw_exp.items():
                    if v is not None and str(v).strip():
                        entry["exp"][str(k)] = str(v)
                        if exam_type:
                            entry["exp_by_type"][exam_type][str(k)] = str(v)
                            if subj_key:
                                entry["exp_by_type_subject"][exam_type][subj_key][str(k)] = str(v)
                        if subj_key:
                            entry["exp_by_subject"].setdefault(subj_key, {})[str(k)] = str(v)

        lower = ap_norm.lower()
        try:
            if lower.endswith(".jsonl"):
                with open(ap_norm, "r", encoding="utf-8-sig") as f:
                    for line in f:
                        s = line.strip()
                        if not s: continue
                        try:
                            obj = json.loads(s)
                            if isinstance(obj, dict):
                                _ingest_line_obj(obj)
                        except Exception:
                            pass
            else:
                with open(ap_norm, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    for obj in data:
                        if isinstance(obj, dict):
                            _ingest_line_obj(obj)
                elif isinstance(data, dict):
                    _ingest_line_obj(data)
        except Exception:
            return mapping

        for stem, bundle in stems.items():
            mapping[stem] = bundle
        return mapping

    # 2) 디렉터리: 기존 방식(fallback, 타입별/과목별 맵은 없음)
    if not os.path.isdir(ap_norm):
        return mapping

    def _priority(name_nfk: str) -> int:
        if name_nfk.endswith("-정답.jsonl"): return 0
        if name_nfk.endswith("-정답.json"):  return 1
        if "정답" in name_nfk and name_nfk.endswith(".jsonl"): return 2
        if "정답" in name_nfk and name_nfk.endswith(".json"):  return 3
        return 9

    try:
        names = os.listdir(ap_norm)
    except Exception:
        return mapping

    for fn in names:
        norm = unicodedata.normalize("NFKC", fn)
        lower = norm.lower()
        if not (lower.endswith(".jsonl") or lower.endswith(".json")):
            continue
        if "정답" not in norm:
            continue
        m = re.match(r"^(\d{4})", norm)
        if not m:
            continue
        stem = m.group(1)
        ap = os.path.join(ap_norm, fn)
        ans_map, exp_map, meta_top, perq_map = load_answer_bundle(ap)
        prev = mapping.get(stem)
        cand = {
            "ans": ans_map,
            "exp": exp_map,
            "meta": meta_top,
            "perq": perq_map,
            "path": ap
        }
        if (not prev) or (_priority(norm) < _priority(os.path.basename(prev["path"]))):
            mapping[stem] = cand
    return mapping

# =============================================================================
# 9) 자원 로그/도우미
# =============================================================================

_IMAGE_LOGS: Dict[str, Dict[str, Any]] = {}

def _bbox_dict(bbox_list):
    b = bbox_list or []
    return {"x1": b[0] if len(b) >= 1 else None,
            "y1": b[1] if len(b) >= 2 else None,
            "x2": b[2] if len(b) >= 3 else None,
            "y2": b[3] if len(b) >= 4 else None}

# =============================================================================
# 10) 라운드 규칙/출력
# =============================================================================

def _choice_order_key(k: str) -> int:
    ks = str(k)
    num = UC2NUM.get(ks)
    if num and num.isdigit(): return int(num)
    if ks.isdigit(): return int(ks)
    return 9999

def _make_content_from_addinfo(ai: Dict[str, Any]) -> str:
    body = ai.get("문제본문", "")
    view = ai.get("문제보기", "") or ""
    choices = ai.get("선택지") or ""
    answer_text = ai.get("정답", "") or ""
    explain_text = ai.get("해설", "") or ""

    lines = [f"[문제]: {body}".rstrip()]
    if view:
        lines[-1] += ("\n" + view)
    if isinstance(choices, dict) and choices:
        for k in sorted(choices.keys(), key=_choice_order_key):
            v = choices[k] or ""
            lines.append(f"{k}: {v}")
    lines += ["", f"[정답]: {answer_text}"]
    if explain_text:
        lines += ["", explain_text]
    return "\n".join(lines).strip()

def _choice_key_num_map(choice_keys) -> Dict[str, str]:
    mp: Dict[str, str] = {}
    for k in choice_keys:
        ch = str(k)
        num = UC2NUM.get(ch)
        if num and num not in mp:
            mp[num] = ch
    return mp

def _to_circled_string(val: Any, choice_keys) -> str:
    s = str(val)
    if re.search(RE_CIRCLED_NUM, s):
        return s
    num_map = _choice_key_num_map(choice_keys)
    has_circled_in_choices = bool(num_map)
    if not has_circled_in_choices:
        return s
    def repl(m: re.Match) -> str:
        num = m.group(1)
        return num_map.get(num, NUM2UC.get(num, num))
    return re.sub(r'(?<!\d)(\d{1,2})(?!\d)', repl, s)

def _size_or_default(sz: Optional[Dict[str, Any]]) -> Dict[str, Optional[int]]:
    ch = ht = wd = None

    def from_tuple(tp):
        nonlocal ch, ht, wd
        if not isinstance(tp, (list, tuple)):
            return
        if len(tp) == 2:
            # (W,H) 또는 (H,W) — 관례상 PIL size는 (W,H).
            # W가 H보다 큰 경우가 흔하므로 먼저 (W,H)로 가정:
            w, h = tp[0], tp[1]
            # 그래도 뒤집어야 할 근거가 없으니 그대로 둠
            wd, ht = int(w), int(h)
        elif len(tp) == 3:
            a, b, c = tp[0], tp[1], tp[2]
            # 채널 수는 보통 1~4, 높이/너비는 보통 수백 이상 → 휴리스틱
            if isinstance(a, int) and a <= 4:
                # (C,H,W)
                ch, ht, wd = int(a), int(b), int(c)
            elif isinstance(c, int) and c <= 4:
                # (H,W,C)
                ht, wd, ch = int(a), int(b), int(c)
            else:
                # 애매하면 (H,W,C)로 가정
                ht, wd, ch = int(a), int(b), int(c)

    if isinstance(sz, dict):
        ch = sz.get("channel") or sz.get("channels") or sz.get("c") or sz.get("depth")
        ht = sz.get("height") or sz.get("h") or sz.get("rows")
        wd = sz.get("width")  or sz.get("w") or sz.get("cols")

        # shape/size 보조 키 처리
        if (ch is None or ht is None or wd is None):
            shp = sz.get("shape") or sz.get("size")
            if isinstance(shp, (list, tuple)):
                from_tuple(shp)

    elif isinstance(sz, (list, tuple)):
        from_tuple(sz)
    else:
        # numpy-like 객체가 올 가능성 (유틸 내부 구현에 따라)
        try:
            shp = getattr(sz, "shape", None) or getattr(sz, "size", None)
            if isinstance(shp, (list, tuple)):
                from_tuple(shp)
        except Exception:
            pass

    return {"channel": int(ch) if ch is not None else None,
            "height": int(ht) if ht is not None else None,
            "width":  int(wd) if wd is not None else None}

# ===== 라운드 적용 여부 (파일명 기준)
def _should_use_rounds(meta_src_path: str) -> bool:
    """
    파일 이름에 대해:
    - '국어' 또는 '수학' 포함
    - '홀' / '짝' 포함 X
    - '문제' 포함 O
    => True (라운드 분리 적용)
    그 외 => False (라운드 미적용)
    """
    base = os.path.basename(meta_src_path)
    name, _ = os.path.splitext(base)
    cond_subj = ("국어" in name) or ("수학" in name)
    has_holjjak = ("홀" in name) or ("짝" in name)
    has_munje = ("문제" in name)
    return cond_subj and (not has_holjjak) and has_munje

# ===== 인덱스/라운드 부여
def _assign_indices(questions: List[Question], use_rounds: bool) -> None:
    round_no = 1
    idx_in_round = 0
    idx_global = 0
    first = True
    for q in questions:
        idx_global += 1
        if use_rounds:
            try:
                n = int(str(q.number).strip())
            except Exception:
                n = None
            if not first and n == 1:
                round_no += 1
                idx_in_round = 0
        else:
            round_no = 1
        first = False
        idx_in_round += 1
        q._round = round_no
        q._idx_in_round = idx_in_round
        q._idx_global = idx_global

def build_kt_jsonl(
    questions: List[Question],
    meta_src_path: str,
    out_jsonl_path: str,
    answer_map: Optional[Dict[str, Any]] = None,
    explain_map: Optional[Dict[str, str]] = None,
    img_cfg: ImageConfig = None,
    answer_meta: Optional[Dict[str, Any]] = None,
    perq_detail: Optional[Dict[str, Dict[str, Any]]] = None,
    # 추가: 시험유형별 매핑
    answer_by_type: Optional[Dict[str, Dict[str, Any]]] = None,
    explain_by_type: Optional[Dict[str, Dict[str, str]]] = None,
    perq_by_type: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None,
    # ★ 추가: 과목/세부과목별(시험유형 무시)
    answer_by_subject: Optional[Dict[str, Dict[str, Any]]] = None,
    explain_by_subject: Optional[Dict[str, Dict[str, str]]] = None,
    perq_by_subject: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None,
    # ★ 추가: (시험유형 → 과목/세부과목 → 맵)
    answer_by_type_subject: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None,
    explain_by_type_subject: Optional[Dict[str, Dict[str, Dict[str, str]]]] = None,
    perq_by_type_subject: Optional[Dict[str, Dict[str, Dict[str, Dict[str, Any]]]]] = None,
):
    # ---- 라운드 적용 여부 결정 + 인덱스 부여
    use_rounds = _should_use_rounds(meta_src_path)
    _assign_indices(questions, use_rounds)

    # --- 파일명으로 파싱한 기본 메타
    meta = _parse_meta_from_filename(meta_src_path)
    base_id = meta.get("data_id", "")
    fallback_year    = meta.get("year", "")
    fallback_exam    = meta.get("exam", "")
    fallback_subject = meta.get("subject", "")
    fallback_examtype = meta.get("type", "")
    data_title = meta.get("data_title", "")
    collected_date = datetime.now().strftime("%Y.%m.%d")

    # --- 정답 메타 우선
    def _m(meta_obj, key):
        if not meta_obj: return ""
        return meta_obj.get(key, "") or ""

    a_year     = _m(answer_meta, "기출연도") or fallback_year
    a_exam     = _m(answer_meta, "시험명")   or fallback_exam
    a_area     = _m(answer_meta, "영역")     or ""
    a_subject  = _m(answer_meta, "과목")     or fallback_subject
    a_type     = _m(answer_meta, "시험유형") or ""
    a_subj2    = _m(answer_meta, "세부과목") or ""

    data_title = "-".join([s for s in [a_year, a_area, a_subj2] if s]) \
                 or "-".join([s for s in [fallback_year, "", fallback_subject] if s])

    asset_seq = 1
    max_choice_in_file = max((len(q.choices or {}) for q in questions), default=0)
    uniform_question_type = f"{max_choice_in_file}지선다" if max_choice_in_file > 0 else "서술형"

    def _lookup(map_obj, qnum: str):
        if not map_obj: return None
        cand = [qnum, qnum.lstrip("0") or qnum]
        if qnum.isdigit(): cand.append(int(qnum))
        for c in cand:
            if c in map_obj: return map_obj[c]
        return None

    def _sorted_choice_keys(keys):
        return sorted(keys, key=_choice_order_key)

    # round -> 시험유형 키 변환 (라운드 적용 파일에서만 의미)
    def _round_to_examtype(r: int) -> Optional[str]:
        return "홀" if r == 1 else ("짝" if r == 2 else None)

    # ★ 추가: 과목/세부과목 선택 유틸
    def _best_subject_key_for_range(
        needed_nums: List[int],
        exam_type_key: Optional[str],
        prefer_keys: List[str],  # 우선 후보 (예: ["국어|국어"] / ["수학|수학"] 등)
        fallback_any: bool = True
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]], Optional[Dict[str, str]], Optional[Dict[str, Dict[str, Any]]]]:
        """
        exam_type_key가 주어지면 (시험유형→과목) 맵에서 우선 탐색,
        없거나 비면 과목 전역 맵에서 최적(필요 번호 커버 수 최대)을 선택.
        prefer_keys에 명시된 키를 최우선으로 고려. 없으면 아무거나(coverage 최대).
        """
        needed_set = set(str(n) for n in needed_nums)
        # 1) type+subject
        best = (None, None, None, None, -1)
        if exam_type_key and answer_by_type_subject and exam_type_key in (answer_by_type_subject or {}):
            type_bucket_a = answer_by_type_subject.get(exam_type_key, {})
            type_bucket_e = (explain_by_type_subject or {}).get(exam_type_key, {})
            type_bucket_p = (perq_by_type_subject or {}).get(exam_type_key, {})
            cand_keys = list(type_bucket_a.keys())
            # prefer_keys 먼저
            ordered = [k for k in prefer_keys if k in cand_keys] + [k for k in cand_keys if k not in prefer_keys]
            for k in ordered:
                amap = type_bucket_a.get(k, {})
                cover = len(needed_set & set(amap.keys()))
                if cover > best[4]:
                    best = (k, amap, type_bucket_e.get(k, {}), type_bucket_p.get(k, {}), cover)
            if best[1] is not None and best[4] > 0:
                return best[0], best[1], best[2], best[3]
        # 2) subject only
        if answer_by_subject:
            cand_keys = list(answer_by_subject.keys())
            ordered = [k for k in prefer_keys if k in cand_keys] + [k for k in cand_keys if k not in prefer_keys]
            for k in ordered:
                amap = answer_by_subject.get(k, {})
                cover = len(needed_set & set(amap.keys()))
                if cover > best[4]:
                    best = (k, amap, (explain_by_subject or {}).get(k, {}), (perq_by_subject or {}).get(k, {}), cover)
            if best[1] is not None and (best[4] > 0 or fallback_any):
                return best[0], best[1], best[2], best[3]
        return None, None, None, None

    # ★ 추가: 국어/수학 구간 트래커
    is_korean_file = ("국어" in (fallback_subject or "")) or ("국어" in (a_subject or ""))
    is_math_file   = ("수학" in (fallback_subject or "")) or ("수학" in (a_subject or ""))

    korean_35_45_block_count = 0   # 0: 아직 없음, 1: 첫 블록(화작), 2: 두번째 블록(언매)
    math_23_30_block_count   = 0   # 1: 확통, 2: 미적분, 3: 기하
    prev_qnum_int: Optional[int] = None

    with open(out_jsonl_path, "w", encoding="utf-8") as fw:
        for q in questions:
            body_pre = q.body or ""
            view_pre = q.view or ""
            choices_pre: Dict[str, str] = q.choices or {}

            # ==== ID 체계 ====
            base_id_with_global = f"{base_id}_{q._idx_global:04d}"

            try:
                qnum_int = int(str(q.number).strip())
            except Exception:
                qnum_int = q._idx_in_round  # 폴백

            if use_rounds:
                q_data_id = f"{base_id}_{q._round:04d}_{qnum_int:04d}"
            else:
                q_data_id = f"{base_id}_{qnum_int:04d}"

            combined_text = "\n".join(
                [body_pre, view_pre] + [choices_pre[k] for k in _sorted_choice_keys(choices_pre.keys())]
            )

            ph_order: List[Tuple[str, int]] = []
            seen = set()
            for m in re.finditer(r"<<(IMG|FORM|TBL)_(\d+)>>", combined_text):
                key = (m.group(1), int(m.group(2)))
                if key not in seen:
                    ph_order.append(key); seen.add(key)

            content_meta: Dict[str, Any] = {}
            ph_to_tag: Dict[Tuple[str, int], str] = {}
            qnum = str(q.number).strip()

            # --- 플레이스홀더 변환
            for kind, idx in ph_order:
                tag = f"tag_{base_id_with_global}_{asset_seq:04d}"
                ph_to_tag[(kind, idx)] = tag

                if kind == "IMG":
                    img = (q._images or [])[idx] if idx < len(q._images or []) else {}
                    bbox = img.get("bbox") or []
                    src_hint = (img.get("file_name")
                                or img.get("image_path")
                                or img.get("img_path")
                                or img.get("src")
                                or "")
                    res = classify_and_store_image(src_hint, base_id_with_global or "0000", tag=tag, kind="img", cfg=img_cfg)
                    if isinstance(res, tuple) and len(res) == 3:
                        new_path, sz, src_abs = res
                    else:
                        new_path, sz = res; src_abs = ""
                    info = {
                        "type": "image",
                        "bbox": _bbox_dict(bbox),
                        "title": f"{q_data_id}_그림",
                        "file_name": new_path or src_hint or "",
                        "img_size": _size_or_default(sz),
                    }
                    content_meta[tag] = info

                    if new_path:
                        src_key = os.path.basename(src_abs) if src_abs else os.path.basename(new_path)
                        _IMAGE_LOGS.setdefault(base_id or "0000", {})[src_key] = {
                            "src": src_abs or "", "dst": new_path, "size": sz, "kind": "img", "tag": tag,
                        }

                elif kind == "FORM":
                    form = (q._forms or [])[idx] if idx < len(q._forms or []) else {}
                    latex_with_delims = form.get("latex", "") or ""
                    bbox = form.get("bbox") or []
                    src_hint = form.get("image_path") or form.get("img_path") or form.get("file_name") or ""
                    new_path = ""; sz = {}; src_abs = ""
                    if src_hint:
                        res = classify_and_store_image(src_hint, base_id_with_global or "0000", tag=tag, kind="form", cfg=img_cfg)
                        if isinstance(res, tuple) and len(res) == 3:
                            new_path, sz, src_abs = res
                        else:
                            new_path, sz = res; src_abs = ""
                    info = {
                        "type": "formula",
                        "bbox": _bbox_dict(bbox),
                        "text": _strip_math_delims(latex_with_delims),
                        "info": "latex",
                        "file_name": new_path or src_hint or "",
                        "img_size": _size_or_default(sz),
                    }
                    content_meta[tag] = info

                    if new_path:
                        src_key = os.path.basename(src_abs) if src_abs else os.path.basename(new_path)
                        _IMAGE_LOGS.setdefault(base_id or "0000", {})[src_key] = {
                            "src": src_abs or "", "dst": new_path, "size": sz, "kind": "form", "tag": tag,
                        }

                elif kind == "TBL":
                    tbl = (q._tables or [])[idx] if idx < len(q._tables or []) else {}
                    bbox = tbl.get("bbox") or []
                    src_hint = (tbl.get("file_name")
                                or tbl.get("image_path")
                                or tbl.get("table_img_path")
                                or tbl.get("img_path")
                                or "")
                    res = classify_and_store_image(src_hint, base_id_with_global or "0000", tag=tag, kind="tbl", cfg=img_cfg)
                    if isinstance(res, tuple) and len(res) == 3:
                        new_path, sz, src_abs = res
                    else:
                        new_path, sz = res; src_abs = ""
                    info = {
                        "type": "table",
                        "bbox": _bbox_dict(bbox),
                        "text": sanitize_table_html(tbl.get("html", "") or ""),
                        "info": "html",
                        "file_name": new_path or src_hint or "",
                        "img_size": _size_or_default(sz),
                    }
                    content_meta[tag] = info

                    if new_path:
                        src_key = os.path.basename(src_abs) if src_abs else os.path.basename(new_path)
                        _IMAGE_LOGS.setdefault(base_id or "0000", {})[src_key] = {
                            "src": src_abs or "", "dst": new_path, "size": sz, "kind": "tbl", "tag": tag,
                        }

                asset_seq += 1

            def replace_ph(txt: str) -> str:
                if not txt: return txt
                def repl(m: re.Match) -> str:
                    k = (m.group(1), int(m.group(2)))
                    tag = ph_to_tag.get(k, "")
                    return f"<{tag}>" if tag else ""
                return re.sub(r"<<(IMG|FORM|TBL)_(\d+)>>", repl, txt)

            replaced_body = replace_ph(body_pre)
            replaced_view = replace_ph(view_pre)
            replaced_choices = {str(k): replace_ph(v) for k, v in choices_pre.items()}
            choices_for_addinfo = {k: clean_text(v or "") for k, v in replaced_choices.items()}

            dtype_set = set(["text"])
            for v in content_meta.values():
                t = v.get("type")
                if t in ("image", "formula", "table"): dtype_set.add(t)
            data_types = sorted(dtype_set)

            # ====== 기본 맵 ======
            ans_maps = answer_map or {}
            exp_maps = explain_map or {}
            perq_maps = perq_detail or {}

            # ====== (기존) 시험유형 매핑 우선 적용 ======
            if use_rounds:
                key = _round_to_examtype(q._round)  # 1→홀, 2→짝
                if key and answer_by_type and key in answer_by_type and answer_by_type[key]:
                    ans_maps = answer_by_type[key]
                if key and explain_by_type and key in explain_by_type and explain_by_type[key]:
                    exp_maps = explain_by_type[key]
                if key and perq_by_type and key in perq_by_type and perq_by_type[key]:
                    perq_maps = perq_by_type[key]

            # ====== ★ 추가 규칙: 국어/수학 파일의 구간별 과목/세부과목 매핑 ======
            #   - 국어:
            #     1~34: 같은 id 아무 과목에서 가져오되, 과목/세부과목 = "국어"
            #     첫 35~45 블록: "화법과작문"
            #     다음 35~45 블록: "언어와매체"
            #   - 수학:
            #     1~22: 같은 id 아무 과목에서 가져오되, 과목/세부과목 = "수학"
            #     첫 23~30 블록: "확률과통계"
            #     다음 블록: "미적분"
            #     다음 블록: "기하"

            override_subject = None  # (과목, 세부과목) 오버라이드 텍스트
            exam_type_key = _round_to_examtype(q._round) if use_rounds else None

            # 블록 경계 탐지 (35→... 또는 23→...)
            if prev_qnum_int is not None:
                # 새 국어 블록 시작(35) 체크
                if is_korean_file and qnum_int == 35 and (prev_qnum_int is None or prev_qnum_int != 35):
                    # 직전이 45였던 케이스가 일반적이지만, 안전하게 35 재등장 시 카운트++ 하자
                    korean_35_45_block_count += 1
                # 새 수학 블록 시작(23)
                if is_math_file and qnum_int == 23 and (prev_qnum_int is None or prev_qnum_int != 23):
                    math_23_30_block_count += 1
            else:
                # 첫 문제에서 바로 35/23으로 시작하는 특이 케이스도 방어
                if is_korean_file and qnum_int == 35:
                    korean_35_45_block_count = 1
                if is_math_file and qnum_int == 23:
                    math_23_30_block_count = 1

            # 국어 규칙
            if is_korean_file:
                if 1 <= qnum_int <= 34:
                    # 과목/세부과목 오버라이드
                    override_subject = ("국어", "국어")
                    # 같은 id 아무거나 → coverage가 가장 큰 과목 선택
                    need = list(range(1, 35))
                    subj_key, a_map2, e_map2, p_map2 = _best_subject_key_for_range(
                        need, exam_type_key, prefer_keys=["화법과작문|화법과작문","언어와매체|언어와매체"], fallback_any=True
                    )
                    if a_map2:
                        ans_maps, exp_maps, perq_maps = a_map2, (e_map2 or {}), (p_map2 or {})
                elif 35 <= qnum_int <= 45:
                    if korean_35_45_block_count <= 1:
                        # 첫 35~45 블록 → 화법과작문
                        override_subject = ("화법과작문", "화법과작문")
                        need = list(range(35, 46))
                        subj_key, a_map2, e_map2, p_map2 = _best_subject_key_for_range(
                            need, exam_type_key, prefer_keys=["화법과작문|화법과작문"], fallback_any=False
                        )
                        if a_map2:
                            ans_maps, exp_maps, perq_maps = a_map2, (e_map2 or {}), (p_map2 or {})
                    else:
                        # 두번째 35~45 블록 → 언어와매체
                        override_subject = ("언어와매체", "언어와매체")
                        need = list(range(35, 46))
                        subj_key, a_map2, e_map2, p_map2 = _best_subject_key_for_range(
                            need, exam_type_key, prefer_keys=["언어와매체|언어와매체"], fallback_any=False
                        )
                        if a_map2:
                            ans_maps, exp_maps, perq_maps = a_map2, (e_map2 or {}), (p_map2 or {})

            # 수학 규칙
            if is_math_file:
                if 1 <= qnum_int <= 22:
                    override_subject = ("수학", "수학")
                    need = list(range(1, 23))
                    # 아무거나 중 coverage 최대(확통/미적/기하 중 하나)
                    subj_key, a_map2, e_map2, p_map2 = _best_subject_key_for_range(
                        need, exam_type_key,
                        prefer_keys=["수학|확률과통계","수학|미적분","수학|기하"],  # 실제 데이터의 key는 "과목|세부과목"
                        fallback_any=True
                    )
                    if a_map2:
                        ans_maps, exp_maps, perq_maps = a_map2, (e_map2 or {}), (p_map2 or {})
                elif 23 <= qnum_int <= 30:
                    # 블록 순서: 1=확통, 2=미적분, 3=기하
                    if math_23_30_block_count <= 1:
                        override_subject = ("수학", "확률과통계")
                        want = "수학|확률과통계"
                    elif math_23_30_block_count == 2:
                        override_subject = ("수학", "미적분")
                        want = "수학|미적분"
                    else:
                        override_subject = ("수학", "기하")
                        want = "수학|기하"
                    need = list(range(23, 31))
                    subj_key, a_map2, e_map2, p_map2 = _best_subject_key_for_range(
                        need, exam_type_key, prefer_keys=[want], fallback_any=False
                    )
                    if a_map2:
                        ans_maps, exp_maps, perq_maps = a_map2, (e_map2 or {}), (p_map2 or {})

            # ====== 정답/해설/부가 선택 후 텍스트 구성 ======
            ans_val = _lookup(ans_maps, qnum)

            def _ans_text(v, choice_keys) -> str:
                if v is None:
                    return ""
                if isinstance(v, dict):
                    inner = v.get("정답", v)
                    if isinstance(inner, (list, tuple)):
                        return ",".join(_to_circled_string(x, choice_keys) for x in inner)
                    return _to_circled_string(inner, choice_keys)
                if isinstance(v, (list, tuple)):
                    return ",".join(_to_circled_string(x, choice_keys) for x in v)
                return _to_circled_string(v, choice_keys)

            answer_text = _ans_text(ans_val, replaced_choices.keys())

            explain_text = ""
            ev = _lookup(exp_maps, qnum)
            if ev is not None:
                explain_text = str(ev).strip()

            pdet = (perq_maps or {}).get(qnum, {}) if perq_maps else {}
            diff  = pdet.get("난이도", "")
            score = pdet.get("배점", "")
            rate  = pdet.get("정답률", "")
            qtype = uniform_question_type

            # ★ 추가: 과목/세부과목 오버라이드 적용
            out_subject = a_subject
            out_subj2   = a_subj2
            if override_subject:
                out_subject, out_subj2 = override_subject
            
            # ▶ 문항별 data_title 재계산 (세부과목 우선)
            data_title_q = "-".join([s for s in [a_year, a_area, (out_subj2 or out_subject or fallback_subject)] if s]) \
                        or "-".join([s for s in [fallback_year, "", fallback_subject] if s])

            ai = {
                "문제번호": qnum,
                "기출연도": a_year,
                "시험명": a_exam,
                "영역": a_area,
                "과목": out_subject,
                **({"시험유형": _round_to_examtype(q._round)} if use_rounds else {"시험유형": a_type} if a_type else {}),
                **({"세부과목": out_subj2} if out_subj2 else {}),
                **({"문제유형": qtype} if qtype != "" else {}),
                "문제본문": replaced_body,
                **({"문제보기": replaced_view} if replaced_view else {}),
                "선택지": choices_for_addinfo,
                "정답": answer_text,
                **({"해설": explain_text} if explain_text else {}),
                **({"난이도": diff} if diff != "" else {}),
                **({"배점": score} if score != "" else {}),
                **({"정답률": rate} if rate != "" else {}),
            }

            content = _make_content_from_addinfo(ai)

            rec = {
                "data_id": q_data_id,
                "data_file": os.path.basename(out_jsonl_path),
                "data_title": data_title_q,
            }

            if answer_meta and "source_url" in answer_meta and answer_meta["source_url"]:
                rec["source_url"] = answer_meta["source_url"]

            rec.update({
                "category_main": _guess_category(out_subject)[0],  # ★ category는 오버라이드된 과목 기준
                **({"category_sub": _guess_category(out_subject)[1]} if _guess_category(out_subject)[1] else {}),
                "data_type": data_types,
                "collected_date": collected_date,
                "content": content,
                **({"content_meta": content_meta} if content_meta else {}),
                "add_info": ai,
            })

            fw.write(json.dumps(rec, ensure_ascii=False) + "\n")

            prev_qnum_int = qnum_int  # ★ 다음 루프를 위해 업데이트

# =============================================================================
# 11) 메인 — LAYOUT_DIR 내의 모든 .json 처리
# =============================================================================

def main():
    ensure_dir(FORMAT_DIR)
    ensure_dir(IMAGES_IMG_DIR)
    ensure_dir(IMAGES_TBL_DIR)
    ensure_dir(IMAGES_FORM_DIR)
    ensure_dir(IMAGES_INDEX_DIR)

    IMG_CFG = ImageConfig(
        src_root=IMAGES_SRC_DIR,
        dst_img_root=IMAGES_IMG_DIR,
        dst_tbl_root=IMAGES_TBL_DIR,
        dst_form_root=IMAGES_FORM_DIR,
        move=IMAGE_MOVE,
        overwrite=IMAGE_OVERWRITE,
    )

    answers_by_stem = find_answers_by_stem(ANSWER_DIR)

    layout_paths = sorted(glob.glob(os.path.join(_abs_norm(LAYOUT_DIR), "*.json")))
    if not layout_paths:
        print(f"⚠️ 입력(.json) 파일이 없습니다: {LAYOUT_DIR}")

    total_q = 0
    total_no_choice = 0
    total_choice_parse_error = 0

    for LAYOUT_JSON in layout_paths:
        base = os.path.basename(LAYOUT_JSON)
        m = re.match(r"^(\d{4})", base)
        if not m:
            print(f"⏭️ 스킵(앞 4자리 숫자 없음): {base}")
            continue

        STEM = m.group(1)
        format_name = os.path.splitext(base)[0].removesuffix("_mathpix_merge")
        FORMAT_JSONL = os.path.join(FORMAT_DIR, f"{format_name}.jsonl")

        a_entry = answers_by_stem.get(STEM, {})
        answer_map = a_entry.get("ans") or {}
        explain_map = a_entry.get("exp") or {}
        answer_meta = a_entry.get("meta") or {}
        perq_detail = a_entry.get("perq") or {}
        ans_path = a_entry.get("path")
        # 추가: 시험유형별 맵
        answer_by_type = a_entry.get("ans_by_type") or {}
        explain_by_type = a_entry.get("exp_by_type") or {}
        perq_by_type = a_entry.get("perq_by_type") or {}
        # ★ 추가: 과목/세부과목별 & (유형×과목) 맵
        answer_by_subject = a_entry.get("ans_by_subject") or {}
        explain_by_subject = a_entry.get("exp_by_subject") or {}
        perq_by_subject = a_entry.get("perq_by_subject") or {}
        answer_by_type_subject = a_entry.get("ans_by_type_subject") or {}
        explain_by_type_subject = a_entry.get("exp_by_type_subject") or {}
        perq_by_type_subject = a_entry.get("perq_by_type_subject") or {}

        try:
            raw = load_json_or_jsonl(LAYOUT_JSON)
            data = extract(raw)

            max_choice_in_file = max((len(q.choices or {}) for q in data), default=0)
            expected_max = min(max_choice_in_file, 5)  # 최대 5지선다만 인정
            choice_parse_errors = []
            if expected_max > 0:
                for q in data:
                    n_choices = len(q.choices or {})
                    # 5개 초과면 무조건 파싱 오류, 기대 개수보다 적어도 오류
                    if n_choices > 5 or n_choices < expected_max:
                        choice_parse_errors.append(q.number)

            total_q += len(data)
            total_choice_parse_error += len(choice_parse_errors)

            build_kt_jsonl(
                data, LAYOUT_JSON, FORMAT_JSONL,
                answer_map=answer_map,
                explain_map=explain_map,
                img_cfg=IMG_CFG,
                answer_meta=answer_meta,
                perq_detail=perq_detail,
                # 전달: 타입별 맵
                answer_by_type=answer_by_type,
                explain_by_type=explain_by_type,
                perq_by_type=perq_by_type,
                # ★ 전달: 과목/세부과목별 & 유형×과목 맵
                answer_by_subject=answer_by_subject,
                explain_by_subject=explain_by_subject,
                perq_by_subject=perq_by_subject,
                answer_by_type_subject=answer_by_type_subject,
                explain_by_type_subject=explain_by_type_subject,
                perq_by_type_subject=perq_by_type_subject,
            )

            if STEM in _IMAGE_LOGS:
                idx_path = os.path.join(IMAGES_INDEX_DIR, f"{STEM}.json")
                ensure_dir(os.path.dirname(idx_path))
                with open(idx_path, "w", encoding="utf-8") as f:
                    json.dump(_IMAGE_LOGS[STEM], f, ensure_ascii=False, indent=2)

            no_choice = [q.number for q in data if not q.choices]
            total_no_choice += len(no_choice)

            print(f"\n📄 처리 대상: {base}")
            print(f"   STEM: {STEM}")
            print(f"   ✅ 입력: {LAYOUT_JSON}")
            if ans_path:
                # 타입별 요약 출력
                type_keys = ", ".join(sorted((answer_by_type or {}).keys()))
                print(f"   ✅ 적용(정답/해설): {ans_path} | 정답 {len(answer_map)}개, 해설 {len(explain_map)}개 | 시험유형 분포: [{type_keys}]")
            else:
                print(f"   ⚠️ 적용할 정답 파일 없음 (STEM={STEM})")
            print(f"   ✅ 저장(KT): {FORMAT_JSONL}")
            if max_choice_in_file > 0:
                print(f"   ✅ 기대 보기 수: {min(max_choice_in_file, 5)}지선다")
                print(f"   ⚠️ 선택지 파싱 오류: {len(choice_parse_errors)} -> {choice_parse_errors}")
            print(f"   ✅ 문항수: {len(data)} / 선택지 없는 문항: {len(no_choice)} -> {no_choice}")
            print(f"   📦 이미지 인덱스: {os.path.join(IMAGES_INDEX_DIR, f'{STEM}.json')}")
        except Exception as e:
            print(f"❌ 오류: {base} 처리 중 예외 발생: {e}")

    print(f"\n🎯 총 처리 파일: {len(layout_paths)} / 총 문항수: {total_q} / 선택지 없는 문항 수: {total_no_choice}")

if __name__ == "__main__":
    main()
