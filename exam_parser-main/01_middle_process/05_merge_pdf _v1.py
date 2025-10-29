# -*- coding: utf-8 -*-
from pathlib import Path
import json, re, unicodedata
from bs4 import BeautifulSoup
from difflib import SequenceMatcher
from typing import Optional, Tuple
from bisect import bisect_left

# ======================================================
# 폴더 경로 (여기만 네 환경에 맞게 수정)
# ======================================================
XHTML_DIR_IN = Path(r"output_html")                        # 각 문제 폴더에 index.xhtml 존재
JSON_DIR_IN  = Path(r"exam_parser-main\01_middle_process\data\choice_process")  # 병합 대상 JSON 폴더(재귀)
OUT_DIR      = Path(r"exam_parser-main\01_middle_process\data\merge_html")                   # 출력 루트

# JSON ↔ XHTML 이름 유사도 매칭 임계값
NAME_SIM_THRESHOLD = 0.66

# ======================================================
# ===== 설정 (원본 그대로) =====
# ======================================================
SIM_THRESHOLD = 0.85     # 유사도 임계값(0~1)
MIN_ANCHOR = 8           # 앵커 조각 최소 길이(정확 매칭 실패 시 사용)
MIN_LEN_FOR_ANCHOR = MIN_ANCHOR * 2 + 4  # 너무 짧은 문자열은 앵커 폴백 생략

# 앵커-윈도우 탐색 파라미터(성능/안정성)
ANCHOR_PAD_MIN   = 80          # 창 최소 여유폭(문자 수)
ANCHOR_PAD_FACTOR= 1.3         # 원문 길이 대비 여유 비율
ANCHOR_MAX_WINDOW= 1200        # 창 최대 길이 상한
LENGTH_TOL_BASE  = 10          # 비교 길이 여유(±)
LENGTH_TOL_RATE  = 0.2         # 비교 길이 여유(± 비율)

# ===== XHTML → 텍스트 =====
def extract_plain_from_xhtml(xhtml_html: str, p_only: bool = True) -> str:
    soup = BeautifulSoup(xhtml_html, "html.parser")

    # 1) converted.html 전용 경로: 페이지/블록 기반 수집
    pages = soup.select("section.page")
    if pages:
        parts = []
        for page in pages:
            # 페이지 내 실제 콘텐츠 그리드
            cols_root = page.select_one(".cols") or page
            # 가이드 박스(.box.*)는 제외, 실제 블록(.blk)만 순서대로
            for blk in cols_root.select(".blk"):
                cls = blk.get("class", [])
                # 텍스트 블록
                if "text" in cls:
                    lines = []
                    for ln in blk.select(".ln"):
                        # <br> → 개행
                        for br in ln.find_all("br"):
                            br.replace_with("\n")
                        t = " ".join(s.strip() for s in ln.stripped_strings)
                        if t:
                            lines.append(t)
                    if lines:
                        parts.append("\n".join(lines))
                # 표 블록
                elif "table" in cls:
                    tbl = blk.find("table")
                    if tbl:
                        rows = []
                        for tr in tbl.find_all("tr"):
                            cells = [" ".join(td.stripped_strings) for td in tr.find_all(["td", "th"])]
                            row = " ".join(c for c in cells if c is not None)
                            if row.strip():
                                rows.append(row.strip())
                        if rows:
                            parts.append("\n".join(rows))
            # 페이지 사이 공백 줄은 굳이 추가하지 않음(정렬 안정성 위해)
        txt = "\n".join(p for p in parts if p)
        # 탭/캐리지리턴 제거(기존 규칙과 동일)
        return txt.replace("\t", "").replace("\r", "")

    # 2) 폴백 A: <p>만 수집 (기존 동작 유지)
    if p_only:
        parts = []
        for p in soup.find_all("p"):
            for br in p.find_all("br"):
                br.replace_with("\n")
            t = "".join(p.strings)
            if not t:
                continue
            t = t.replace("\t", "").replace("\r", "")
            parts.append(t)
        return "\n".join(parts)

    # 3) 폴백 B: 문서 전체 텍스트
    for br in soup.find_all("br"):
        br.replace_with("\n")
    return "".join(soup.strings).replace("\t", "").replace("\r", "")

# ===== 유틸: 한자 식별 =====
def _is_han(ch: str) -> bool:
    cp = ord(ch)
    return (
        0x3400 <= cp <= 0x4DBF or 0x4E00 <= cp <= 0x9FFF or
        0xF900 <= cp <= 0xFAFF or 0x20000 <= cp <= 0x2A6DF or
        0x2A700 <= cp <= 0x2B73F or 0x2B740 <= cp <= 0x2B81F or
        0x2B820 <= cp <= 0x2CEAF or 0x2CEB0 <= cp <= 0x2EBEF or
        0x30000 <= cp <= 0x3134F
    )

# ===== 범용 한자 클래스(정규식용) =====
_HAN_BLOCKS = [
    (0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF),
    (0x20000, 0x2A6DF), (0x2A700, 0x2B73F), (0x2B740, 0x2B81F),
    (0x2B820, 0x2CEAF), (0x2CEB0, 0x2EBEF), (0x30000, 0x3134F),
]
def _build_han_charclass() -> str:
    parts = []
    for lo, hi in _HAN_BLOCKS:
        if hi <= 0xFFFF: parts.append(f"\\u{lo:04X}-\\u{hi:04X}")
        else:            parts.append(f"\\U{lo:08X}-\\U{hi:08X}")
    return "".join(parts)
_HAN_CLASS   = _build_han_charclass()
_RE_HAN_ALL  = re.compile(f"[{_HAN_CLASS}]")

# 첫 글자가 한자인지 식별
def _starts_with_han(s: str) -> bool:
    for ch in s or "":
        if ch.isspace():
            continue
        return bool(_RE_HAN_ALL.match(ch))
    return False

def _maybe_left_extend_one(src_text: str, start_adj: int, end_adj: int,
                           orig: str, new_text: str):
    """앞머리 1글자 누락 시 왼쪽 1칸 확장(유사도 악화 X 또는 확장 문자가 '의미문자'면 허용)."""
    o0 = _first_nonspace_char(orig)
    n0 = _first_nonspace_char(new_text)
    if not (o0 and n0):
        return start_adj, new_text, False

    if unicodedata.normalize("NFKC", o0).casefold() == unicodedata.normalize("NFKC", n0).casefold():
        return start_adj, new_text, False

    if start_adj <= 0:
        return start_adj, new_text, False
    if src_text[start_adj - 1] in ("\n", "\r"):
        return start_adj, new_text, False

    cand_text = src_text[start_adj - 1: end_adj].replace("\t", "").replace("\r", "")
    base_hanless = _norm_remove_ws_and_han(new_text)
    cand_hanless = _norm_remove_ws_and_han(cand_text)
    base_sim = _similarity(_norm_remove_ws_and_han(orig), base_hanless)
    cand_sim = _similarity(_norm_remove_ws_and_han(orig), cand_hanless)

    prev_ch = src_text[start_adj - 1]
    accept_extend = (
        cand_sim + 1e-9 >= base_sim   # 유사도 악화가 없으면 OK
        or _is_left_extensible_char(prev_ch)  # 글자/숫자/한자/마커/열거기호면 OK
    )
    if accept_extend:
        return start_adj - 1, cand_text, True
    return start_adj, new_text, False

# ===== 좌측 마커(도형/딩배트) =====
_MARKER_CLASS      = r"\u25A0-\u25FF\u2700-\u27BF"
_re_left_markers   = re.compile(rf"([{_MARKER_CLASS}]+[ \t]*)$")

def _reinclude_left_marker_context(src: str, start_idx: int, lookback: int = 8) -> int:
    a = max(0, start_idx - lookback)
    left = src[a:start_idx]
    m = _re_left_markers.search(left)
    if m:
        return a + m.start()
    return start_idx

# ===== 정규화(검색용) =====
def _simplify_and_map(src: str):
    s_norm_chars, idx_map = [], []
    for i, ch in enumerate(src):
        if ch == "\u318D":
            continue
        ch_fold = unicodedata.normalize("NFKC", ch).casefold()
        if not ch_fold:
            continue
        if unicodedata.category(ch)[0] in ("L", "N"):
            s_norm_chars.append(ch_fold[0]); idx_map.append(i)
    return "".join(s_norm_chars), idx_map

def _simplify_and_map_nohan(src: str):
    s_norm_chars, idx_map = [], []
    for i, ch in enumerate(src):
        if ch == "\u318D":
            continue
        if _is_han(ch):
            continue
        ch_fold = unicodedata.normalize("NFKC", ch).casefold()
        if not ch_fold:
            continue
        if unicodedata.category(ch)[0] in ("L", "N"):
            s_norm_chars.append(ch_fold[0]); idx_map.append(i)
    return "".join(s_norm_chars), idx_map

def _simplify_only(s: str) -> str:
    out = []
    for ch in s or "":
        if ch == "\u318D":
            continue
        ch_fold = unicodedata.normalize("NFKC", ch).casefold()
        if ch_fold and unicodedata.category(ch)[0] in ("L", "N"):
            out.append(ch_fold[0])
    return "".join(out)

def _simplify_letters_and_map(src: str):
    s_norm_chars, idx_map = [], []
    for i, ch in enumerate(src):
        if ch == "\u318D": continue
        ch_fold = unicodedata.normalize("NFKC", ch).casefold()
        if not ch_fold: continue
        if unicodedata.category(ch).startswith("L"):
            s_norm_chars.append(ch_fold[0]); idx_map.append(i)
    return "".join(s_norm_chars), idx_map

def _simplify_letters_and_map_nohan(src: str):
    s_norm_chars, idx_map = [], []
    for i, ch in enumerate(src):
        if ch == "\u318D": continue
        if _is_han(ch):     continue
        ch_fold = unicodedata.normalize("NFKC", ch).casefold()
        if not ch_fold: continue
        if unicodedata.category(ch).startswith("L"):
            s_norm_chars.append(ch_fold[0]); idx_map.append(i)
    return "".join(s_norm_chars), idx_map

def _simplify_letters_only(s: str) -> str:
    out = []
    for ch in s or "":
        if ch == "\u318D": continue
        ch_fold = unicodedata.normalize("NFKC", ch).casefold()
        if ch_fold and unicodedata.category(ch).startswith("L"):
            out.append(ch_fold[0])
    return "".join(out)

# ===== 열거기호 =====
def _is_enumerator(ch: str) -> bool:
    cp = ord(ch)
    return (0x2460 <= cp <= 0x2473) or (cp in (0x24EA, 0x24FF)) or (0x2776 <= cp <= 0x277F) or (0x2780 <= cp <= 0x2793)

def _first_nonspace_is_enumerator(s: str) -> bool:
    for ch in s or "":
        if not ch.isspace():
            return _is_enumerator(ch)
    return False

def _absorb_left_enumerator(src: str, start_idx: int) -> int:
    if start_idx <= 0: return start_idx
    j = start_idx - 1
    while j >= 0 and src[j].isspace(): j -= 1
    if j >= 0 and _is_enumerator(src[j]): return j
    return start_idx

_re_marker_single = re.compile(f"[{_MARKER_CLASS}]")

def _is_left_extensible_char(ch: str) -> bool:
    if not ch:
        return False
    cat0 = unicodedata.category(ch)[0]  # 'L', 'N', ...
    if cat0 in ("L", "N"):
        return True
    if _RE_HAN_ALL.match(ch):   # (한자는 Lo라 위에서 L로 잡히지만, 안전하게)
        return True
    if _re_marker_single.match(ch):  # ■●◇ 같은 도형/딩배트
        return True
    if _is_enumerator(ch):      # ①②⑩ 등 열거기호
        return True
    return False

# ===== 유사도/보조 =====
def _norm_remove_ws_only(s: str) -> str:
    if s is None: return ""
    s = unicodedata.normalize("NFKC", s).casefold()
    s = s.replace("\u200b", "").replace("\u318D", "·")
    s = re.sub(r"[\s\u00A0\u1680\u180E\u2000-\u200B\u2028\u2029\u202F\u205F\u3000]+", "", s)
    return s

def _norm_remove_ws_only_with_map(s: str) -> Tuple[str, list]:
    out, mp = [], []
    s_nf = unicodedata.normalize("NFKC", s).casefold()
    for i, ch in enumerate(s_nf):
        if ch == "\u200b":   # zero-width space
            continue
        if ch == "\u318D":   # ㆍ
            ch = "·"
        if ch.isspace():
            continue
        out.append(ch)
        mp.append(i)
    return "".join(out), mp

def _remove_han(s: str) -> str:
    if not s: return s or ""
    return _RE_HAN_ALL.sub("", s)

# === 새로 추가: "한자 제거 + 공백 제거" 정규화 ===
def _norm_remove_ws_and_han(s: str) -> str:
    return _norm_remove_ws_only(_remove_han(s))

def _norm_remove_ws_and_han_with_map(s: str) -> Tuple[str, list]:
    # NFKC/casefold 후 공백 제거 + 한자 제거. 맵은 정규화 문자열 인덱스로 기록(기존 방식 유지)
    out, mp = [], []
    s_nf = unicodedata.normalize("NFKC", s).casefold()
    for i, ch in enumerate(s_nf):
        if ch == "\u200b":
            continue
        if ch == "\u318D":
            ch = "·"
        if ch.isspace():
            continue
        # 한자 제거
        if _RE_HAN_ALL.match(ch):
            continue
        out.append(ch)
        mp.append(i)
    return "".join(out), mp

def _similarity(a: str, b: str) -> float:
    if a is None and b is None: return 1.0
    if not a or not b: return 0.0
    return round(SequenceMatcher(None, a, b).ratio(), 4)

def _count_edge_nonln(s: str):
    lead = 0
    for ch in s or "":
        if unicodedata.category(ch)[0] not in ("L", "N"): lead += 1
        else: break
    trail = 0
    for ch in reversed(s or ""):
        if unicodedata.category(ch)[0] not in ("L", "N"): trail += 1
        else: break
    return lead, trail

def _include_left_nonln(src: str, start_idx: int, max_count: int):
    i, got = start_idx - 1, 0
    while i >= 0 and got < max_count:
        if unicodedata.category(src[i])[0] not in ("L", "N"): i -= 1; got += 1
        else: break
    return i + 1, got

def _include_right_nonln(src: str, end_excl: int, max_count: int):
    j, got = end_excl, 0
    while j < len(src) and got < max_count:
        if unicodedata.category(src[j])[0] not in ("L", "N"): j += 1; got += 1
        else: break
    return j, got

# ===== 숫자/OCR 혼입 케이스 판단 =====
# ①~⑩(2460–2473), ⑴~⑽(2474–247D), ⑴⑵… 대역(2776–2793)까지 포함
RE_CIRCLED_NUM = r"\u2460-\u2473\u2474-\u247D\u2776-\u2793"
_re_hangul_digit_mix = re.compile(
    rf"(?:[가-힣]\s*[0-9{RE_CIRCLED_NUM}]+|[0-9{RE_CIRCLED_NUM}]+\s*[가-힣])"
)

def _needs_symbol_tolerant_fallback(orig: str) -> bool:
    return bool(_re_hangul_digit_mix.search(orig or ""))

def _refine_within_raw_window(orig: str, src_text: str, raw_start: int, raw_end_excl: int):
    """
    [raw_start, raw_end_excl) 윈도우 안에서 orig와 '한자 제거+공백 제거' 기준으로
    가장 유사한 부분 문자열을 찾는다. (길이 ±tol 탐색)
    """
    LEFT_HEAD_SLACK = 2
    raw_start = max(0, raw_start - LEFT_HEAD_SLACK)
    raw_end_excl = min(len(src_text), max(raw_start, raw_end_excl))
    window = src_text[raw_start:raw_end_excl]
    if not window:
        return None

    w_norm, w_map = _norm_remove_ws_and_han_with_map(window)
    target = _norm_remove_ws_and_han(orig)
    if not w_norm or not target:
        return None

    Lr = len(target)
    tol = max(LENGTH_TOL_BASE, int(Lr * LENGTH_TOL_RATE))
    lmin = max(5, Lr - tol)
    lmax = min(len(w_norm), Lr + tol)

    best = (0.0, 0, 0)  # (sim, s, e)
    for length in range(lmin, lmax + 1):
        for s in range(0, len(w_norm) - length + 1):
            seg = w_norm[s:s+length]
            r = SequenceMatcher(None, target, seg).ratio()
            if r > best[0]:
                best = (r, s, s + length)

    if best[0] < SIM_THRESHOLD:
        return None

    s_norm, e_norm = best[1], best[2]
    raw_s = raw_start + w_map[s_norm]
    raw_e = raw_start + w_map[e_norm - 1] + 1

    # 좌/우 비-문자 보정 + 한자/마커 보정은 기존 규칙 재사용
    lead_nonln, trail_nonln = _count_edge_nonln(orig)
    raw_s, _ = _include_left_nonln(src_text, raw_s, lead_nonln)
    raw_e, _ = _include_right_nonln(src_text, raw_e, trail_nonln)
    if _first_nonspace_is_enumerator(orig):
        raw_s = _absorb_left_enumerator(src_text, raw_s)
    raw_s = _reinclude_left_han_context(src_text, raw_s)
    raw_s = _reinclude_left_marker_context(src_text, raw_s)

    new_text = src_text[raw_s:raw_e].replace("\t","").replace("\r","")
    sim = _similarity(_norm_remove_ws_and_han(orig), _norm_remove_ws_and_han(new_text))
    return (raw_s, raw_e, new_text, sim)

# ===== '제…조' 머리 교정 =====
_re_article_head_span = re.compile(r"^\s*제(.{0,8})조")
def _maybe_fix_article_head(orig: str) -> Optional[str]:
    m = _re_article_head_span.search(orig or "")
    if not m: return None
    s, e = m.span()
    return (orig[:s] + "제조" + orig[e:])

_re_article_head_token = re.compile(r"^\s*제\s*([^\s]{1,8})\s*조\s*$")
def _normalize_article_token(tok: str) -> str:
    t = re.sub(r"\s+", "", tok or "")
    trans = str.maketrans({"0":"○","O":"○","o":"○","ㅇ":"○","〇":"○"})
    t = t.translate(trans)
    t = re.sub(r"으+", "○", t)
    t = re.sub(r"[○□◇△◦•]", "○", t)
    return t

def _article_head_equivalent(a: str, b: str) -> bool:
    ma = _re_article_head_token.match(a or "")
    mb = _re_article_head_token.match(b or "")
    if not ma or not mb: return False
    return _normalize_article_token(ma.group(1)) == _normalize_article_token(mb.group(1))

_re_article_scan_raw = re.compile(r"제[ \t\u00A0]*([^\s]{1,8})[ \t\u00A0]*조")

# ===== 범용 텍스트-온리 폴백 =====
_re_speaker = re.compile(r"(^|[\s([{<])([^\W\d_])\s*[:\uFF1A]\s*", re.UNICODE)
_re_jamo    = re.compile(r"[\u1100-\u11FF\u3130-\u318F\uA960-\uA97F\uD7B0-\uD7FF]+")

def _simplify_only_strip_generic_noise(s: str) -> str:
    if not s: return ""
    s2 = _re_speaker.sub(r"\1", s)
    s2 = _RE_HAN_ALL.sub("", s2)
    s2 = _re_jamo.sub("", s2)
    return _simplify_only(s2)

# ===== 울트라 폴백(완성형 한글+숫자) =====
def _simplify_kosyllable_digit_only(s: str) -> str:
    out = []
    for ch in unicodedata.normalize("NFKC", s or ""):
        cp = ord(ch)
        if 0xAC00 <= cp <= 0xD7A3: out.append(ch)
        else:
            if unicodedata.category(ch).startswith("N"): out.append(ch)
    return "".join(out)

# ===== 한자 좌측 컨텍스트 재포함 =====
_re_left_han_colon = re.compile(f"([{_HAN_CLASS}]+[ \\t]*[:\\uFF1A])[ \\t]*$")
_re_left_han_wave  = re.compile(f"([{_HAN_CLASS}][ \\t]*[~\\u301C\\uFF5E\\u223C][ \\t]*[{_HAN_CLASS}])[ \\t]*$")
def _reinclude_left_han_context(src: str, start_idx: int, lookback: int = 24) -> int:
    a = max(0, start_idx - lookback)
    left = src[a:start_idx]
    m = _re_left_han_colon.search(left)
    if m: return a + m.start()
    m = _re_left_han_wave.search(left)
    if m: return a + m.start()
    j = start_idx - 1
    while j >= 0 and _RE_HAN_ALL.match(src[j]): j -= 1
    return j + 1 if j + 1 < start_idx else start_idx

def _starts_with_colon(s: str) -> bool:
    for ch in s or "":
        if not ch.isspace():
            return ch in (":", "\uFF1A")
    return False

def _first_nonspace_char(s: str) -> str:
    for ch in s or "":
        if not ch.isspace():
            return ch
    return ""

# ===== (NEW) 앵커 기반 폴백(위치만) =====
def _anchored_fallback(q_norm: str,
                       src_norm: str, src_map: list, cursor_src: int,
                       src_norm_nohan: str, src_map_nohan: list, cursor_nh: int
                       ) -> Tuple[int, Optional[list], Optional[str], Optional[int]]:
    L = len(q_norm)
    if L < MIN_LEN_FOR_ANCHOR:
        return -1, None, None, None

    anchors = []
    anchors.append( (q_norm[:MIN_ANCHOR], "head") )
    mid = L // 2
    m_start = max(0, mid - MIN_ANCHOR // 2)
    anchors.append( (q_norm[m_start:m_start+MIN_ANCHOR], "mid") )
    anchors.append( (q_norm[-MIN_ANCHOR:], "tail") )

    for frag, _tag in anchors:
        if not frag: continue
        p = src_norm.find(frag, cursor_src)
        if p != -1:
            return p, src_map, "default", L
        p = src_norm_nohan.find(frag, cursor_nh)
        if p != -1:
            return p, src_map_nohan, "nohan", L
    return -1, None, None, None

# ===== (NEW) 앵커-윈도우 구명줄(창 내부 최고 유사도; 한자 무시 비교) =====
def _anchor_rescue(orig: str, q_norm: str, src_text: str,
                   src_norm: str, src_map: list, cursor_src: int,
                   src_norm_nohan: str, src_map_nohan: list, cursor_nh: int
                   ) -> Optional[Tuple[int,int,str,float]]:
    Lq = len(q_norm)
    if Lq < MIN_LEN_FOR_ANCHOR:
        return None

    # 1) 앵커로 raw 위치 수집
    anchors = [q_norm[:MIN_ANCHOR]]
    mid = Lq // 2
    m_start = max(0, mid - MIN_ANCHOR // 2)
    anchors.append(q_norm[m_start:m_start+MIN_ANCHOR])
    anchors.append(q_norm[-MIN_ANCHOR:])

    raw_hits = []
    for frag in anchors:
        if not frag:
            continue
        p = src_norm.find(frag, cursor_src)
        if p != -1:
            raw_hits.append(src_map[p]); continue
        p = src_norm_nohan.find(frag, cursor_nh)
        if p != -1:
            raw_hits.append(src_map_nohan[p]); continue

    if not raw_hits:
        return None

    # 2) 원문 창 계산
    raw_min, raw_max = min(raw_hits), max(raw_hits)
    pad = max(ANCHOR_PAD_MIN, int(len(orig) * ANCHOR_PAD_FACTOR))
    win_start = max(0, raw_min - pad)
    win_end   = min(len(src_text), raw_max + pad)
    if win_end - win_start > ANCHOR_MAX_WINDOW:
        center = (raw_min + raw_max) // 2
        half = ANCHOR_MAX_WINDOW // 2
        win_start = max(0, center - half)
        win_end   = min(len(src_text), center + half)

    window = src_text[win_start:win_end]

    # 3) 창을 "한자 제거 + 공백 제거"로 정규화(매핑 포함)
    w_norm, w_map = _norm_remove_ws_and_han_with_map(window)
    target = _norm_remove_ws_and_han(orig)
    if not w_norm or not target:
        return None

    # 4) 창에서 대상과 가장 비슷한 부분문자열(길이 근사) 탐색
    Lr = len(target)
    tol = max(LENGTH_TOL_BASE, int(Lr * LENGTH_TOL_RATE))
    lmin = max(5, Lr - tol)
    lmax = min(len(w_norm), Lr + tol)

    best = (0.0, 0, 0)  # (sim, s, e)
    for length in range(lmin, lmax + 1):
        for s in range(0, len(w_norm) - length + 1):
            seg = w_norm[s:s+length]
            r = SequenceMatcher(None, target, seg).ratio()
            if r > best[0]:
                best = (r, s, s + length)

    if best[0] < SIM_THRESHOLD:
        return None

    # 5) 정규화 인덱스를 원문 인덱스로 환산
    s_norm, e_norm = best[1], best[2]
    raw_s = win_start + w_map[s_norm]
    raw_e = win_start + w_map[e_norm - 1] + 1

    # 6) 비-L/N 포함 + 열거기호/한자/마커 보정
    lead_nonln, trail_nonln = _count_edge_nonln(orig)
    raw_s, _ = _include_left_nonln(src_text, raw_s, lead_nonln)
    raw_e, _ = _include_right_nonln(src_text, raw_e, trail_nonln)
    if _first_nonspace_is_enumerator(orig):
        raw_s = _absorb_left_enumerator(src_text, raw_s)
    raw_s = _reinclude_left_han_context(src_text, raw_s)
    raw_s = _reinclude_left_marker_context(src_text, raw_s)

    new_text = src_text[raw_s:raw_e].replace("\t","").replace("\r","")

    # 7) 최종 유사도(한자 제거 + 공백 제거)로만 판정
    orig_hanless = _norm_remove_ws_and_han(orig)
    new_hanless  = _norm_remove_ws_and_han(new_text)
    sim = _similarity(orig_hanless, new_hanless)

    if orig_hanless == new_hanless or sim >= SIM_THRESHOLD:
        return raw_s, raw_e, new_text, sim
    return None

# ===== 치환 =====
def replace_text_blocks(jdata: list, src_text: str) -> int:
    FN_NAME = "replace_text_blocks"

    # 1차: 기본 앵커들
    src_norm, src_map = _simplify_and_map(src_text)
    src_norm_nohan, src_map_nohan = _simplify_and_map_nohan(src_text)
    src_norm_letters, src_map_letters = _simplify_letters_and_map(src_text)
    src_norm_letters_nohan, src_map_letters_nohan = _simplify_letters_and_map_nohan(src_text)

    cursor_default = cursor_nohan = cursor_letters = cursor_letters_nohan = 0
    replaced_cnt = 0

    def _probe_next_body_raw_start(start_idx_in_j):
        for k in range(start_idx_in_j + 1, len(jdata)):
            nb = jdata[k]
            if nb.get("type") != "text":
                continue
            q2 = _simplify_only(nb.get("text", ""))
            if len(q2) < 20:
                continue
            pos2 = src_norm.find(q2, cursor_default); used_map2 = src_map
            if pos2 == -1:
                pos2 = src_norm_nohan.find(q2, cursor_nohan); used_map2 = src_map_nohan
            if pos2 == -1 and _needs_symbol_tolerant_fallback(nb.get("text", "")):
                q2l = _simplify_letters_only(nb.get("text", ""))
                if q2l:
                    pos2 = src_norm_letters.find(q2l, cursor_letters); used_map2 = src_map_letters
                    if pos2 == -1:
                        pos2 = src_norm_letters_nohan.find(q2l, cursor_letters_nohan); used_map2 = src_map_letters_nohan
            if pos2 != -1:
                return used_map2[pos2]
        return None

    for i, blk in enumerate(jdata):
        did_left_expand = False
        if blk.get("type") == "inline_equation":
            blk["_xhtml_replace"] = {
                "function": FN_NAME, "replaced": "inline_equation",
                "similarity": None, "similarity_mode": "no_ws_hanless",
                "start_orig": None, "end_orig": None,
                "reason": "non_text_block",
            }
            continue
        if blk.get("type") != "text":
            continue

        orig = blk.get("text", "")
        q_norm = _simplify_only(orig)

        log = {
            "function": FN_NAME, "replaced": False,
            "similarity": None, "similarity_mode": "no_ws_hanless",
            "start_orig": None, "end_orig": None, "reason": ""
        }

        if not q_norm:
            log["reason"] = "empty_query"
            blk["_xhtml_replace"] = log
            continue

        # ========= 특수 경로: '제**조' 단독 =========
        if _re_article_head_token.match(orig):
            next_raw = _probe_next_body_raw_start(i)

            def _raw_from_cursor(cur, mp):
                if not mp: return 0
                if cur <= 0: return mp[0]
                if cur >= len(mp): return mp[-1] + 1
                return mp[cur]

            start_raw = max(
                _raw_from_cursor(cursor_default,       src_map),
                _raw_from_cursor(cursor_nohan,         src_map_nohan),
                _raw_from_cursor(cursor_letters,       src_map_letters),
                _raw_from_cursor(cursor_letters_nohan, src_map_letters_nohan),
            )
            end_raw = next_raw if next_raw is not None else len(src_text)

            chosen = None
            for m in _re_article_scan_raw.finditer(src_text, pos=start_raw):
                if m.start() >= end_raw:
                    break
                if _article_head_equivalent(orig, m.group(0)):
                    chosen = m

            if chosen is not None:
                start_adj = chosen.start()
                end_adj   = chosen.end()
                new_text  = src_text[start_adj:end_adj].replace("\t","").replace("\r","")

                # 한자 무시 유사도
                orig_hanless = _norm_remove_ws_and_han(orig)
                new_hanless  = _norm_remove_ws_and_han(new_text)
                sim = _similarity(orig_hanless, new_hanless)

                blk["text"] = new_text
                log.update({
                    "replaced": True,
                    "similarity": sim,
                    "start_orig": start_adj,
                    "end_orig": end_adj,
                    "reason": "제**조-window"
                })
                blk["_xhtml_replace"] = log
                replaced_cnt += 1

                idx_after            = bisect_left(src_map,               end_adj)
                idx_after_nohan      = bisect_left(src_map_nohan,         end_adj)
                idx_after_letters    = bisect_left(src_map_letters,       end_adj)
                idx_after_letters_nh = bisect_left(src_map_letters_nohan, end_adj)
                cursor_default       = max(cursor_default,       idx_after)
                cursor_nohan         = max(cursor_nohan,         idx_after_nohan)
                cursor_letters       = max(cursor_letters,       idx_after_letters)
                cursor_letters_nohan = max(cursor_letters_nohan, idx_after_letters_nh)
                continue
            # 못 찾으면 일반 경로로

        # ------------------- 일반 경로 -------------------
        used_map = None
        used_cursor = None
        pos = -1
        q_len_for_cursor = None
        used_anchor = False   # <- 앵커 사용 여부 플래그

        # (0) '제…조' 교정 쿼리
        fixed = _maybe_fix_article_head(orig)
        if fixed is not None:
            q_fixed = _simplify_only(fixed)
            if q_fixed:
                pos = src_norm.find(q_fixed, cursor_default)
                if pos != -1:
                    used_map = src_map; used_cursor = "default"; q_len_for_cursor = len(q_fixed)

        # (1) 기본 L/N
        if pos == -1:
            pos = src_norm.find(q_norm, cursor_default)
            used_map = src_map; used_cursor = "default"; q_len_for_cursor = len(q_norm)

        # (2) nohan
        if pos == -1:
            pos = src_norm_nohan.find(q_norm, cursor_nohan)
            used_map = src_map_nohan; used_cursor = "nohan"; q_len_for_cursor = len(q_norm)

        # (2.6) 범용 텍스트-온리
        if pos == -1:
            q_generic = _simplify_only_strip_generic_noise(orig)
            if q_generic and q_generic != q_norm:
                pos = src_norm_nohan.find(q_generic, cursor_nohan)
                if pos != -1:
                    used_map = src_map_nohan; used_cursor = "nohan"; q_len_for_cursor = len(q_generic)

        # (2.7) 완성형 한글+숫자
        if pos == -1:
            q_kod = _simplify_kosyllable_digit_only(orig)
            if q_kod and q_kod != q_norm:
                pos = src_norm_nohan.find(q_kod, cursor_nohan)
                if pos != -1:
                    used_map = src_map_nohan; used_cursor = "nohan"; q_len_for_cursor = len(q_kod)

        # (2.8) 앵커 기반 폴백(위치만)
        if pos == -1:
            ppos, umap, ucur, qlen = _anchored_fallback(
                q_norm, src_norm, src_map, cursor_default,
                src_norm_nohan, src_map_nohan, cursor_nohan
            )
            pos, used_map, used_cursor, q_len_for_cursor = ppos, umap, ucur, qlen
            if pos != -1:
                used_anchor = True   # 앵커 사용 표시

        # (3) 숫자/OCR → letters-only
        if pos == -1 and _needs_symbol_tolerant_fallback(orig):
            q_letters = _simplify_letters_only(orig)
            if q_letters:
                pos = src_norm_letters.find(q_letters, cursor_letters)
                if pos != -1:
                    used_map = src_map_letters; used_cursor = "letters"; q_len_for_cursor = len(q_letters)
            if pos == -1 and q_letters:
                pos = src_norm_letters_nohan.find(q_letters, cursor_letters_nohan)
                if pos != -1:
                    used_map = src_map_letters_nohan; used_cursor = "letters_nohan"; q_len_for_cursor = len(q_letters)

        # ---- 완전 실패 → 앵커-윈도우 구명줄 ----
        if pos == -1 or used_map is None:
            rescue = _anchor_rescue(
                orig, q_norm, src_text,
                src_norm, src_map, cursor_default,
                src_norm_nohan, src_map_nohan, cursor_nohan
            )
            if rescue is not None:
                start_adj, end_adj, new_text, sim = rescue

                start_adj2, new_text2, did_ext = _maybe_left_extend_one(src_text, start_adj, end_adj, orig, new_text)
                if did_ext:
                    start_adj = start_adj2
                    new_text = new_text2
                    sim = _similarity(_norm_remove_ws_and_han(orig), _norm_remove_ws_and_han(new_text))

                blk["text"] = new_text
                log.update({
                    "replaced": True,
                    "similarity": sim,
                    "start_orig": start_adj,
                    "end_orig": end_adj,
                    "reason": "anchor_rescue" + ("+left_extend" if did_ext else "")
                })
                blk["_xhtml_replace"] = log
                replaced_cnt += 1
                # 커서 전진(원문 기준 동기화)
                idx_after            = bisect_left(src_map,               end_adj)
                idx_after_nohan      = bisect_left(src_map_nohan,         end_adj)
                idx_after_letters    = bisect_left(src_map_letters,       end_adj)
                idx_after_letters_nh = bisect_left(src_map_letters_nohan, end_adj)
                cursor_default       = max(cursor_default,       idx_after)
                cursor_nohan         = max(cursor_nohan,         idx_after_nohan)
                cursor_letters       = max(cursor_letters,       idx_after_letters)
                cursor_letters_nohan = max(cursor_letters_nohan, idx_after_letters_nh)
                continue

            log["reason"] = "not_found"
            blk["_xhtml_replace"] = log
            continue

        # ---- (pos, used_map) 확보된 정상 경로 ----
        start_orig = used_map[pos]
        end_idx_norm = min(pos + q_len_for_cursor - 1, len(used_map) - 1)
        end_orig_excl = used_map[end_idx_norm] + 1

        # JSON 양끝 non-L/N 개수만큼 포함
        lead_nonln, trail_nonln = _count_edge_nonln(orig)
        start_adj, _ = _include_left_nonln(src_text, start_orig, lead_nonln)
        end_adj, _   = _include_right_nonln(src_text, end_orig_excl, trail_nonln)

        # 열거기호 흡수
        if _first_nonspace_is_enumerator(orig):
            start_adj = _absorb_left_enumerator(src_text, start_adj)

        # 한자 라벨/범위 재포함 (조건부)
        if used_cursor in ("nohan", "letters_nohan") or _starts_with_colon(orig):
            start_adj = _reinclude_left_han_context(src_text, start_adj)

        # 마커/한자 항상 재포함
        start_adj = _reinclude_left_han_context(src_text, start_adj)
        start_adj = _reinclude_left_marker_context(src_text, start_adj)

        # 최종 슬라이스
        new_text = src_text[start_adj:end_adj].replace("\t", "").replace("\r", "")
        if not new_text:
            log.update({"start_orig": start_adj, "end_orig": end_adj, "reason": "empty_slice", "similarity": 0.0})
            blk["_xhtml_replace"] = log
            if used_cursor == "default":       cursor_default       = pos + q_len_for_cursor
            elif used_cursor == "nohan":       cursor_nohan         = pos + q_len_for_cursor
            elif used_cursor == "letters":     cursor_letters       = pos + q_len_for_cursor
            else:                              cursor_letters_nohan = pos + q_len_for_cursor
            continue

        # --- [NEW] 첫 글자 오탐(앞머리 1글자 누락) 보정 --------------------
        o0 = _first_nonspace_char(orig)
        n0 = _first_nonspace_char(new_text)
        if o0 and n0 and unicodedata.normalize("NFKC", o0).casefold() != unicodedata.normalize("NFKC", n0).casefold():
            # 왼쪽 1글자만 확장 시도 (줄바꿈은 건너뜀)
            if start_adj > 0 and src_text[start_adj-1] not in ("\n", "\r"):
                cand_text = src_text[start_adj-1:end_adj].replace("\t", "").replace("\r", "")

                # 한자/공백 제거 기준 유사도 비교
                base_hanless = _norm_remove_ws_and_han(new_text)
                cand_hanless = _norm_remove_ws_and_han(cand_text)
                base_sim = _similarity(_norm_remove_ws_and_han(orig), base_hanless)
                cand_sim = _similarity(_norm_remove_ws_and_han(orig), cand_hanless)

                prev_ch = src_text[start_adj-1]
                accept_extend = (
                    cand_sim + 1e-9 >= base_sim              # 유사도 안 나빠지면 OK
                    or _is_left_extensible_char(prev_ch)     # 앞 글자가 글자/한자/마커/열거기호면 OK
                )
                if accept_extend:
                    start_adj -= 1
                    new_text = cand_text
                    did_left_expand = True

        # === (추가) 윈도우 재정렬로 끝/앞머리 보정 + 다음 블록 흡수 방지 ===
        next_raw = _probe_next_body_raw_start(i)
        orig_hanless = _norm_remove_ws_and_han(orig)
        new_hanless  = _norm_remove_ws_and_han(new_text)
        base_sim     = _similarity(orig_hanless, new_hanless)

        need_refine = False
        if used_anchor or _needs_symbol_tolerant_fallback(orig):
            need_refine = True
        elif next_raw is not None and end_adj <= next_raw:
            if len(orig_hanless) >= len(new_hanless) + 1:
                need_refine = True

        if need_refine and next_raw is not None and end_adj <= next_raw:
            refined = _refine_within_raw_window(orig, src_text, start_adj, next_raw)
            if refined is not None:
                r_s, r_e, r_text, r_sim = refined
                if (r_sim + 1e-9 >= base_sim and (r_e - r_s) >= (end_adj - start_adj) and (not did_left_expand or r_s <= start_adj)):
                    start_adj, end_adj, new_text = r_s, r_e, r_text
                    new_hanless = _norm_remove_ws_and_han(new_text)
                    base_sim    = r_sim

        # === 유사도 업데이트
        sim = base_sim
        log.update({"similarity": sim, "start_orig": start_adj, "end_orig": end_adj})

        # 승인/거절
        accepted = False
        reason = ""
        if _article_head_equivalent(orig, new_text):
            accepted = True; reason = "제**조"
        elif (orig_hanless == new_hanless or sim >= SIM_THRESHOLD):
            accepted = True; reason = "passed_threshold_no_ws"

        if not accepted:
            # 마지막 구명줄: 앵커-윈도우 재시도
            rescue = _anchor_rescue(
                orig, q_norm, src_text,
                src_norm, src_map, cursor_default,
                src_norm_nohan, src_map_nohan, cursor_nohan
            )
            if rescue is not None:
                start_adj, end_adj, new_text, sim2 = rescue

                start_adj2, new_text2, did_ext = _maybe_left_extend_one(src_text, start_adj, end_adj, orig, new_text)
                if did_ext:
                    start_adj = start_adj2
                    new_text = new_text2
                    sim2 = _similarity(_norm_remove_ws_and_han(orig), _norm_remove_ws_and_han(new_text))

                blk["text"] = new_text
                log.update({
                    "replaced": True,
                    "similarity": sim2,
                    "start_orig": start_adj,
                    "end_orig": end_adj,
                    "reason": "anchor_rescue" + ("+left_extend" if did_ext else "")
                })
                blk["_xhtml_replace"] = log
                replaced_cnt += 1

                idx_after            = bisect_left(src_map,               end_adj)
                idx_after_nohan      = bisect_left(src_map_nohan,         end_adj)
                idx_after_letters    = bisect_left(src_map_letters,       end_adj)
                idx_after_letters_nh = bisect_left(src_map_letters_nohan, end_adj)
                cursor_default       = max(cursor_default,       idx_after)
                cursor_nohan         = max(cursor_nohan,         idx_after_nohan)
                cursor_letters       = max(cursor_letters,       idx_after_letters)
                cursor_letters_nohan = max(cursor_letters_nohan, idx_after_letters_nh)
                continue

            log["reason"] = "low_similarity"
            blk["_xhtml_replace"] = log
            continue

        # 승인됨
        blk["text"] = new_text
        log["replaced"] = True
        log["reason"] = reason
        blk["_xhtml_replace"] = log
        replaced_cnt += 1

        # 커서 전진
        if used_cursor == "default":
            cursor_default = pos + q_len_for_cursor
        elif used_cursor == "nohan":
            cursor_nohan = pos + q_len_for_cursor
        elif used_cursor == "letters":
            cursor_letters = pos + q_len_for_cursor
        else:
            cursor_letters_nohan = pos + q_len_for_cursor

    return replaced_cnt

# ======================================================
# ===== 폴더 매칭/실행 유틸 (이름 유사도 기반) =====
# ======================================================
_SUFFIX_CLEAN_RE = re.compile(r"(?:_content_list|_choice_plain|_layout|_middle)(?:_[^\\/]+)?$", re.I)

def _json_key_from_path(p: Path) -> str:
    stem = unicodedata.normalize("NFC", p.stem)
    stem = _SUFFIX_CLEAN_RE.sub("", stem)
    return _simplify_only(stem)

def _xhtml_key_from_path(p: Path) -> str:
    name = unicodedata.normalize("NFC", p.stem)
    # index.xhtml이므로 부모 폴더명을 키로 사용
    return _simplify_only(name)

def _best_match_xhtml_for_json(json_path: Path, xhtml_files) -> Tuple[Optional[Path], float, str, str]:
    jkey_raw = json_path.stem
    jkey = _json_key_from_path(json_path)
    best, best_score, best_key = None, 0.0, ""
    for x in xhtml_files:
        xkey_raw = x.stem
        xkey = _xhtml_key_from_path(x)
        if jkey and xkey and (jkey in xkey or xkey in jkey):
            score = 1.1  # 부분 포함 보너스(사실상 확정)
        else:
            score = SequenceMatcher(None, jkey, xkey).ratio()
        if score > best_score:
            best, best_score, best_key = x, score, xkey_raw
    return best, best_score, jkey_raw, best_key

def _process_one_pair(json_path: Path, xhtml_path: Path) -> int:
    jdata = json.loads(json_path.read_text(encoding="utf-8"))
    xhtml = xhtml_path.read_text(encoding="utf-8")
    src_text = extract_plain_from_xhtml(xhtml, p_only=True)
    replaced = replace_text_blocks(jdata, src_text)

    out_dir = OUT_DIR / json_path.relative_to(JSON_DIR_IN).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = json_path.stem.removesuffix("_process")
    out_path = out_dir / f"{stem}_merge.json"
    out_path.write_text(json.dumps(jdata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[WRITE] {out_path}")
    return replaced

def folder_main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 모든 index.xhtml 수집
    def _collect_html_files(base: Path):
        return (
            list(base.rglob("*.html"))
            + list(base.rglob("*.xhtml"))
            + list(base.rglob("*.htm"))
        )

    xhtml_files = _collect_html_files(XHTML_DIR_IN)
    if not xhtml_files:
        print(f"[WARN] html 없음: {XHTML_DIR_IN}")

    # 모든 JSON 수집
    json_files  = [p for p in JSON_DIR_IN.rglob("*.json")]
    if not json_files:
        print(f"[WARN] JSON 없음: {JSON_DIR_IN}")

    total = 0
    matched = 0
    for j in sorted(json_files):
        total += 1
        best_x, score, jname, xname = _best_match_xhtml_for_json(j, xhtml_files)
        if best_x is None or score < NAME_SIM_THRESHOLD:
            print(f"[SKIP] 이름 유사도 낮음: {j.name}  (best={xname}, score={score:.3f})")
            continue
        print(f"[PAIR] {j.name}  <->  {best_x}  (score={score:.3f})")
        try:
            replaced = _process_one_pair(j, best_x)
            print(f"       → replaced {replaced} blocks")
            matched += 1
        except Exception as e:
            print(f"[ERROR] 처리 중 오류: {j} / {best_x}\n    {e}")

    print(f"\n[DONE] 총 JSON {total}개 중 {matched}개 매칭/병합 완료. 출력 루트: {OUT_DIR}")

# ===== 엔트리포인트 =====
if __name__ == "__main__":
    folder_main()
