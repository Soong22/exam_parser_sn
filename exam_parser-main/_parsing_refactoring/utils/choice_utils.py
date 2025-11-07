# -*- coding: utf-8 -*-
from __future__ import annotations
import re
from typing import Any, Dict, List, Optional, Tuple
from constants import (
    RE_MARKER_TOKEN, RE_BOGI_TAG_LINE, RE_VIEW_ENUM_START, RE_CHOICE_INLINE,
    RE_VIEW_UNIT, RE_ZERO_WIDTH, PH_RE
)
from text_utils import normalize_text, clean_text_keep_newlines, clean_text

_CONNECTOR_ONLY_RE = re.compile(r"^[\s\-\~–—,·.\u00B7／/\\()\[\]{}<>:;|]*$")

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
    tokens = PH_RE.findall("\n".join(bodies))
    if not tokens: return pairs
    only_assets = [((re.sub(PH_RE, "", b) or "").strip() == "") for b in bodies]
    if all(only_assets):
        it = iter(tokens)
        new_pairs = []
        for lab, body in pairs:
            tok = next(it, None)
            new_pairs.append((lab, tok if tok is not None else body))
        return new_pairs
    return pairs

def _is_marker_decor_line(s: str) -> bool:
    if not s or not s.strip(): return False
    u = normalize_text(s)
    circ_cnt = len(RE_MARKER_TOKEN.findall(u))
    if circ_cnt < 2: return False
    tmp = RE_MARKER_TOKEN.sub("", u)
    return _CONNECTOR_ONLY_RE.match((tmp.strip() or "")) is not None

def _find_tail_start_idx(marks: List[Tuple[int,int,str,Optional[int],str]]) -> Optional[int]:
    n = len(marks)
    if n < 2: return None
    for i in range(n - 2, -1, -1):
        # marks[i][3] == 1 means circled "①"
        if marks[i][3] == 1:
            return i
    return n - 2

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
    from constants import UC2NUM
    out: List[Tuple[int, int, str, Optional[int], str]] = []
    raw_keep = text or ""
    for m in RE_MARKER_TOKEN.finditer(raw_keep):
        ch = m.group(1)
        num = UC2NUM.get(ch)
        num_i = int(num) if num and num.isdigit() else None
        out.append((m.start(), m.end(), ch, num_i, "circled"))
    out.sort(key=lambda x: x[0])
    return out

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
        m2 = RE_VIEW_ENUM_START.search(aft)
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