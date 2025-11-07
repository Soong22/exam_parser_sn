# -*- coding: utf-8 -*-
from __future__ import annotations
import os, re, unicodedata
from typing import Dict, Tuple

def _strip_known_suffixes(base: str) -> str:
    import re as _re
    return _re.sub(r"(?:_content_list|_layout|_middle)$", "", base, flags=_re.I)

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
