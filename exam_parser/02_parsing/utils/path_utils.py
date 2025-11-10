# -*- coding: utf-8 -*-
from __future__ import annotations
import os, sys, glob, unicodedata
from typing import List

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

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
