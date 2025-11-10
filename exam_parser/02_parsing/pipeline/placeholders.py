# -*- coding: utf-8 -*-
from __future__ import annotations
import re
from typing import Any, Dict
from utils.table_utils import sanitize_table_html
from core.constants import RE_INLINE_TEX, RE_HTML_TABLE

FORMULA_BLOCK_TYPES = {"formula", "inline_equation"}

def _strip_math_delims(s: str) -> str:
    s = (s or "").strip()
    if s.startswith(r"\(") and s.endswith(r"\)"):
        return s[2:-2].strip()
    return s

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
