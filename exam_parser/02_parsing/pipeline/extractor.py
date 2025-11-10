# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Any, Dict, List, Tuple, Optional
import re
from core.models import Question
from core.constants import (
    RE_Q_HEADER, RE_CHOICE_INLINE, RE_VIEW_UNIT, RE_HTML_TABLE
)
from core.constants import RE_BOGI_TAG_LINE, RE_ZERO_WIDTH
from utils.text_utils import normalize_text, strip_leading_qnum
from pipeline.placeholders import (
    insert_formula_placeholders, insert_table_placeholders,
    FORMULA_BLOCK_TYPES, _latex_from_block
)
from utils.choice_utils import (
    explode_inline_choice_blocks, route_prefix_generically,
    slice_choices_by_markers_tailonly
)
from pipeline.image_title import _guess_image_title

def _split_question_and_view(text: str) -> Tuple[str, str]:
    if not text: return "", ""
    s = RE_ZERO_WIDTH.sub("", text or "")
    m_bogi = RE_BOGI_TAG_LINE.search(s)
    start_idx = m_bogi.start() if m_bogi else None
    m_bul = re.compile(r"(?:^|\n|\s)(?:ㄱ|ㄴ|ㄷ|ㄹ|ㅁ|ㅂ|ㅅ|ㅇ|ㅈ|ㅊ|ㅋ|ㅌ|ㅍ|ㅎ)\s*[)\.]\s*").search(s)
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

def _pack_view_with_newlines(vtxt: str) -> str:
    # choice_utils._pack_view_with_newlines의 라이트 버전 (의존 순환 방지용)
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
    import re as _re
    out = "\n".join(it for it in items if it)
    out = _re.sub(r"\n+", "\n", out).strip()
    return out

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
                            title = (caption or nb.get("title") or nb.get("caption") or "") or ""
                            cur._tables.append({
                                "html": m_tbl.group(0),
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
                    import re as _re
                    big2 = _re.sub(r"[ \t]+\n", "\n", big)
                    big2 = _re.sub(r"\n{3,}", "\n\n", big2).strip()
                    cur.body += ("\n" if cur.body else "") + big2
                i = j; continue
        i += 1

    push()
    return res
