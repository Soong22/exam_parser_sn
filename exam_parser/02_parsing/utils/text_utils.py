# -*- coding: utf-8 -*-
from __future__ import annotations
import unicodedata, re
from core.constants import RE_ZERO_WIDTH, RE_LEAD_QNUM

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

def strip_leading_qnum(s: str) -> str:
    return RE_LEAD_QNUM.sub("", s or "", count=1)
