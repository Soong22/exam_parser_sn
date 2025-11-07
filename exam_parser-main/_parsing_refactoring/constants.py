# -*- coding: utf-8 -*-
from __future__ import annotations
import re

# ---- 공통 정규식/상수 (원문 그대로) ----
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

RE_VIEW_ENUM_START = re.compile(r"(?:^|\n|\s)(?:ㄱ|ㄴ|ㄷ|ㄹ|ㅁ|ㅂ|ㅅ|ㅇ|ㅈ|ㅊ|ㅋ|ㅌ|ㅍ|ㅎ)\s*[)\.]\s*")
RE_CHOICE_INLINE = re.compile(r"[\u2460-\u2473]")
RE_INLINE_TEX = re.compile(r"(\\\(.+?\\\))", re.S)
RE_HTML_TABLE = re.compile(r"<\s*table\b.*?</\s*table\s*>", re.I | re.S)
RE_ZERO_WIDTH = re.compile("[\u200B\u200C\u200D\u200E\u200F\u2060\u2066\u2067\u2068\u2069\ufeff\u00ad\u2028\u2029]")
RE_VIEW_UNIT = re.compile(r"(?<!\S)([㉠-㉯]|[ㄱ-ㅎ])\s*[)\.]?\s*")

# 플레이스홀더 탐지
PH_RE = re.compile(r"<<(?:IMG|FORM|TBL)_\d+>>")

# 원형숫자 ↔ 숫자 매핑
UC2NUM = {chr(0x2460 + i): str(i + 1) for i in range(20)}
UC2NUM.update({"\u24ea": "0"})
UC2NUM.update({chr(0x24F5 + i): str(i + 1) for i in range(10)})

NUM2UC = {"0": "\u24ea", **{str(i): chr(0x2460 + (i - 1)) for i in range(1, 21)}}
