# table_utils.py

import re
from html.parser import HTMLParser
from typing import List, Tuple

_ALLOWED_TABLE_TAGS = {
    "table","thead","tbody","tfoot","tr","th","td","caption","colgroup","col","br"
}

# ✅ 허용 속성: 병합정보 + (옵션) 테두리
_ALLOWED_ATTRS = {
    "table": {"border"},    # <- 추가
    "td": {"rowspan", "colspan", "headers"},
    "th": {"rowspan", "colspan", "headers", "scope", "abbr"},
    "col": {"span"},
}

# ✅ 정수만 허용할 속성
_INT_ATTRS = {
    ("table", "border"),    # <- 추가
    ("td", "rowspan"), ("td", "colspan"),
    ("th", "rowspan"), ("th", "colspan"),
    ("col", "span"),
}

# 필요시 끄고 켤 수 있게 토글
FORCE_TABLE_BORDER = True   # True면 테두리 자동 추가, False면 원본에 있을 때만 통과

def _render_attrs(tag: str, attrs: List[Tuple[str, str]]) -> str:
    allowed = _ALLOWED_ATTRS.get(tag, set())
    out = []
    for k, v in (attrs or []):
        if v is None:
            continue
        k = (k or "").lower()
        if k not in allowed:
            continue
        if (tag, k) in _INT_ATTRS:
            try:
                vv = str(max(1, int(str(v).strip())))
            except Exception:
                continue
            out.append(f'{k}="{vv}"')
        else:
            vv = (str(v)
                  .replace("&", "&amp;")
                  .replace('"', "&quote;")
                  .replace("<", "&lt;"))
            out.append(f'{k}="{vv}"')
    return (" " + " ".join(out)) if out else ""

class _TableSkeletonParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out: List[str] = []
        self._table_depth = 0

    def handle_starttag(self, tag, attrs):
        t = tag.lower()
        if t == "table":
            self._table_depth += 1
        if t in _ALLOWED_TABLE_TAGS and self._table_depth > 0:
            if t == "br":
                self.out.append("<br/>")
                return
            attrs_html = _render_attrs(t, attrs)

            # ✅ 테두리 자동 부여(옵션)
            if t == "table" and FORCE_TABLE_BORDER and 'border="' not in attrs_html:
                attrs_html = (attrs_html + ' border="1"').rstrip()

            self.out.append(f"<{t}{attrs_html}>")

    def handle_startendtag(self, tag, attrs):
        t = tag.lower()
        if t in _ALLOWED_TABLE_TAGS and self._table_depth > 0:
            self.out.append(f"<{t}{_render_attrs(t, attrs)}/>")

    def handle_endtag(self, tag):
        t = tag.lower()
        if t in _ALLOWED_TABLE_TAGS and self._table_depth > 0 and t != "br":
            self.out.append(f"</{t}>")
        if t == "table" and self._table_depth > 0:
            self._table_depth -= 1

    def handle_data(self, data):
        if self._table_depth > 0 and data:
            self.out.append(data)

    def get_html(self) -> str:
        return "".join(self.out)

def sanitize_table_html(html_src: str) -> str:
    if not html_src:
        return ""
    try:
        p = _TableSkeletonParser()
        p.feed(html_src)
        return p.get_html()
    except Exception:
        s = re.sub(
            r"<(\w+)(\s+[^>]*)?>",
            lambda m: f"<{m.group(1).lower()}>" if m.group(1).lower() in _ALLOWED_TABLE_TAGS else "",
            html_src,
        )
        s = re.sub(
            r"</(?!table|thead|tbody|tfoot|tr|th|td|caption|colgroup|col|br)\w+\s*>",
            "",
            s,
            flags=re.I,
        )
        return s
