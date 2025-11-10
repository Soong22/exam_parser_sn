# -*- coding: utf-8 -*-
from __future__ import annotations
import json, re, math
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from collections import defaultdict

# =========================
# 하드코딩 경로/옵션
# =========================
IN_DIR         = Path(r"exam_parser-main\01_middle_process\data\merge_html")  # 입력 폴더
FILE_GLOB      = "**/*.json"          # JSON 배열 파일만 대상
OVERWRITE_INPLACE = False             # ✅ 기본: 원본 보존 (True로 하면 원본 덮어씀)
OUT_DIR        = Path(r"exam_parser-main\01_middle_process\data\layout")      # 결과 저장 폴더
PRESERVE_TREE  = True                 # 하위 경로 구조 보존
VERBOSE        = True                 # 진행 로그

# 라운드 분기 조건: "lte"(같거나 작아지면 새 라운드) / "lt"(작아질 때만 새 라운드)
ROUND_TRIGGER_MODE = "lte"

# 2단 정렬 옵션
COLUMN_MODE = "auto"                  # "auto" | "single"
MIN_GAP_PX = 60                       # 좌/우 칼럼 중심 간 최소 간격
MIN_CLUSTER_PCT = 0.2                 # 작은 클러스터 최소 비율

# 헤더 판정 여백 허용치(px): (페이지,칼럼) 좌측 여백 + 허용치 이내만 헤더로 인정
HEADER_INDENT_TOL = 25

# 디버깅 태그 출력
DEBUG_ANNOTATE = False

# =========================
# 문항 헤더 패턴
# =========================
RE_Q_HEADER = re.compile(
    r"^\s*(?:문\s*)?([1-9]\d{0,2})\s*(?:\)|[.。．])\s*",
    re.UNICODE
)

def norm_text(s: str) -> str:
    if not s:
        return ""
    s = s.replace("\u200b", "").replace("\u200c", "").replace("\u200d", "")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _bbox(b: Dict[str, Any]) -> Tuple[float,float,float,float]:
    x1,y1,x2,y2 = (b.get("bbox") or [0,0,0,0])[:4]
    return float(x1), float(y1), float(x2), float(y2)

# =========================
# 2단 읽기 순서 만들기
# =========================
def _two_means_x1(xs: List[float], iters: int = 8) -> Tuple[float,float]:
    xs = sorted(xs)
    if not xs:
        return 0.0, 1.0
    c1 = xs[len(xs)//4]
    c2 = xs[(len(xs)*3)//4]
    for _ in range(iters):
        g1 = [x for x in xs if abs(x-c1) <= abs(x-c2)]
        g2 = [x for x in xs if abs(x-c1) >  abs(x-c2)]
        if not g1 or not g2:
            break
        c1 = sum(g1)/len(g1)
        c2 = sum(g2)/len(g2)
    if c1 > c2:
        c1, c2 = c2, c1
    return c1, c2

def reading_order_indices(blocks: List[Dict[str, Any]]) -> List[int]:
    """
    페이지별로 좌/우 칼럼을 판정하여 좌→우, 각 칼럼 y오름차순으로 블록 인덱스 반환.
    텍스트/비텍스트 가리지 않고 전체 블록을 대상으로 순서만 만든다.
    """
    if COLUMN_MODE != "auto":
        # 단일 칼럼: page → y → x
        order = sorted(range(len(blocks)),
                       key=lambda i: (int(blocks[i].get("page_idx",0)),
                                      _bbox(blocks[i])[1],
                                      _bbox(blocks[i])[0],
                                      i))
        return order

    # 페이지별 그룹
    pages: Dict[int, List[int]] = defaultdict(list)
    for i, b in enumerate(blocks):
        pages[int(b.get("page_idx",0))].append(i)

    order: List[int] = []
    for p in sorted(pages.keys()):
        idxs = pages[p]
        if len(idxs) < 6:
            order += sorted(idxs, key=lambda i: (_bbox(blocks[i])[1], _bbox(blocks[i])[0], i))
            continue

        x1s = [ _bbox(blocks[i])[0] for i in idxs ]
        c1, c2 = _two_means_x1(x1s)
        gap = abs(c2 - c1)

        # 칼럼 할당
        left_idxs, right_idxs = [], []
        for i in idxs:
            x1 = _bbox(blocks[i])[0]
            if abs(x1 - c1) <= abs(x1 - c2):
                left_idxs.append(i)
            else:
                right_idxs.append(i)

        small = min(len(left_idxs), len(right_idxs))
        if gap < MIN_GAP_PX or small < len(idxs) * MIN_CLUSTER_PCT:
            # 칼럼이 애매하면 단일 칼럼 처리
            order += sorted(idxs, key=lambda i: (_bbox(blocks[i])[1], _bbox(blocks[i])[0], i))
        else:
            # 좌→우, 각 칼럼 y→x
            order += sorted(left_idxs,  key=lambda i: (_bbox(blocks[i])[1], _bbox(blocks[i])[0], i))
            order += sorted(right_idxs, key=lambda i: (_bbox(blocks[i])[1], _bbox(blocks[i])[0], i))
    return order

# (page,col)별 좌측 여백 계산
def column_left_margins(blocks: List[Dict[str, Any]], order: List[int]) -> Dict[Tuple[int,int], float]:
    """
    읽기 순서에 따라 각 페이지를 좌/우 칼럼으로 다시 분할하고,
    (page, col) 별 최소 x1을 좌측 여백으로 기록.
    col: 0=left, 1=right (단일 칼럼인 경우 모두 0으로 본다)
    """
    margins: Dict[Tuple[int,int], float] = {}
    # 페이지별로 다시 모아 x1 클러스터링
    pages: Dict[int, List[int]] = defaultdict(list)
    for i in order:
        pages[int(blocks[i].get("page_idx",0))].append(i)

    for p in pages:
        idxs = pages[p]
        x1s = [ _bbox(blocks[i])[0] for i in idxs ]
        if COLUMN_MODE == "single" or len(idxs) < 6:
            for i in idxs:
                x1 = _bbox(blocks[i])[0]
                margins[(p,0)] = min(margins.get((p,0), x1), x1) if (p,0) in margins else x1
            continue

        c1, c2 = _two_means_x1(x1s)
        gap = abs(c2 - c1)
        if gap < MIN_GAP_PX:
            for i in idxs:
                x1 = _bbox(blocks[i])[0]
                margins[(p,0)] = min(margins.get((p,0), x1), x1) if (p,0) in margins else x1
        else:
            # 칼럼 할당
            for i in idxs:
                x1 = _bbox(blocks[i])[0]
                col = 0 if abs(x1 - c1) <= abs(x1 - c2) else 1
                margins[(p,col)] = min(margins.get((p,col), x1), x1) if (p,col) in margins else x1
    return margins

def detect_header_qnum(block: Dict[str, Any], margin_map: Dict[Tuple[int,int], float]) -> Optional[int]:
    if not isinstance(block, dict) or block.get("type") != "text":
        return None
    text = norm_text(block.get("text", "") or "")
    m = RE_Q_HEADER.match(text)
    if not m:
        return None
    page = int(block.get("page_idx",0))
    x1 = _bbox(block)[0]

    # 칼럼 결정: 가장 가까운 (page, col) margin 을 찾는다
    candidates = [((page, col), margin_map[(page,col)]) for (pp,col) in margin_map.keys() if pp == page]
    if not candidates:
        return int(m.group(1))  # fallback
    # col 선택
    best = min(candidates, key=lambda kv: abs(x1 - kv[1]))
    col_margin = best[1]
    if x1 <= col_margin + HEADER_INDENT_TOL:
        return int(m.group(1))
    return None

# =========================
# 라운드/문항 블록 수집 (2단 읽기 순서 사용)
# =========================
class QuestionBlock:
    def __init__(self, qnum: int, first_idx: int):
        self.qnum = qnum
        self.indices: List[int] = [first_idx]
        self.first_idx = first_idx

class RoundBundle:
    def __init__(self):
        self.preamble: List[int] = []
        self.questions: List[QuestionBlock] = []

def _should_start_new_round(cur_started: bool, last_qnum: Optional[int], qnum: int) -> bool:
    if not cur_started or last_qnum is None:
        return False
    if ROUND_TRIGGER_MODE == "lt":
        return qnum < last_qnum
    return qnum <= last_qnum

def collect_rounds(blocks: List[Dict[str, Any]]) -> Tuple[List[RoundBundle], List[int]]:
    """
    읽기 순서를 기준으로 라운드/문항을 수집.
    반환: (rounds, linear_order)
      - rounds: 라운드 묶음
      - linear_order: 2단 정렬을 반영한 전체 블록 인덱스 순서
    """
    order = reading_order_indices(blocks)
    margin_map = column_left_margins(blocks, order)

    rounds: List[RoundBundle] = []
    cur_round: Optional[RoundBundle] = None
    cur_q: Optional[QuestionBlock] = None
    last_qnum_in_round: Optional[int] = None
    started_round = False

    def start_new_round():
        nonlocal cur_round, cur_q, last_qnum_in_round, started_round
        if cur_round is not None:
            if cur_q is not None:
                cur_round.questions.append(cur_q)
            rounds.append(cur_round)
        cur_round = RoundBundle()
        cur_q = None
        last_qnum_in_round = None
        started_round = False

    start_new_round()

    for i in order:
        b = blocks[i]
        qnum = detect_header_qnum(b, margin_map)
        if qnum is not None:
            if _should_start_new_round(started_round, last_qnum_in_round, qnum):
                start_new_round()
            if cur_q is not None:
                cur_round.questions.append(cur_q)
            cur_q = QuestionBlock(qnum, i)
            last_qnum_in_round = qnum
            started_round = True
        else:
            if cur_q is not None:
                cur_q.indices.append(i)
            else:
                cur_round.preamble.append(i)

    if cur_q is not None:
        cur_round.questions.append(cur_q)
    if cur_round is not None:
        rounds.append(cur_round)

    return rounds, order

# =========================
# 정렬 & 재배열
# =========================
def reorder_by_rounds(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rounds, lin_order = collect_rounds(blocks)
    if not rounds:
        # 그래도 2단 읽기 순서만 반영
        return [blocks[i] for i in lin_order]

    new_order: List[int] = []
    for r in rounds:
        # 라운드 내 문항을 번호 오름차순(동번호는 first_idx)으로 정렬
        r.questions.sort(key=lambda q: (q.qnum, q.first_idx))
        # preamble 유지
        new_order.extend(r.preamble)
        for q in r.questions:
            new_order.extend(q.indices)

    # 누락 방지
    seen = set(new_order)
    if len(seen) != len(blocks):
        for i in lin_order:
            if i not in seen:
                new_order.append(i); seen.add(i)

    reordered = [blocks[i] for i in new_order]

    # 디버깅 태그(선택)
    if DEBUG_ANNOTATE:
        # 라운드/문항 태깅
        tag_map: Dict[int, Tuple[int, Optional[int]]] = {}
        ridx = 1
        for r in rounds:
            for bi in r.preamble:
                tag_map[bi] = (ridx, None)
            for q in r.questions:
                for bi in q.indices:
                    tag_map[bi] = (ridx, q.qnum)
            ridx += 1
        for orig_idx, item in zip(new_order, reordered):
            if isinstance(item, dict) and orig_idx in tag_map:
                rr, qq = tag_map[orig_idx]
                item["_round"] = rr
                item["_qnum"] = qq

    return reordered

# =========================
# I/O
# =========================
def iter_input_files() -> List[Path]:
    return sorted([p for p in IN_DIR.glob(FILE_GLOB) if p.is_file()], key=lambda x: str(x).lower())

def out_path_for(src: Path) -> Path:
    rel_parent = src.relative_to(IN_DIR).parent if PRESERVE_TREE else Path()
    return OUT_DIR / rel_parent / f"{src.stem}.json"

def process_one_file(src: Path) -> Tuple[int, int]:
    try:
        txt = src.read_text(encoding="utf-8")
    except Exception:
        txt = src.read_text(encoding="utf-8", errors="ignore")

    try:
        blocks = json.loads(txt)
        if not isinstance(blocks, list):
            if VERBOSE: print(f"[SKIP] 배열 JSON 아님: {src}")
            return (0, 0)
    except Exception:
        if VERBOSE: print(f"[SKIP] JSON 파싱 실패: {src}")
        return (0, 0)

    if not blocks:
        return (0, 0)

    reordered = reorder_by_rounds(blocks)

    if OVERWRITE_INPLACE:
        dst = src
    else:
        dst = out_path_for(src)
        dst.parent.mkdir(parents=True, exist_ok=True)

    with dst.open("w", encoding="utf-8") as f:
        json.dump(reordered, f, ensure_ascii=False, indent=2)

    return (len(blocks), len(reordered))

def main():
    files = iter_input_files()
    if not files:
        print(f"[WARN] 입력 없음: {IN_DIR} ({FILE_GLOB})")
        return

    total_in = total_out = 0
    for i, src in enumerate(files, 1):
        nin, nout = process_one_file(src)
        total_in += nin; total_out += nout
        if VERBOSE:
            where = "in-place" if OVERWRITE_INPLACE else str(out_path_for(src))
            print(f"[{i:04d}/{len(files)}] {src} -> {where} (blocks {nin} → {nout})")

    print(f"\n[SUMMARY] files={len(files)}, blocks={total_in} → {total_out}")

if __name__ == "__main__":
    main()
