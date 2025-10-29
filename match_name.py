# -*- coding: utf-8 -*-
from __future__ import annotations
import json, re, shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import unicodedata as ud
from collections import defaultdict

# =========================
# 설정
# =========================
JSONL_PATH = Path(r"total_structure.jsonl")
TARGET_DIR = Path(r"수능기출문제/middle/2025/문제")
DRY_RUN = False
YEAR_FILTER = "2025"

# 형 포함/미포함 파일명 포맷
NAME_FMT_NOFORM   = "{root_id}-{year}-{area}-{subject}-문제"
NAME_FMT_WITHFORM = "{root_id}-{year}-{area}-{subject}-{form}-문제"

ALLOW_EXTS = {".json"}

# =========================
# 과목 규칙 (키는 가능하면 JSONL '시험명'과 동일)
# =========================
SUBJECT_RULES: Dict[str, List[str]] = {
    # 수학(구버전)
    "수리_가형": ["수리_가형", "수학영역_가형", "수학 가형", "가형"],
    "수리_나형": ["수리_나형", "수학영역_나형", "수학 나형", "나형"],

    # 탐구 공통
    "생활과 윤리": ["생활과 윤리"],
    "윤리와 사상": ["윤리와 사상"],
    "사회문화": ["사회·문화", "사회문화"],
    "세계사": ["세계사"],
    "동아시아사": ["동아시아사"],
    "세계지리": ["세계지리"],
    "한국지리": ["한국지리"],
    "경제": ["경제", "사탐(경제)"],
    "정치": ["정치와 법", "정치"],

    # 직탐
    "공업일반": ["공업 일반"],
    "인간발달": ["인간 발달"],
    "성공적인직업생활": ["성공적인 직업생활", "성공적인직업생활"],
    "농업기초기술": ["농업 기초 기술", "농업기초기술"],
    "농업이해": ["직탐(농업 이해)", "농업 이해"],
    "기초제도": ["직탐(기초 제도)", "기초 제도"],
    "회계원리": ["직탐(회계 원리)", "회계 원리"],
    "생활서비스산업의이해": ["직탐(생활 서비스 산업의 이해)", "생활 서비스 산업의 이해"],
    "수산해운산업기초": ["수산해운산업기초", "수산·해운 산업 기초", "수산해운 산업기초", "수산해운", "수산", "해운"],
    "해양의이해": ["직탐(해양의 이해)", "해양의 이해"],
    "상업경제": ["직탐(상업 경제)", "상업 경제", "상업경제"],

    # 과탐
    "물리1": ["물리학Ⅰ", "물리1", "과탐(물리학 I)"],
    "물리2": ["물리학Ⅱ", "물리2", "과탐(물리학 II)"],
    "화학1": ["화학Ⅰ", "화학1", "과탐(화학 I)"],
    "화학2": ["화학Ⅱ", "화학2", "과탐(화학 II)"],
    "생명과학1": ["생명과학Ⅰ", "생명과학1", "과탐(생명과학 I)"],
    "생명과학2": ["생명과학Ⅱ", "생명과학2", "과탐(생명과학 II)"],
    "지구과학1": ["지구과학Ⅰ", "지구과학1", "과탐(지구과학 I)"],
    "지구과학2": ["지구과학Ⅱ", "지구과학2", "과탐(지구과학 II)"],

    # 공통/영역
    "한국사": ["한국사", "한국사영역"],
    "국어영역": ["국어영역", "국어"],   # 경계매칭으로 '중국어'와 충돌 방지
    "수학영역": ["수학영역", "수학"],
    "영어": ["영어영역", "영어"],

    # 제2외국어
    "독일어": ["독일어Ⅰ", "독일어"],
    "프랑스어": ["프랑스어Ⅰ", "프랑스어"],
    "러시아어": ["러시아어Ⅰ", "러시아어", "러시아Ⅰ"],
    "중국어": ["중국어Ⅰ", "중국어"],
    "일본어": ["일본어Ⅰ", "일본어"],
    "스페인어": ["스페인어Ⅰ", "스페인어"],
    "아랍어": ["아랍어Ⅰ", "아랍어"],
    "베트남어Ⅰ": ["베트남어Ⅰ", "베트남어"],
    "한문": ["한문Ⅰ", "한문"],
}

# ✅ 형(홀/짝) 규칙 — '홀수형/짝수형'을 직접 인식하도록 추가
FORM_RULES: Dict[str, List[str]] = {
    "홀수": ["홀수형", "홀수", "홀"],
    "짝수": ["짝수형", "짝수", "짝"],
}

# 영역 힌트 → JSONL '영역' 문자열
AREA_HINTS: Dict[str, List[str]] = {
    "사회탐구영역": ["사탐", "사회탐구영역", "사회탐구"],
    "과학탐구영역": ["과탐", "과학탐구영역", "과학탐구"],
    "직업탐구영역": ["직탐", "직업탐구영역", "직업탐구"],
    "제2외국어":   ["제2외국어"],
    "국어영역":    ["국어영역", "국어"],
    "수학영역":    ["수학영역", "수학"],
    "영어영역":    ["영어영역", "영어"],
    "한국사영역":  ["한국사영역", "한국사"],
}

# =========================
# 유틸
# =========================
def nfc(s: str) -> str:
    return ud.normalize("NFC", s or "")

def stem_without_ext(p: Path) -> str:
    return nfc(p.stem)

def compact(s: str) -> str:
    return re.sub(r"[\s_·]+", "", nfc(s))

def safe_dest(dest: Path) -> Path:
    """동일 파일명 존재 시 _1, _2 ... 붙여서 충돌 회피"""
    if not dest.exists():
        return dest
    base, ext = dest.stem, dest.suffix
    i = 1
    while True:
        cand = dest.with_name(f"{base}_{i}{ext}")
        if not cand.exists():
            return cand
        i += 1

# =========================
# 경계 기반 정규식 유틸
# =========================
_WORD_BOUNDARY = r'(?<![0-9A-Za-z가-힣]){pat}(?![0-9A-Za-z가-힣])'

def _compile_variant_regex(variant: str):
    v = nfc(variant)
    compact_v = compact(v)
    r1 = re.compile(_WORD_BOUNDARY.format(pat=re.escape(v)))
    r2 = re.compile(_WORD_BOUNDARY.format(pat=re.escape(compact_v)))
    return r1, r2

def _sorted_candidates_from_rules(rules: Dict[str, List[str]]):
    """(길이 내림차순, canonical, r1, r2) 목록 생성"""
    cand = []
    for canonical, variants in rules.items():
        for v in list(variants) + [canonical]:
            r1, r2 = _compile_variant_regex(v)
            cand.append((len(v), canonical, r1, r2))
    cand.sort(key=lambda x: -x[0])  # 긴 패턴 우선
    return cand

SUBJECT_CANDIDATES = _sorted_candidates_from_rules(SUBJECT_RULES)
FORM_CANDIDATES    = _sorted_candidates_from_rules(FORM_RULES)

def detect_subject(filename_stem: str) -> Optional[str]:
    f = nfc(filename_stem); fc = compact(f)
    for _, canonical, r1, r2 in SUBJECT_CANDIDATES:
        if r1.search(f) or r2.search(fc):
            return canonical
    return None

def detect_form(filename_stem: str) -> Optional[str]:
    f = nfc(filename_stem); fc = compact(f)
    for _, form, r1, r2 in FORM_CANDIDATES:
        if r1.search(f) or r2.search(fc):
            return form
    return None

def detect_area_hint(filename_stem: str) -> Optional[str]:
    f = nfc(filename_stem); fc = compact(f)
    hint_cands = []
    for area, hints in AREA_HINTS.items():
        for h in hints + [area]:
            r1, r2 = _compile_variant_regex(h)
            hint_cands.append((len(h), area, r1, r2))
    hint_cands.sort(key=lambda x: -x[0])
    for _, area, r1, r2 in hint_cands:
        if r1.search(f) or r2.search(fc):
            return area
    return None

# =========================
# 데이터 로드
# =========================
def load_items_for_year(jsonl_path: Path, year: str) -> List[Dict]:
    items: List[Dict] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            info = obj.get("structure", {}).get("시험정보", {})
            if nfc(info.get("년도", "")) == year:
                items.append(obj)
    return items

# =========================
# 메인
# =========================
def main():
    print("=== Heuristic Match & Rename (경계기반, 충돌감지, 홀/짝, 영역힌트) ===")
    print(f"- JSONL_PATH : {JSONL_PATH}")
    print(f"- TARGET_DIR : {TARGET_DIR}")
    print(f"- YEAR_FILTER: {YEAR_FILTER}")
    print(f"- DRY_RUN    : {DRY_RUN}")

    assert JSONL_PATH.exists(), f"JSONL이 없음: {JSONL_PATH}"
    assert TARGET_DIR.exists(), f"대상 폴더가 없음: {TARGET_DIR}"

    records = load_items_for_year(JSONL_PATH, YEAR_FILTER)
    print(f"[INFO] JSONL {YEAR_FILTER} 항목 수: {len(records)}")

    # 인덱스 구성
    subj_to_recs: Dict[str, List[Dict]] = defaultdict(list)
    subj_form_to_rec: Dict[Tuple[str, str], Dict] = {}

    for rec in records:
        info = rec["structure"]["시험정보"]
        subj = nfc(info.get("시험명", ""))
        form = nfc(info.get("형", "") or "")
        if subj:
            subj_to_recs[subj].append(rec)
            if form:
                subj_form_to_rec[(subj, form)] = rec

    files = [p for p in TARGET_DIR.iterdir() if p.is_file() and p.suffix in ALLOW_EXTS]
    print(f"[INFO] 대상 폴더 내 파일 수: {len(files)}")

    plan: List[Tuple[Path, Path, str, Optional[str]]] = []
    unmatched: List[str] = []

    for src in files:
        name = stem_without_ext(src)

        subj = detect_subject(name)
        form = detect_form(name)  # "홀수"/"짝수" 또는 None
        area_hint = detect_area_hint(name)  # 영역 힌트 (있으면 후보 좁힘)

        if not subj:
            unmatched.append(src.name)
            continue

        rec: Optional[Dict] = None

        # 1) (과목,형) 완전 일치 우선
        if form:
            rec = subj_form_to_rec.get((subj, form))

        # 2) 과목만 일치 → 영역 힌트가 있으면 같은 영역만 선택
        if rec is None:
            candidates = subj_to_recs.get(subj, [])
            if area_hint:
                candidates = [r for r in candidates if nfc(r["structure"]["시험정보"].get("영역", "")) == area_hint]
            if candidates:
                # 동일 과목/연도/영역이면 실질적으로 동일 → 첫 번째 사용
                rec = candidates[0]

        if rec is None:
            unmatched.append(src.name)
            continue

        info = rec["structure"]["시험정보"]
        rid  = rec["root_id"]
        year = info["년도"]; area = info["영역"]; subj_canon = info["시험명"]
        form_canon = info.get("형", "") or (form or "")

        if form_canon:
            dst_stem = NAME_FMT_WITHFORM.format(
                root_id=rid, year=year, area=area, subject=subj_canon, form=form_canon
            )
        else:
            dst_stem = NAME_FMT_NOFORM.format(
                root_id=rid, year=year, area=area, subject=subj_canon
            )

        dst = src.with_name(dst_stem + src.suffix)
        dst = safe_dest(dst)  # 최종 충돌 회피
        plan.append((src, dst, subj_canon, form_canon or None))

    # 목적지 스템 충돌 사전 감지
    stem_to_sources = defaultdict(list)
    for src, dst, subj_canon, form_canon in plan:
        stem_to_sources[dst.stem].append(src.name)
    conflicts = {stem: names for stem, names in stem_to_sources.items() if len(names) > 1}

    print("\n=== Rename Plan ===")
    for src, dst, subj_canon, form_canon in plan:
        tag = subj_canon + (f"/{form_canon}" if form_canon else "")
        print(f"[{tag}] {src.name}  -->  {dst.name}")

    if conflicts:
        print("\n[WARN] 목적지 파일명(확장자 제외) 충돌 감지:")
        for stem, names in conflicts.items():
            print(f"  - {stem}  <= {', '.join(names)}")
        print("    ※ safe_dest 로 실제 파일 충돌은 자동 회피됩니다. (접미사 _1, _2 부여)")
        print("    ※ 하지만 형(홀/짝) 탐지 문제는 이제 해결됐는지 로그로 확인해 주세요.")

    print("\n=== Summary ===")
    print(f"- 스캔 파일 수 : {len(files)}")
    print(f"- 매칭된 파일  : {len(plan)}")
    print(f"- 미매칭 파일  : {len(unmatched)}")
    if unmatched:
        for name in unmatched:
            print(f"  · {name}")

    if not DRY_RUN:
        ok = 0; fail = 0
        for src, dst, _, _ in plan:
            try:
                shutil.move(str(src), str(dst)); ok += 1
            except Exception as e:
                print(f"[ERROR] rename 실패: {src.name} -> {dst.name} ({e})"); fail += 1
        print("\n=== Apply Result ===")
        print(f"- 성공: {ok}")
        print(f"- 실패: {fail}")
        print("[DONE] 실제 변경 완료")

if __name__ == "__main__":
    main()
