# -*- coding: utf-8 -*-
from __future__ import annotations

import re
import shutil
from pathlib import Path
import unicodedata
import sys

# =============================
# 하드코딩 경로 (원하면 여길 수정)
# =============================
INPUT_DIR = Path(r"수능기출문제\middle\2025")   # 분류할 폴더 경로

# 생성될 하위 폴더명
ANS_DIRNAME = "정답"
PROB_DIRNAME = "문제"

# 인식할 확장자 (요청은 json이지만, 관례상 jsonl도 함께 처리)
EXTS = {".json", ".jsonl"}

# 파일명이 이런 패턴으로 끝나면 '정답'으로 분류
# 예: foo_정답.json, foo-정답지.jsonl, foo 정답.json
ANSWER_SUFFIX_RE = re.compile(r"(?:[_\-\s])(정답_middle|정답지_middle|정답표_middle|듣기대본_middle|듣기평가대본_middle)$", re.IGNORECASE)


def nfc(s: str) -> str:
    """한글 파일명 정규화 (NFC)."""
    return unicodedata.normalize("NFC", s)


def is_answer_file(stem: str) -> bool:
    """
    확장자를 제외한 파일명(stem)을 보고
    끝자리가 _정답 / -정답 / 공백정답 / _정답지 등인지 판단.
    """
    s = nfc(stem)
    return bool(ANSWER_SUFFIX_RE.search(s))


def safe_move(src: Path, dst_dir: Path) -> Path:
    """
    이름 충돌나면 (1), (2) 붙여서 안전하게 이동.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    target = dst_dir / src.name
    if not target.exists():
        return src.rename(target)

    base = target.stem
    ext = target.suffix
    i = 1
    while True:
        cand = dst_dir / f"{base} ({i}){ext}"
        if not cand.exists():
            return src.rename(cand)
        i += 1


def main():
    # 옵션: 커맨드라인 인자로 경로를 주면 그걸 우선 사용
    root = INPUT_DIR
    if len(sys.argv) > 1:
        root = Path(sys.argv[1])

    if not root.exists() or not root.is_dir():
        print(f"[ERR] 폴더가 없어요: {root}")
        sys.exit(1)

    ans_dir = root / ANS_DIRNAME
    prob_dir = root / PROB_DIRNAME

    moved_ans, moved_prob = 0, 0

    for p in sorted(root.iterdir()):
        # 하위 폴더는 건너뛰고, 파일만 처리
        if not p.is_file():
            continue

        # 확장자 필터
        if p.suffix.lower() not in EXTS:
            continue

        # 분류
        if is_answer_file(p.stem):
            safe_move(p, ans_dir)
            moved_ans += 1
        else:
            safe_move(p, prob_dir)
            moved_prob += 1

    print(f"[DONE] 정답: {moved_ans}개 이동  |  문제: {moved_prob}개 이동")
    print(f" - 정답 폴더: {ans_dir}")
    print(f" - 문제 폴더: {prob_dir}")


if __name__ == "__main__":
    main()
