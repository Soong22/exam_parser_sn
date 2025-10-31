# -*- coding: utf-8 -*-
"""
choice_process 폴더의 JSON들을 읽어
- 블록 순서를 그대로 유지하면서
- 연속된 text 블록들만 공백 1개(" ")로 합쳐서 하나의 text 블록으로 축약하고
- inline_equation, interline_equation, image, table 등 비-텍스트 블록은 그대로 보존
한 새 JSON을 폴더 구조를 유지해 저장합니다.

출력 예)
[ text, inline_equation, text, image, text ... ]
여기서 연속된 text 들은 각각 1개로 병합되어 나갑니다.
"""

from pathlib import Path
import argparse
import json
import unicodedata
from typing import Iterable, Union, List, Dict, Any
import copy

# ===== 기본 경로 (필요시 CLI로 덮어쓰기 가능) =====
DEFAULT_JSON_DIR_IN = Path(r"exam_parser-main\01_middle_process\data\choice_process")
DEFAULT_OUT_DIR     = Path(r"exam_parser-main\01_middle_process\data\text_coalesced")

# 텍스트 병합 시 구분자: 공백 1개
MERGE_SEP = " "

# 비-텍스트 블록 타입 (그대로 보존)
NON_TEXT_TYPES = {
    "inline_equation",
    "interline_equation",
    "image",
    "table",
}

# 입력 파일 확장자
IN_EXT = ".json"


# ---------- 유틸 ----------
def nfc(s: str) -> str:
    """한글/결합문자 정규화(NFC)."""
    return unicodedata.normalize("NFC", s or "")


def _as_block_list(jdata: Union[List[Dict[str, Any]], Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    다양한 JSON 루트를 '블록 리스트'로 변환.
    - 리스트면 그대로
    - {"blocks": [...]} 있으면 그 리스트
    - 단일 블록(dict) 이면 [dict]
    - 그 외는 빈 리스트
    """
    if isinstance(jdata, list):
        return [b for b in jdata if isinstance(b, dict)]
    if isinstance(jdata, dict):
        if isinstance(jdata.get("blocks"), list):
            return [b for b in jdata["blocks"] if isinstance(b, dict)]
        return [jdata] if "type" in jdata else []
    return []


def _flush_text_buffer(buf: List[str], out_blocks: List[Dict[str, Any]]):
    """버퍼에 모인 text 조각들을 공백 1칸으로 합쳐 하나의 text 블록으로 푸시."""
    # 앞뒤 공백 제거 + 빈 조각 제거
    pieces = [nfc(t).strip() for t in buf if (t or "").strip()]
    if not pieces:
        buf.clear()
        return
    # 경계는 정확히 공백 1개
    merged = pieces[0]
    for p in pieces[1:]:
        if not merged.endswith(MERGE_SEP):
            merged += MERGE_SEP
        merged += p.lstrip()
    out_blocks.append({"type": "text", "text": merged})
    buf.clear()


def coalesce_text_blocks(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    입력 블록 시퀀스를 순회하며:
    - 연속된 text 블록은 병합하여 하나의 text 블록으로
    - non-text 블록(NON_TEXT_TYPES)은 그대로 out 시퀀스에 복사
    - non-text 블록은 병합 경계로 동작(앞/뒤 텍스트는 분리 병합)
    """
    out: List[Dict[str, Any]] = []
    text_buf: List[str] = []

    for blk in blocks:
        btype = blk.get("type")
        if btype == "text":
            text_buf.append(blk.get("text", ""))
            continue

        # non-text 또는 기타 타입을 만나면: 지금까지 모은 text를 먼저 flush
        _flush_text_buffer(text_buf, out)

        # 원본 블록은 그대로 보존 (깊은 복사로 원본 변형 방지)
        out.append(copy.deepcopy(blk))

    # 종료 후 남은 텍스트 flush
    _flush_text_buffer(text_buf, out)
    return out


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------- 핵심 처리 ----------
def process_one_file(json_path: Path, out_root: Path) -> Path:
    """
    하나의 입력 JSON을 순서 보존 + 연속 text 병합한 블록 리스트로 저장.
    출력 경로는 입력 폴더 구조를 보존하고, 파일명에 _text_coalesced suffix를 붙인다.
    """
    jdata = read_json(json_path)
    blocks = _as_block_list(jdata)
    out_blocks = coalesce_text_blocks(blocks)

    # 출력 파일명: <원본스텀>_text_coalesced.json
    # 입력 루트(json_path.parents[0]) 아래 폴더 구조 보존
    out_dir = out_root / json_path.relative_to(json_path.parents[0]).parent
    out_name = f"{json_path.stem}_text_coalesced.json"
    out_path = out_dir / out_name

    write_json(out_path, out_blocks)
    return out_path


def folder_main(json_dir_in: Path, out_dir: Path) -> None:
    """
    입력 폴더 전체 재귀 순회하여 *.json 처리.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(p for p in json_dir_in.rglob(f"*{IN_EXT}") if p.is_file())
    if not json_files:
        print(f"[WARN] 입력 JSON이 없습니다: {json_dir_in}")
        return

    total = 0
    for jp in json_files:
        total += 1
        try:
            out_path = process_one_file(jp, out_dir)
            print(f"[WRITE] {out_path}")
        except Exception as e:
            print(f"[ERROR] 처리 실패: {jp}\n   -> {e}")

    print(f"\n[DONE] 총 {total}개 처리 완료. 출력 루트: {out_dir}")


# ---------- CLI ----------
def parse_args():
    ap = argparse.ArgumentParser(
        description="JSON 블록 순서를 유지하며 연속된 text 블록만 공백 1개로 병합합니다 (비-텍스트 블록은 그대로 보존)."
    )
    ap.add_argument(
        "--in",
        dest="json_dir_in",
        type=Path,
        default=DEFAULT_JSON_DIR_IN,
        help=f"입력 폴더 루트 (기본: {DEFAULT_JSON_DIR_IN})",
    )
    ap.add_argument(
        "--out",
        dest="out_dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"출력 폴더 루트 (기본: {DEFAULT_OUT_DIR})",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    folder_main(args.json_dir_in, args.out_dir)


if __name__ == "__main__":
    main()
