from __future__ import annotations
import json
from pathlib import Path
from typing import Any, List

# =========================
# 하드코딩 경로/옵션 (필요시 수정)
# =========================
IN_DIR         = Path(r"exam_parser-main\02_parsing\data\정답\2025_직업탐구영역_정답_개선본")       # JSON 파일들이 모여 있는 폴더
FILE_GLOB      = "*.json"                       # 스캔할 패턴
OUT_JSONL      = Path(r"exam_parser-main\02_parsing\data\정답\2025_직업탐구영역_정답_개선본.jsonl")
RECURSIVE      = False                          # 하위폴더까지 검색하려면 True
SORT_INPUT_BY  = "name"                         # "name" | "mtime"

# =========================

def find_json_files(root: Path, pattern: str, recursive: bool) -> List[Path]:
    if recursive:
        files = sorted(root.rglob(pattern))
    else:
        files = sorted(root.glob(pattern))
    if SORT_INPUT_BY == "mtime":
        files.sort(key=lambda p: p.stat().st_mtime)
    elif SORT_INPUT_BY == "name":
        files.sort(key=lambda p: p.name)
    return [p for p in files if p.is_file()]

def try_parse_whole(text: str) -> Any:
    """파일 전체를 한 방에 JSON으로 파싱 시도."""
    return json.loads(text)

def iter_jsonl_lines(text: str):
    """JSONL처럼 줄마다 JSON 파싱을 시도."""
    for ln, line in enumerate(text.splitlines(), start=1):
        s = line.strip()
        if not s:
            continue
        try:
            yield json.loads(s)
        except json.JSONDecodeError:
            # JSONL 형식이 아니면 건너뜀
            continue

def dump_jsonl_line(fp, obj: Any):
    fp.write(json.dumps(obj, ensure_ascii=False) + "\n")

def main():
    in_dir: Path = IN_DIR
    out_path: Path = OUT_JSONL
    out_path.parent.mkdir(parents=True, exist_ok=True)

    files = find_json_files(in_dir, FILE_GLOB, RECURSIVE)
    total_files = len(files)
    written = 0
    skipped_files = 0
    parsed_as_jsonl = 0
    parsed_as_array = 0
    parsed_as_object = 0

    if total_files == 0:
        print(f"[INFO] 대상 파일이 없습니다: {in_dir} ({FILE_GLOB})")
        return

    with out_path.open("w", encoding="utf-8", newline="\n") as out_fp:
        for f in files:
            try:
                text = f.read_text(encoding="utf-8")
            except Exception as e:
                print(f"[WARN] 파일 읽기 실패: {f} ({e})")
                skipped_files += 1
                continue

            # 1) 전체 JSON으로 파싱 시도
            try:
                data = try_parse_whole(text)
                if isinstance(data, list):
                    for item in data:
                        dump_jsonl_line(out_fp, item)
                        written += 1
                    parsed_as_array += 1
                elif isinstance(data, dict):
                    dump_jsonl_line(out_fp, data)
                    written += 1
                    parsed_as_object += 1
                else:
                    # 숫자/문자열/기타 => 한 줄로 기록
                    dump_jsonl_line(out_fp, data)
                    written += 1
                    parsed_as_object += 1
                continue
            except json.JSONDecodeError:
                # 2) JSONL 라인별 파싱 시도
                any_line = False
                for obj in iter_jsonl_lines(text):
                    dump_jsonl_line(out_fp, obj)
                    written += 1
                    any_line = True
                if any_line:
                    parsed_as_jsonl += 1
                    continue

                # 3) 완전히 실패
                print(f"[WARN] JSON 형식 인식 실패, 건너뜀: {f}")
                skipped_files += 1

    print("==== Merge Summary ====")
    print(f"- 입력 폴더     : {in_dir}")
    print(f"- 출력 파일     : {out_path}")
    print(f"- 스캔 파일 수  : {total_files}")
    print(f"- 기록 라인 수  : {written}")
    print(f"- 배열로 파싱   : {parsed_as_array} 파일")
    print(f"- 객체로 파싱   : {parsed_as_object} 파일")
    print(f"- JSONL로 파싱  : {parsed_as_jsonl} 파일")
    print(f"- 건너뛴 파일   : {skipped_files} 파일")

if __name__ == "__main__":
    main()
