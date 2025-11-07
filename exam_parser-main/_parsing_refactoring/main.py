# -*- coding: utf-8 -*-
from __future__ import annotations
import os, re, glob, json
from typing import List

from image_utils import ImageConfig
from config import (
    IMAGES_SRC_DIR, IMAGES_IMG_DIR, IMAGES_TBL_DIR, IMAGES_FORM_DIR, IMAGES_INDEX_DIR,
    IMAGE_MOVE, IMAGE_OVERWRITE, LAYOUT_DIR, FORMAT_DIR, ANSWER_DIR
)
from path_utils import ensure_dir, _abs_norm
from answers import find_answers_by_stem, load_json_or_jsonl
from extractor import extract
from build import build_kt_jsonl, IMAGE_LOGS

def main():
    ensure_dir(FORMAT_DIR)
    ensure_dir(IMAGES_IMG_DIR)
    ensure_dir(IMAGES_TBL_DIR)
    ensure_dir(IMAGES_FORM_DIR)
    ensure_dir(IMAGES_INDEX_DIR)

    IMG_CFG = ImageConfig(
        src_root=IMAGES_SRC_DIR,
        dst_img_root=IMAGES_IMG_DIR,
        dst_tbl_root=IMAGES_TBL_DIR,
        dst_form_root=IMAGES_FORM_DIR,
        move=IMAGE_MOVE,
        overwrite=IMAGE_OVERWRITE,
    )

    answers_by_stem = find_answers_by_stem(ANSWER_DIR)

    layout_paths = sorted(glob.glob(os.path.join(_abs_norm(LAYOUT_DIR), "*.json")))
    if not layout_paths:
        print(f"⚠️ 입력(.json) 파일이 없습니다: {LAYOUT_DIR}")

    total_q = 0
    total_no_choice = 0
    total_choice_parse_error = 0

    for LAYOUT_JSON in layout_paths:
        base = os.path.basename(LAYOUT_JSON)
        m = re.match(r"^(\d{4})", base)
        if not m:
            print(f"⏭️ 스킵(앞 4자리 숫자 없음): {base}")
            continue

        STEM = m.group(1)
        # 원본 로직 유지: _mathpix_merge 접미사 제거
        format_name = os.path.splitext(base)[0].removesuffix("_mathpix_merge")
        FORMAT_JSONL = os.path.join(FORMAT_DIR, f"{format_name}.jsonl")

        a_entry = answers_by_stem.get(STEM, {})
        answer_map = a_entry.get("ans") or {}
        explain_map = a_entry.get("exp") or {}
        answer_meta = a_entry.get("meta") or {}
        perq_detail = a_entry.get("perq") or {}
        ans_path = a_entry.get("path")

        answer_by_type = a_entry.get("ans_by_type") or {}
        explain_by_type = a_entry.get("exp_by_type") or {}
        perq_by_type = a_entry.get("perq_by_type") or {}

        answer_by_subject = a_entry.get("ans_by_subject") or {}
        explain_by_subject = a_entry.get("exp_by_subject") or {}
        perq_by_subject = a_entry.get("perq_by_subject") or {}
        answer_by_type_subject = a_entry.get("ans_by_type_subject") or {}
        explain_by_type_subject = a_entry.get("exp_by_type_subject") or {}
        perq_by_type_subject = a_entry.get("perq_by_type_subject") or {}

        try:
            raw = load_json_or_jsonl(LAYOUT_JSON)
            data = extract(raw)

            max_choice_in_file = max((len(q.choices or {}) for q in data), default=0)
            expected_max = min(max_choice_in_file, 5)
            choice_parse_errors = []
            if expected_max > 0:
                for q in data:
                    n_choices = len(q.choices or {})
                    if n_choices > 5 or n_choices < expected_max:
                        choice_parse_errors.append(q.number)

            total_q += len(data)
            total_choice_parse_error += len(choice_parse_errors)

            # 초기화(덮어쓰기): build_kt_jsonl 내부는 append로 쓰므로 파일 생성 전 비움
            if os.path.exists(FORMAT_JSONL):
                os.remove(FORMAT_JSONL)

            build_kt_jsonl(
                data, LAYOUT_JSON, FORMAT_JSONL,
                answer_map=answer_map,
                explain_map=explain_map,
                img_cfg=IMG_CFG,
                answer_meta=answer_meta,
                perq_detail=perq_detail,
                answer_by_type=answer_by_type,
                explain_by_type=explain_by_type,
                perq_by_type=perq_by_type,
                answer_by_subject=answer_by_subject,
                explain_by_subject=explain_by_subject,
                perq_by_subject=perq_by_subject,
                answer_by_type_subject=answer_by_type_subject,
                explain_by_type_subject=explain_by_type_subject,
                perq_by_type_subject=perq_by_type_subject,
            )

            if STEM in IMAGE_LOGS:
                idx_path = os.path.join(IMAGES_INDEX_DIR, f"{STEM}.json")
                ensure_dir(os.path.dirname(idx_path))
                with open(idx_path, "w", encoding="utf-8") as f:
                    json.dump(IMAGE_LOGS[STEM], f, ensure_ascii=False, indent=2)

            no_choice = [q.number for q in data if not q.choices]
            total_no_choice += len(no_choice)

            print(f"\n📄 처리 대상: {base}")
            print(f"   STEM: {STEM}")
            print(f"   ✅ 입력: {LAYOUT_JSON}")
            if ans_path:
                type_keys = ", ".join(sorted((answer_by_type or {}).keys()))
                print(f"   ✅ 적용(정답/해설): {ans_path} | 정답 {len(answer_map)}개, 해설 {len(explain_map)}개 | 시험유형 분포: [{type_keys}]")
            else:
                print(f"   ⚠️ 적용할 정답 파일 없음 (STEM={STEM})")
            print(f"   ✅ 저장(KT): {FORMAT_JSONL}")
            if max_choice_in_file > 0:
                print(f"   ✅ 기대 보기 수: {min(max_choice_in_file, 5)}지선다")
                print(f"   ⚠️ 선택지 파싱 오류: {len(choice_parse_errors)} -> {choice_parse_errors}")
            print(f"   ✅ 문항수: {len(data)} / 선택지 없는 문항: {len(no_choice)} -> {no_choice}")
            print(f"   📦 이미지 인덱스: {os.path.join(IMAGES_INDEX_DIR, f'{STEM}.json')}")
        except Exception as e:
            print(f"❌ 오류: {base} 처리 중 예외 발생: {e}")

    print(f"\n🎯 총 처리 파일: {len(layout_paths)} / 총 문항수: {total_q} / 선택지 없는 문항 수: {total_no_choice}")

if __name__ == "__main__":
    main()
