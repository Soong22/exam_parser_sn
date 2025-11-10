# -*- coding: utf-8 -*-
from __future__ import annotations
import os

# ===== 이미지/테이블/수식 저장 루트 =====
IMAGES_SRC_DIR = r"images"
IMAGES_IMG_DIR = os.path.join(IMAGES_SRC_DIR, "image")
IMAGES_TBL_DIR = os.path.join(IMAGES_SRC_DIR, "table")
IMAGES_FORM_DIR = os.path.join(IMAGES_SRC_DIR, "formula")
IMAGES_INDEX_DIR = os.path.join(IMAGES_SRC_DIR, "_index")

# 이동/덮어쓰기 스위치
IMAGE_MOVE = False
IMAGE_OVERWRITE = True

# ===== 입력/출력 경로 =====
LAYOUT_DIR = r"exam_parser-main\01_middle_process\data\2025_layout"
FORMAT_DIR = r"exam_parser-main\02_parsing\data\00_final\00_2025"
ANSWER_DIR = r"exam_parser-main\02_parsing\data\정답\수능정답파일.jsonl"
