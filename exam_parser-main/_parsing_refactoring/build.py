# -*- coding: utf-8 -*-
from __future__ import annotations
import os, re, json
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

from image_utils import ImageConfig, classify_and_store_image
from table_utils import sanitize_table_html

from models import Question
from meta import _parse_meta_from_filename, _guess_category
from text_utils import clean_text
from constants import UC2NUM, NUM2UC, RE_CIRCLED_NUM

# 외부 공개(메인에서 이미지 인덱스 저장용)
IMAGE_LOGS: Dict[str, Dict[str, Any]] = {}

def _choice_order_key(k: str) -> int:
    ks = str(k)
    num = UC2NUM.get(ks)
    if num and num.isdigit(): return int(num)
    if ks.isdigit(): return int(ks)
    return 9999

def _make_content_from_addinfo(ai: Dict[str, Any]) -> str:
    body = ai.get("문제본문", "")
    view = ai.get("문제보기", "") or ""
    choices = ai.get("선택지") or ""
    answer_text = ai.get("정답", "") or ""
    explain_text = ai.get("해설", "") or ""

    lines = [f"[문제]: {body}".rstrip()]
    if view:
        lines[-1] += ("\n" + view)
    if isinstance(choices, dict) and choices:
        for k in sorted(choices.keys(), key=_choice_order_key):
            v = choices[k] or ""
            lines.append(f"{k}: {v}")
    lines += ["", f"[정답]: {answer_text}"]
    if explain_text:
        lines += ["", explain_text]
    return "\n".join(lines).strip()

def _choice_key_num_map(choice_keys) -> Dict[str, str]:
    mp: Dict[str, str] = {}
    for k in choice_keys:
        ch = str(k)
        num = UC2NUM.get(ch)
        if num and num not in mp:
            mp[num] = ch
    return mp

def _to_circled_string(val: Any, choice_keys) -> str:
    s = str(val)
    if re.search(RE_CIRCLED_NUM, s):
        return s
    num_map = _choice_key_num_map(choice_keys)
    has_circled_in_choices = bool(num_map)
    if not has_circled_in_choices:
        return s
    def repl(m: re.Match) -> str:
        num = m.group(1)
        return num_map.get(num, NUM2UC.get(num, num))
    return re.sub(r'(?<!\d)(\d{1,2})(?!\d)', repl, s)

def _size_or_default(sz: Optional[Dict[str, Any]]) -> Dict[str, Optional[int]]:
    ch = ht = wd = None

    def from_tuple(tp):
        nonlocal ch, ht, wd
        if not isinstance(tp, (list, tuple)):
            return
        if len(tp) == 2:
            w, h = tp[0], tp[1]
            wd, ht = int(w), int(h)
        elif len(tp) == 3:
            a, b, c = tp[0], tp[1], tp[2]
            if isinstance(a, int) and a <= 4:
                ch, ht, wd = int(a), int(b), int(c)
            elif isinstance(c, int) and c <= 4:
                ht, wd, ch = int(a), int(b), int(c)
            else:
                ht, wd, ch = int(a), int(b), int(c)

    if isinstance(sz, dict):
        ch = sz.get("channel") or sz.get("channels") or sz.get("c") or sz.get("depth")
        ht = sz.get("height") or sz.get("h") or sz.get("rows")
        wd = sz.get("width")  or sz.get("w") or sz.get("cols")
        if (ch is None or ht is None or wd is None):
            shp = sz.get("shape") or sz.get("size")
            if isinstance(shp, (list, tuple)):
                from_tuple(shp)
    elif isinstance(sz, (list, tuple)):
        from_tuple(sz)
    else:
        try:
            shp = getattr(sz, "shape", None) or getattr(sz, "size", None)
            if isinstance(shp, (list, tuple)):
                from_tuple(shp)
        except Exception:
            pass

    return {"channel": int(ch) if ch is not None else None,
            "height": int(ht) if ht is not None else None,
            "width":  int(wd) if wd is not None else None}

def _bbox_dict(bbox_list):
    b = bbox_list or []
    return {"x1": b[0] if len(b) >= 1 else None,
            "y1": b[1] if len(b) >= 2 else None,
            "x2": b[2] if len(b) >= 3 else None,
            "y2": b[3] if len(b) >= 4 else None}

def _should_use_rounds(meta_src_path: str) -> bool:
    base = os.path.basename(meta_src_path)
    name, _ = os.path.splitext(base)
    cond_subj = ("국어" in name) or ("수학" in name)
    has_holjjak = ("홀" in name) or ("짝" in name)
    has_munje = ("문제" in name)
    return cond_subj and (not has_holjjak) and has_munje

def _assign_indices(questions: List[Question], use_rounds: bool) -> None:
    round_no = 1
    idx_in_round = 0
    idx_global = 0
    first = True
    for q in questions:
        idx_global += 1
        if use_rounds:
            try:
                n = int(str(q.number).strip())
            except Exception:
                n = None
            if not first and n == 1:
                round_no += 1
                idx_in_round = 0
        else:
            round_no = 1
        first = False
        idx_in_round += 1
        q._round = round_no
        q._idx_in_round = idx_in_round
        q._idx_global = idx_global

def build_kt_jsonl(
    questions: List[Question],
    meta_src_path: str,
    out_jsonl_path: str,
    answer_map: Optional[Dict[str, Any]] = None,
    explain_map: Optional[Dict[str, str]] = None,
    img_cfg: ImageConfig = None,
    answer_meta: Optional[Dict[str, Any]] = None,
    perq_detail: Optional[Dict[str, Dict[str, Any]]] = None,
    answer_by_type: Optional[Dict[str, Dict[str, Any]]] = None,
    explain_by_type: Optional[Dict[str, Dict[str, str]]] = None,
    perq_by_type: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None,
    answer_by_subject: Optional[Dict[str, Dict[str, Any]]] = None,
    explain_by_subject: Optional[Dict[str, Dict[str, str]]] = None,
    perq_by_subject: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None,
    answer_by_type_subject: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None,
    explain_by_type_subject: Optional[Dict[str, Dict[str, Dict[str, str]]]] = None,
    perq_by_type_subject: Optional[Dict[str, Dict[str, Dict[str, Dict[str, Any]]]]] = None,
):
    use_rounds = _should_use_rounds(meta_src_path)
    _assign_indices(questions, use_rounds)

    meta = _parse_meta_from_filename(meta_src_path)
    base_id = meta.get("data_id", "")
    fallback_year    = meta.get("year", "")
    fallback_exam    = meta.get("exam", "")
    fallback_subject = meta.get("subject", "")
    fallback_examtype = meta.get("type", "")
    data_title = meta.get("data_title", "")
    collected_date = datetime.now().strftime("%Y.%m.%d")

    def _m(meta_obj, key):
        if not meta_obj: return ""
        return meta_obj.get(key, "") or ""

    a_year     = _m(answer_meta, "기출연도") or fallback_year
    a_exam     = _m(answer_meta, "시험명")   or fallback_exam
    a_area     = _m(answer_meta, "영역")     or ""
    a_subject  = _m(answer_meta, "과목")     or fallback_subject
    a_type     = _m(answer_meta, "시험유형") or ""
    a_subj2    = _m(answer_meta, "세부과목") or ""

    data_title = "-".join([s for s in [a_year, a_area, a_subj2] if s]) \
                 or "-".join([s for s in [fallback_year, "", fallback_subject] if s])

    asset_seq = 1
    max_choice_in_file = max((len(q.choices or {}) for q in questions), default=0)
    uniform_question_type = f"{max_choice_in_file}지선다" if max_choice_in_file > 0 else "서술형"

    def _lookup(map_obj, qnum: str):
        if not map_obj: return None
        cand = [qnum, qnum.lstrip("0") or qnum]
        if qnum.isdigit(): cand.append(int(qnum))
        for c in cand:
            if c in map_obj: return map_obj[c]
        return None

    def _sorted_choice_keys(keys):
        return sorted(keys, key=_choice_order_key)

    def _round_to_examtype(r: int) -> Optional[str]:
        return "홀" if r == 1 else ("짝" if r == 2 else None)

    def _best_subject_key_for_range(
        needed_nums: List[int],
        exam_type_key: Optional[str],
        prefer_keys: List[str],
        fallback_any: bool = True
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]], Optional[Dict[str, str]], Optional[Dict[str, Dict[str, Any]]]]:
        needed_set = set(str(n) for n in needed_nums)
        best = (None, None, None, None, -1)
        if exam_type_key and answer_by_type_subject and exam_type_key in (answer_by_type_subject or {}):
            type_bucket_a = answer_by_type_subject.get(exam_type_key, {})
            type_bucket_e = (explain_by_type_subject or {}).get(exam_type_key, {})
            type_bucket_p = (perq_by_type_subject or {}).get(exam_type_key, {})
            cand_keys = list(type_bucket_a.keys())
            ordered = [k for k in prefer_keys if k in cand_keys] + [k for k in cand_keys if k not in prefer_keys]
            for k in ordered:
                amap = type_bucket_a.get(k, {})
                cover = len(needed_set & set(amap.keys()))
                if cover > best[4]:
                    best = (k, amap, type_bucket_e.get(k, {}), type_bucket_p.get(k, {}), cover)
            if best[1] is not None and best[4] > 0:
                return best[0], best[1], best[2], best[3]

        if answer_by_subject:
            cand_keys = list(answer_by_subject.keys())
            ordered = [k for k in prefer_keys if k in cand_keys] + [k for k in cand_keys if k not in prefer_keys]
            for k in ordered:
                amap = answer_by_subject.get(k, {})
                cover = len(needed_set & set(amap.keys()))
                if cover > best[4]:
                    best = (k, amap, (explain_by_subject or {}).get(k, {}), (perq_by_subject or {}).get(k, {}), cover)
            if best[1] is not None and (best[4] > 0 or fallback_any):
                return best[0], best[1], best[2], best[3]
        return None, None, None, None

    is_korean_file = ("국어" in (fallback_subject or "")) or ("국어" in (a_subject or ""))
    is_math_file   = ("수학" in (fallback_subject or "")) or ("수학" in (a_subject or ""))

    korean_35_45_block_count = 0
    math_23_30_block_count   = 0
    prev_qnum_int: Optional[int] = None

    with open(out_jsonl_path, "w", encoding="utf-8") as fw:
        for q in questions:
            body_pre = q.body or ""
            view_pre = q.view or ""
            choices_pre: Dict[str, str] = q.choices or {}

            base_id_with_global = f"{base_id}_{q._idx_global:04d}"

            try:
                qnum_int = int(str(q.number).strip())
            except Exception:
                qnum_int = q._idx_in_round

            if use_rounds:
                q_data_id = f"{base_id}_{q._round:04d}_{qnum_int:04d}"
            else:
                q_data_id = f"{base_id}_{qnum_int:04d}"

            combined_text = "\n".join(
                [body_pre, view_pre] + [choices_pre[k] for k in _sorted_choice_keys(choices_pre.keys())]
            )

            ph_order: List[Tuple[str, int]] = []
            seen = set()
            for m in re.finditer(r"<<(IMG|FORM|TBL)_(\d+)>>", combined_text):
                key = (m.group(1), int(m.group(2)))
                if key not in seen:
                    ph_order.append(key); seen.add(key)

            content_meta: Dict[str, Any] = {}
            ph_to_tag: Dict[Tuple[str, int], str] = {}
            qnum = str(q.number).strip()

            for kind, idx in ph_order:
                tag = f"tag_{base_id_with_global}_{asset_seq:04d}"
                ph_to_tag[(kind, idx)] = tag

                if kind == "IMG":
                    img = (q._images or [])[idx] if idx < len(q._images or []) else {}
                    bbox = img.get("bbox") or []
                    src_hint = (img.get("file_name")
                                or img.get("image_path")
                                or img.get("img_path")
                                or img.get("src")
                                or "")
                    res = classify_and_store_image(src_hint, base_id_with_global or "0000", tag=tag, kind="img", cfg=img_cfg)
                    if isinstance(res, tuple) and len(res) == 3:
                        new_path, sz, src_abs = res
                    else:
                        new_path, sz = res; src_abs = ""
                    info = {
                        "type": "image",
                        "bbox": _bbox_dict(bbox),
                        "title": f"{q_data_id}_그림",
                        "file_name": new_path or src_hint or "",
                        "img_size": _size_or_default(sz),
                    }
                    content_meta[tag] = info

                    if new_path:
                        src_key = os.path.basename(src_abs) if src_abs else os.path.basename(new_path)
                        IMAGE_LOGS.setdefault(base_id or "0000", {})[src_key] = {
                            "src": src_abs or "", "dst": new_path, "size": sz, "kind": "img", "tag": tag,
                        }

                elif kind == "FORM":
                    form = (q._forms or [])[idx] if idx < len(q._forms or []) else {}
                    latex_with_delims = form.get("latex", "") or ""
                    bbox = form.get("bbox") or []
                    src_hint = form.get("image_path") or form.get("img_path") or form.get("file_name") or ""
                    new_path = ""; sz = {}; src_abs = ""
                    if src_hint:
                        res = classify_and_store_image(src_hint, base_id_with_global or "0000", tag=tag, kind="form", cfg=img_cfg)
                        if isinstance(res, tuple) and len(res) == 3:
                            new_path, sz, src_abs = res
                        else:
                            new_path, sz = res; src_abs = ""
                    info = {
                        "type": "formula",
                        "bbox": _bbox_dict(bbox),
                        "text": sanitize_table_html(latex_with_delims[2:-2].strip()) if latex_with_delims.startswith(r"\(") and latex_with_delims.endswith(r"\)") else latex_with_delims,
                        "info": "latex",
                        "file_name": new_path or src_hint or "",
                        "img_size": _size_or_default(sz),
                    }
                    content_meta[tag] = info

                    if new_path:
                        src_key = os.path.basename(src_abs) if src_abs else os.path.basename(new_path)
                        IMAGE_LOGS.setdefault(base_id or "0000", {})[src_key] = {
                            "src": src_abs or "", "dst": new_path, "size": sz, "kind": "form", "tag": tag,
                        }

                elif kind == "TBL":
                    tbl = (q._tables or [])[idx] if idx < len(q._tables or []) else {}
                    bbox = tbl.get("bbox") or []
                    src_hint = (tbl.get("file_name")
                                or tbl.get("image_path")
                                or tbl.get("table_img_path")
                                or tbl.get("img_path")
                                or "")
                    res = classify_and_store_image(src_hint, base_id_with_global or "0000", tag=tag, kind="tbl", cfg=img_cfg)
                    if isinstance(res, tuple) and len(res) == 3:
                        new_path, sz, src_abs = res
                    else:
                        new_path, sz = res; src_abs = ""
                    info = {
                        "type": "table",
                        "bbox": _bbox_dict(bbox),
                        "text": sanitize_table_html(tbl.get("html", "") or ""),
                        "info": "html",
                        "file_name": new_path or src_hint or "",
                        "img_size": _size_or_default(sz),
                    }
                    content_meta[tag] = info

                    if new_path:
                        src_key = os.path.basename(src_abs) if src_abs else os.path.basename(new_path)
                        IMAGE_LOGS.setdefault(base_id or "0000", {})[src_key] = {
                            "src": src_abs or "", "dst": new_path, "size": sz, "kind": "tbl", "tag": tag,
                        }

                asset_seq += 1

            def replace_ph(txt: str) -> str:
                if not txt: return txt
                def repl(m: re.Match) -> str:
                    k = (m.group(1), int(m.group(2)))
                    tag = ph_to_tag.get(k, "")
                    return f"<{tag}>" if tag else ""
                return re.sub(r"<<(IMG|FORM|TBL)_(\d+)>>", repl, txt)

            replaced_body = replace_ph(body_pre)
            replaced_view = replace_ph(view_pre)
            replaced_choices = {str(k): replace_ph(v) for k, v in choices_pre.items()}
            choices_for_addinfo = {k: clean_text(v or "") for k, v in replaced_choices.items()}

            dtype_set = set(["text"])
            for v in content_meta.values():
                t = v.get("type")
                if t in ("image", "formula", "table"): dtype_set.add(t)
            data_types = sorted(dtype_set)

            ans_maps = answer_map or {}
            exp_maps = explain_map or {}
            perq_maps = perq_detail or {}

            if use_rounds:
                key = _round_to_examtype(q._round)
                if key and answer_by_type and key in answer_by_type and answer_by_type[key]:
                    ans_maps = answer_by_type[key]
                if key and explain_by_type and key in explain_by_type and explain_by_type[key]:
                    exp_maps = explain_by_type[key]
                if key and perq_by_type and key in perq_by_type and key in perq_by_type and perq_by_type[key]:
                    perq_maps = perq_by_type[key]

            override_subject = None
            exam_type_key = _round_to_examtype(q._round) if use_rounds else None

            if prev_qnum_int is not None:
                if is_korean_file and qnum_int == 35 and (prev_qnum_int is None or prev_qnum_int != 35):
                    korean_35_45_block_count += 1
                if is_math_file and qnum_int == 23 and (prev_qnum_int is None or prev_qnum_int != 23):
                    math_23_30_block_count += 1
            else:
                if is_korean_file and qnum_int == 35:
                    korean_35_45_block_count = 1
                if is_math_file and qnum_int == 23:
                    math_23_30_block_count = 1

            if is_korean_file:
                if 1 <= qnum_int <= 34:
                    override_subject = ("국어", "국어")
                    need = list(range(1, 35))
                    _, a_map2, e_map2, p_map2 = _best_subject_key_for_range(
                        need, exam_type_key, prefer_keys=["화법과작문|화법과작문","언어와매체|언어와매체"], fallback_any=True
                    )
                    if a_map2:
                        ans_maps, exp_maps, perq_maps = a_map2, (e_map2 or {}), (p_map2 or {})
                elif 35 <= qnum_int <= 45:
                    if korean_35_45_block_count <= 1:
                        override_subject = ("화법과작문", "화법과작문")
                        need = list(range(35, 46))
                        _, a_map2, e_map2, p_map2 = _best_subject_key_for_range(
                            need, exam_type_key, prefer_keys=["화법과작문|화법과작문"], fallback_any=False
                        )
                        if a_map2:
                            ans_maps, exp_maps, perq_maps = a_map2, (e_map2 or {}), (p_map2 or {})
                    else:
                        override_subject = ("언어와매체", "언어와매체")
                        need = list(range(35, 46))
                        _, a_map2, e_map2, p_map2 = _best_subject_key_for_range(
                            need, exam_type_key, prefer_keys=["언어와매체|언어와매체"], fallback_any=False
                        )
                        if a_map2:
                            ans_maps, exp_maps, perq_maps = a_map2, (e_map2 or {}), (p_map2 or {})

            if is_math_file:
                if 1 <= qnum_int <= 22:
                    override_subject = ("수학", "수학")
                    need = list(range(1, 23))
                    _, a_map2, e_map2, p_map2 = _best_subject_key_for_range(
                        need, exam_type_key,
                        prefer_keys=["수학|확률과통계","수학|미적분","수학|기하"],
                        fallback_any=True
                    )
                    if a_map2:
                        ans_maps, exp_maps, perq_maps = a_map2, (e_map2 or {}), (p_map2 or {})
                elif 23 <= qnum_int <= 30:
                    if math_23_30_block_count <= 1:
                        override_subject = ("수학", "확률과통계"); want = "수학|확률과통계"
                    elif math_23_30_block_count == 2:
                        override_subject = ("수학", "미적분");       want = "수학|미적분"
                    else:
                        override_subject = ("수학", "기하");         want = "수학|기하"
                    need = list(range(23, 31))
                    _, a_map2, e_map2, p_map2 = _best_subject_key_for_range(
                        need, exam_type_key, prefer_keys=[want], fallback_any=False
                    )
                    if a_map2:
                        ans_maps, exp_maps, perq_maps = a_map2, (e_map2 or {}), (p_map2 or {})

            ans_val = _lookup(ans_maps, qnum)

            def _ans_text(v, choice_keys) -> str:
                if v is None:
                    return ""
                if isinstance(v, dict):
                    inner = v.get("정답", v)
                    if isinstance(inner, (list, tuple)):
                        return ",".join(_to_circled_string(x, choice_keys) for x in inner)
                    return _to_circled_string(inner, choice_keys)
                if isinstance(v, (list, tuple)):
                    return ",".join(_to_circled_string(x, choice_keys) for x in v)
                return _to_circled_string(v, choice_keys)

            answer_text = _ans_text(ans_val, replaced_choices.keys())

            explain_text = ""
            ev = _lookup(exp_maps, qnum)
            if ev is not None:
                explain_text = str(ev).strip()

            pdet = (perq_maps or {}).get(qnum, {}) if perq_maps else {}
            diff  = pdet.get("난이도", "")
            score = pdet.get("배점", "")
            rate  = pdet.get("정답률", "")
            qtype = uniform_question_type

            out_subject = a_subject
            out_subj2   = a_subj2
            if override_subject:
                out_subject, out_subj2 = override_subject

            data_title_q = "-".join([s for s in [a_year, a_area, (out_subj2 or out_subject or fallback_subject)] if s]) \
                        or "-".join([s for s in [fallback_year, "", fallback_subject] if s])

            ai = {
                "문제번호": qnum,
                "기출연도": a_year,
                "시험명": a_exam,
                "영역": a_area,
                "과목": out_subject,
                **({"시험유형": _round_to_examtype(q._round)} if use_rounds else {"시험유형": a_type} if a_type else {}),
                **({"세부과목": out_subj2} if out_subj2 else {}),
                **({"문제유형": qtype} if qtype != "" else {}),
                "문제본문": replaced_body,
                **({"문제보기": replaced_view} if replaced_view else {}),
                "선택지": {k: clean_text(v or "") for k, v in replaced_choices.items()},
                "정답": answer_text,
                **({"해설": explain_text} if explain_text else {}),
                **({"난이도": diff} if diff != "" else {}),
                **({"배점": score} if score != "" else {}),
                **({"정답률": rate} if rate != "" else {}),
            }

            content = _make_content_from_addinfo(ai)

            rec = {
                "data_id": q_data_id,
                "data_file": os.path.basename(out_jsonl_path),
                "data_title": data_title_q,
            }

            if answer_meta and "source_url" in answer_meta and answer_meta["source_url"]:
                rec["source_url"] = answer_meta["source_url"]

            cat_main, cat_sub = _guess_category(out_subject)
            rec.update({
                "category_main": cat_main,
                **({"category_sub": cat_sub} if cat_sub else {}),
                "data_type": data_types,
                "collected_date": collected_date,
                "content": content,
                **({"content_meta": content_meta} if content_meta else {}),
                "add_info": ai,
            })

            with open(out_jsonl_path, "a", encoding="utf-8") as _fw_append:
                _fw_append.write(json.dumps(rec, ensure_ascii=False) + "\n")

            prev_qnum_int = qnum_int
