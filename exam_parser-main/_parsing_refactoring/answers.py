# -*- coding: utf-8 -*-
from __future__ import annotations
import os, re, json, unicodedata
from typing import Any, Dict, List, Tuple
from path_utils import safe_glob, _abs_norm

def load_json_or_jsonl(path: str):
    if path.lower().endswith(".jsonl"):
        items = []
        with open(path, "r", encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                items.append(json.loads(line))
        return items
    else:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

def load_answer_bundle(path: str):
    def _k(s):
        return (s or "").replace(" ", "")

    ans_map: Dict[str, Any] = {}
    exp_map: Dict[str, str] = {}
    meta_top: Dict[str, Any] = {}
    perq_map: Dict[str, Dict[str, Any]] = {}

    if not path or not os.path.exists(path):
        return ans_map, exp_map, meta_top, perq_map

    def _ingest_obj(obj: Dict[str, Any]):
        for k in ["기출연도", "기출 연도", "시험명", "영역", "과목", "시험유형", "세부과목"]:
            if k in obj and obj[k] not in (None, ""):
                meta_top[_k(k)] = obj[k]
        if "source_url" in obj:
            val = obj["source_url"]
            if isinstance(val, str) and val.strip() and not meta_top.get("source_url"):
                meta_top["source_url"] = val.strip()

        raw_ans = obj.get("문제번호_정답") or obj.get("문제번호-정답") or obj.get("문제번호:정답")
        if isinstance(raw_ans, dict):
            for qk, v in raw_ans.items():
                sk = str(qk)
                if isinstance(v, dict):
                    if "정답" in v:
                        ans_map[sk] = v.get("정답")
                    sub = {}
                    for kk in ["난이도", "배점", "정답률", "문제유형"]:
                        if kk in v and v[kk] not in (None, ""):
                            sub[kk] = v[kk]
                    if sub:
                        perq_map.setdefault(sk, {}).update(sub)
                else:
                    ans_map[sk] = v
        elif isinstance(raw_ans, list):
            for item in raw_ans:
                if isinstance(item, dict):
                    for qk, v in item.items():
                        sk = str(qk)
                        if isinstance(v, dict):
                            if "정답" in v:
                                ans_map[sk] = v.get("정답")
                            sub = {}
                            for kk in ["난이도", "배점", "정답률", "문제유형"]:
                                if kk in v and v[kk] not in (None, ""):
                                    sub[kk] = v[kk]
                            if sub:
                                perq_map.setdefault(sk, {}).update(sub)
                        else:
                            ans_map[sk] = v

        raw_exp = obj.get("문제번호_해설") or obj.get("문제번호-해설") or obj.get("문제번호:해설")
        if isinstance(raw_exp, dict):
            for k, v in raw_exp.items():
                if v is not None and str(v).strip():
                    exp_map[str(k)] = str(v)
        elif isinstance(raw_exp, list):
            for item in raw_exp:
                if isinstance(item, dict):
                    for k, v in item.items():
                        if v is not None and str(v).strip():
                            exp_map[str(k)] = str(v)

    lower = path.lower()
    if lower.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8-sig") as f:
            for line in f:
                s = line.strip()
                if not s: continue
                try:
                    obj = json.loads(s)
                except Exception:
                    continue
                if isinstance(obj, dict):
                    _ingest_obj(obj)
    else:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return ans_map, exp_map, meta_top, perq_map

        if isinstance(data, dict):
            _ingest_obj(data)
        elif isinstance(data, list):
            for obj in data:
                if isinstance(obj, dict):
                    _ingest_obj(obj)

    return ans_map, exp_map, meta_top, perq_map

def collect_answer_bundles_auto(dir_hint: str, layout_dir: str) -> Dict[str, Dict[str, Any]]:
    mapping: Dict[str, Dict[str, Any]] = {}
    roots: List[str] = []
    if dir_hint and os.path.isdir(_abs_norm(dir_hint)):
        roots.append(dir_hint)
    ld = _abs_norm(layout_dir)
    ld_parent = os.path.dirname(ld)
    ld_grand = os.path.dirname(ld_parent)
    for r in [ld_parent, ld_grand, os.getcwd()]:
        if r and os.path.isdir(r):
            roots.append(r)

    seen_files: List[str] = []
    for root in roots:
        for pat in ["*-정답.jsonl", "*정답*.jsonl"]:
            found = safe_glob(root, pat, recursive=True)
            seen_files.extend(found)

    files = sorted({f for f in seen_files if f.lower().endswith(".jsonl")})

    for ap in files:
        base = os.path.basename(ap)
        m = re.match(r"^(\d{4})", base)
        if not m:
            continue
        stem = m.group(1)
        a_map, e_map, meta_top, perq_map = load_answer_bundle(ap)
        prev = mapping.get(stem)
        if prev:
            if base.endswith("-정답.jsonl") and not os.path.basename(prev["path"]).endswith("-정답.jsonl"):
                mapping[stem] = {"ans": a_map, "exp": e_map, "meta": meta_top, "perq": perq_map, "path": ap}
        else:
            mapping[stem] = {"ans": a_map, "exp": e_map, "meta": meta_top, "perq": perq_map, "path": ap}

    print(f"🔎 정답 파일 탐색 루트: {[ _abs_norm(r) for r in roots ]}")
    print(f"🔎 발견 정답 파일 수: {len(files)}")
    return mapping

def find_answers_by_stem(answer_path: str) -> Dict[str, Dict[str, Any]]:
    mapping: Dict[str, Dict[str, Any]] = {}
    if not answer_path:
        return mapping
    ap_norm = _abs_norm(answer_path)

    # 1) 단일 파일(json/jsonl): 유형/과목/세부과목까지 그룹핑
    if os.path.isfile(ap_norm):
        stems: Dict[str, Dict[str, Any]] = {}

        def _ingest_line_obj(obj: Dict[str, Any]):
            stem = str(obj.get("data_id", ""))[:4]
            if not (stem.isdigit() and len(stem) == 4):
                return
            exam_type = str(obj.get("시험유형", "") or "").strip()  # "홀" / "짝" / ""
            subj = str(obj.get("과목", "") or "").strip()
            subj2 = str(obj.get("세부과목", "") or "").strip()
            subj_key = f"{subj}|{subj2}" if (subj or subj2) else ""

            entry = stems.setdefault(stem, {
                "ans": {}, "exp": {}, "meta": {}, "perq": {},
                "ans_by_type": {}, "exp_by_type": {}, "perq_by_type": {},
                "ans_by_subject": {}, "exp_by_subject": {}, "perq_by_subject": {},
                "ans_by_type_subject": {}, "exp_by_type_subject": {}, "perq_by_type_subject": {},
                "path": ap_norm
            })

            for k in ["기출연도","시험명","영역","과목","시험유형","세부과목","source_url"]:
                if k in obj and obj[k] not in (None, ""):
                    entry["meta"][k] = obj[k]

            if exam_type:
                entry["ans_by_type"].setdefault(exam_type, {})
                entry["exp_by_type"].setdefault(exam_type, {})
                entry["perq_by_type"].setdefault(exam_type, {})
                if subj_key:
                    entry["ans_by_type_subject"].setdefault(exam_type, {}).setdefault(subj_key, {})
                    entry["exp_by_type_subject"].setdefault(exam_type, {}).setdefault(subj_key, {})
                    entry["perq_by_type_subject"].setdefault(exam_type, {}).setdefault(subj_key, {})

            raw_ans = obj.get("문제번호_정답")
            if isinstance(raw_ans, dict):
                for qk, v in raw_ans.items():
                    sk = str(qk)
                    if isinstance(v, dict) and "정답" in v:
                        entry["ans"][sk] = v.get("정답")
                        sub = {}
                        for kk in ["난이도","배점","정답률","문제유형"]:
                            if kk in v and v[kk] not in (None, ""):
                                sub[kk] = v[kk]
                        if sub:
                            entry["perq"].setdefault(sk, {}).update(sub)
                    else:
                        entry["ans"][sk] = v

                    if exam_type:
                        if isinstance(v, dict) and "정답" in v:
                            entry["ans_by_type"][exam_type][sk] = v.get("정답")
                            sub_t = {}
                            for kk in ["난이도","배점","정답률","문제유형"]:
                                if kk in v and v[kk] not in (None, ""):
                                    sub_t[kk] = v[kk]
                            if sub_t:
                                entry["perq_by_type"][exam_type].setdefault(sk, {}).update(sub_t)
                        else:
                            entry["ans_by_type"][exam_type][sk] = v

                        if subj_key:
                            if isinstance(v, dict) and "정답" in v:
                                entry["ans_by_type_subject"][exam_type][subj_key][sk] = v.get("정답")
                                if sub_t:
                                    entry["perq_by_type_subject"][exam_type][subj_key].setdefault(sk, {}).update(sub_t)
                            else:
                                entry["ans_by_type_subject"][exam_type][subj_key][sk] = v

                    if subj_key:
                        if isinstance(v, dict) and "정답" in v:
                            entry["ans_by_subject"].setdefault(subj_key, {})[sk] = v.get("정답")
                            sub_s = {}
                            for kk in ["난이도","배점","정답률","문제유형"]:
                                if kk in v and v[kk] not in (None, ""):
                                    sub_s[kk] = v[kk]
                            if sub_s:
                                entry["perq_by_subject"].setdefault(subj_key, {}).setdefault(sk, {}).update(sub_s)
                        else:
                            entry["ans_by_subject"].setdefault(subj_key, {})[sk] = v

            raw_exp = obj.get("문제번호_해설")
            if isinstance(raw_exp, dict):
                for k, v in raw_exp.items():
                    if v is not None and str(v).strip():
                        entry["exp"][str(k)] = str(v)
                        if exam_type:
                            entry["exp_by_type"][exam_type][str(k)] = str(v)
                            if subj_key:
                                entry["exp_by_type_subject"][exam_type][subj_key][str(k)] = str(v)
                        if subj_key:
                            entry["exp_by_subject"].setdefault(subj_key, {})[str(k)] = str(v)

        lower = ap_norm.lower()
        try:
            if lower.endswith(".jsonl"):
                with open(ap_norm, "r", encoding="utf-8-sig") as f:
                    for line in f:
                        s = line.strip()
                        if not s: continue
                        try:
                            obj = json.loads(s)
                            if isinstance(obj, dict):
                                _ingest_line_obj(obj)
                        except Exception:
                            pass
            else:
                with open(ap_norm, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    for obj in data:
                        if isinstance(obj, dict):
                            _ingest_line_obj(obj)
                elif isinstance(data, dict):
                    _ingest_line_obj(data)
        except Exception:
            return mapping

        for stem, bundle in stems.items():
            mapping[stem] = bundle
        return mapping

    # 2) 디렉터리: 우선순위만 적용(fallback)
    if not os.path.isdir(ap_norm):
        return mapping

    def _priority(name_nfk: str) -> int:
        if name_nfk.endswith("-정답.jsonl"): return 0
        if name_nfk.endswith("-정답.json"):  return 1
        if "정답" in name_nfk and name_nfk.endswith(".jsonl"): return 2
        if "정답" in name_nfk and name_nfk.endswith(".json"):  return 3
        return 9

    try:
        names = os.listdir(ap_norm)
    except Exception:
        return mapping

    for fn in names:
        norm = unicodedata.normalize("NFKC", fn)
        lower = norm.lower()
        if not (lower.endswith(".jsonl") or lower.endswith(".json")):
            continue
        if "정답" not in norm:
            continue
        m = re.match(r"^(\d{4})", norm)
        if not m:
            continue
        stem = m.group(1)
        ap = os.path.join(ap_norm, fn)
        ans_map, exp_map, meta_top, perq_map = load_answer_bundle(ap)
        prev = mapping.get(stem)
        cand = {"ans": ans_map, "exp": exp_map, "meta": meta_top, "perq": perq_map, "path": ap}
        if (not prev) or (_priority(norm) < _priority(os.path.basename(prev["path"]))):
            mapping[stem] = cand
    return mapping
