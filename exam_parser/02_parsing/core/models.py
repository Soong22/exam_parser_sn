# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List

@dataclass
class Question:
    number: str
    body: str = ""
    view: str = ""
    choices: Dict[str, str] = field(default_factory=dict)
    _images: List[Dict[str, Any]] = field(default_factory=list)
    _forms: List[Dict[str, Any]] = field(default_factory=list)
    _tables: List[Dict[str, Any]] = field(default_factory=list)
    _chunks: List[str] = field(default_factory=list)
    _round: int = 1
    _idx_in_round: int = 1
    _idx_global: int = 1
