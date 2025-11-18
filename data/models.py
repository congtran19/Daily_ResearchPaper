from dataclasses import dataclass
from datetime import date
from typing import Optional

@dataclass
class Paper:
    title: str
    authors: list[str]
    abstract: str
    url: str
    source: str       # ví dụ: "arXiv" hoặc "Semantic Scholar"
    published_date: Optional[date] = None
    embedding: Optional[list[float]] = None  # dùng cho RAG

@dataclass
class Summary:
    paper_title: str
    short_summary: str
    key_points: list[str]


@dataclass
class Digest:
    date: date
    papers: list[Paper]
    summaries: list[Summary]
