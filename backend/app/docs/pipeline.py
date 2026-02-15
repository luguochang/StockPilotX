from __future__ import annotations

import hashlib
import re
from typing import Any


class DocumentPipeline:
    """文档处理工程流水线（MVP）。

    步骤：
    1) 解析（按扩展名）
    2) 表格抽取（规则占位）
    3) 清洗去噪
    4) 分块
    5) 质量门禁（低置信度入复核队列）
    """

    def process(self, *, doc_id: str, filename: str, content: str, source: str) -> dict[str, Any]:
        ext = _ext(filename)
        parsed_text, confidence = self._parse(ext, content)
        tables = self._extract_tables(parsed_text)
        cleaned = self._clean(parsed_text)
        chunks = self._split(cleaned, chunk_size=900, overlap=120)
        needs_review = confidence < 0.7
        return {
            "doc_id": doc_id,
            "filename": filename,
            "source": source,
            "ext": ext,
            "doc_hash": _sha256(content),
            "parsed_text": parsed_text,
            "tables": tables,
            "cleaned_text": cleaned,
            "chunks": chunks,
            "parse_confidence": confidence,
            "needs_review": needs_review,
            "version": 1,
        }

    def _parse(self, ext: str, content: str) -> tuple[str, float]:
        if ext in (".html", ".htm"):
            text = re.sub(r"<[^>]+>", " ", content)
            return text, 0.9
        if ext == ".pdf":
            # MVP 环境无第三方解析器时，保留文本并标中等置信度
            return content, 0.65
        if ext == ".docx":
            return content, 0.72
        return content, 0.8

    def _extract_tables(self, text: str) -> list[dict[str, Any]]:
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        tables = []
        for line in lines:
            if "|" in line and len(line.split("|")) >= 3:
                cells = [c.strip() for c in line.split("|") if c.strip()]
                tables.append({"raw": line, "cells": cells})
        return tables

    def _clean(self, text: str) -> str:
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _split(self, text: str, chunk_size: int, overlap: int) -> list[str]:
        if not text:
            return []
        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = min(len(text), start + chunk_size)
            chunks.append(text[start:end])
            if end >= len(text):
                break
            start = end - overlap
        return chunks


def _ext(filename: str) -> str:
    idx = filename.rfind(".")
    return filename[idx:].lower() if idx >= 0 else ""


def _sha256(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8", errors="ignore")).hexdigest()

