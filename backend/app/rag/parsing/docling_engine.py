from __future__ import annotations

import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any

from .models import ParseQuality, ParseResult, ParseTrace


class DoclingEngine:
    """Optional parser backed by Docling.

    This engine is best-effort. If runtime dependency is unavailable or conversion fails,
    callers should fallback to legacy parser.
    """

    _SUPPORTED_EXT = {
        ".pdf",
        ".docx",
        ".pptx",
        ".xlsx",
        ".xlsm",
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".tiff",
        ".webp",
    }

    def __init__(self) -> None:
        self._converter_cls: Any | None = None
        self._load_error = ""
        try:
            from docling.document_converter import DocumentConverter  # type: ignore

            self._converter_cls = DocumentConverter
        except Exception as ex:  # noqa: BLE001
            self._load_error = str(ex)
            self._converter_cls = None

    @property
    def available(self) -> bool:
        return self._converter_cls is not None

    @property
    def load_error(self) -> str:
        return self._load_error

    def supports(self, *, filename: str) -> bool:
        ext = str(Path(filename).suffix or "").lower()
        return ext in self._SUPPORTED_EXT

    def extract(self, *, filename: str, raw_bytes: bytes) -> ParseResult:
        if not self.available:
            raise RuntimeError(f"docling_not_available:{self._load_error}")
        started = time.perf_counter()
        ext = str(Path(filename).suffix or "").lower()
        converter = self._converter_cls()  # type: ignore[operator]
        notes: list[str] = []
        ocr_used = ext in {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
        with tempfile.NamedTemporaryFile(prefix="rag-docling-", suffix=ext or ".bin", delete=False) as tmp:
            tmp.write(raw_bytes)
            tmp_path = Path(tmp.name)
        try:
            result = converter.convert(str(tmp_path))
            text = self._extract_text_from_docling_result(result)
            if not text.strip():
                notes.append("docling_empty_text")
            else:
                notes.append("docling_extract")
            duration_ms = int((time.perf_counter() - started) * 1000)
            normalized = re.sub(r"\s+", " ", str(text or "")).strip()
            coverage = min(1.0, len(normalized) / 3000.0)
            garbled_count = sum(1 for ch in normalized if ch in {"�", "?"})
            garbled_ratio = (garbled_count / max(1, len(normalized))) if normalized else 0.0
            ocr_conf = 0.8 if ocr_used and normalized else (0.0 if ocr_used else 1.0)
            quality_score = max(0.0, min(1.0, coverage * 0.65 + (1.0 - garbled_ratio) * 0.35))
            trace = ParseTrace(
                parser_name="docling",
                parser_version="1",
                ocr_used=ocr_used,
                duration_ms=duration_ms,
                notes=notes,
            )
            quality = ParseQuality(
                text_coverage_ratio=coverage,
                garbled_ratio=garbled_ratio,
                ocr_confidence_avg=ocr_conf,
                quality_score=quality_score,
            )
            return ParseResult(
                plain_text=text,
                parse_note=",".join(notes),
                trace=trace,
                quality=quality,
            )
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:  # noqa: BLE001
                pass

    @staticmethod
    def _extract_text_from_docling_result(result: Any) -> str:
        if result is None:
            return ""
        # Docling ConversionResult usually has `.document`.
        document = getattr(result, "document", None)
        if document is not None:
            for method_name in ("export_to_text", "to_text"):
                fn = getattr(document, method_name, None)
                if callable(fn):
                    try:
                        payload = fn()
                        if payload is not None:
                            text = str(payload)
                            if text.strip():
                                return text
                    except Exception:  # noqa: BLE001
                        continue
            for method_name in ("export_to_markdown", "to_markdown"):
                fn = getattr(document, method_name, None)
                if callable(fn):
                    try:
                        payload = fn()
                        if payload is not None:
                            text = str(payload)
                            if text.strip():
                                return text
                    except Exception:  # noqa: BLE001
                        continue
            # Last resort: flatten document object string.
            text = str(document)
            if text.strip():
                return text

        for attr in ("text", "markdown", "content"):
            value = getattr(result, attr, None)
            if value is not None:
                text = str(value)
                if text.strip():
                    return text
        return str(result or "")

