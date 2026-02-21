from __future__ import annotations

from pathlib import Path

from .doc_convert_engine import DocConvertEngine
from .docling_engine import DoclingEngine
from .legacy_engine import LegacyParsingEngine
from .models import ParseResult


class DocumentParsingRouter:
    """Route uploaded bytes to parser engines with deterministic fallback."""

    def __init__(self, *, prefer_docling: bool = True) -> None:
        self._prefer_docling = bool(prefer_docling)
        self._docling = DoclingEngine()
        self._legacy = LegacyParsingEngine()
        self._doc_convert = DocConvertEngine()

    def parse(self, *, filename: str, raw_bytes: bytes, content_type: str = "") -> ParseResult:
        safe_name = str(filename or "uploaded.bin")
        ext = str(Path(safe_name).suffix or "").lower()
        notes: list[str] = []

        payload_bytes = raw_bytes
        payload_filename = safe_name

        if ext == ".doc":
            converted, note = self._doc_convert.convert_doc_to_docx(raw_bytes=raw_bytes, filename=safe_name)
            notes.append(note)
            if converted:
                payload_bytes = converted
                payload_filename = f"{Path(safe_name).stem}.docx"

        if self._prefer_docling and self._docling.available and self._docling.supports(filename=payload_filename):
            try:
                result = self._docling.extract(filename=payload_filename, raw_bytes=payload_bytes)
                if notes:
                    merged = [n for n in notes if n]
                    if result.parse_note:
                        merged.append(result.parse_note)
                    result.parse_note = ",".join(merged)
                    result.trace.notes = merged + [x for x in result.trace.notes if x not in merged]
                return result
            except Exception as ex:  # noqa: BLE001
                notes.append(f"docling_fallback:{str(ex)[:120]}")

        result = self._legacy.extract(filename=payload_filename, raw_bytes=payload_bytes, content_type=content_type)
        if notes:
            merged = [n for n in notes if n]
            if result.parse_note:
                merged.append(result.parse_note)
            result.parse_note = ",".join(merged)
            result.trace.notes = merged + [x for x in result.trace.notes if x not in merged]
        return result

