from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ParseTrace:
    parser_name: str
    parser_version: str = ""
    ocr_used: bool = False
    duration_ms: int = 0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "parser_name": self.parser_name,
            "parser_version": self.parser_version,
            "ocr_used": bool(self.ocr_used),
            "duration_ms": int(max(0, self.duration_ms)),
            "notes": list(self.notes),
        }


@dataclass(slots=True)
class ParseQuality:
    text_coverage_ratio: float = 0.0
    garbled_ratio: float = 0.0
    ocr_confidence_avg: float = 0.0
    quality_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "text_coverage_ratio": float(max(0.0, min(1.0, self.text_coverage_ratio))),
            "garbled_ratio": float(max(0.0, min(1.0, self.garbled_ratio))),
            "ocr_confidence_avg": float(max(0.0, min(1.0, self.ocr_confidence_avg))),
            "quality_score": float(max(0.0, min(1.0, self.quality_score))),
        }


@dataclass(slots=True)
class ParseResult:
    plain_text: str
    parse_note: str
    trace: ParseTrace
    quality: ParseQuality
    blocks: list[dict[str, Any]] = field(default_factory=list)
    pages: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "plain_text": str(self.plain_text or ""),
            "parse_note": str(self.parse_note or ""),
            "trace": self.trace.to_dict(),
            "quality": self.quality.to_dict(),
            "blocks": list(self.blocks),
            "pages": list(self.pages),
        }

