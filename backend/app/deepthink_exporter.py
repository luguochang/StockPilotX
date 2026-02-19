from __future__ import annotations

from datetime import datetime
from typing import Any


class DeepThinkReportExporter:
    """DeepThink report exporter for Markdown/PDF outputs.

    Notes:
    - Markdown export is deterministic and readable for human review.
    - PDF export uses a minimal built-in PDF writer (no external dependency),
      so it can run in locked-down environments.
    """

    def export_markdown(self, session: dict[str, Any]) -> str:
        session_id = str(session.get("session_id", ""))
        question = str(session.get("question", ""))
        stock_codes = list(session.get("stock_codes", []))
        rounds = list(session.get("rounds", []))

        lines: list[str] = [
            f"# DeepThink Report - {session_id}",
            "",
            f"- Question: {question}",
            f"- Stocks: {', '.join(map(str, stock_codes)) or 'N/A'}",
            f"- Current Round: {int(session.get('current_round', 0) or 0)}",
            f"- Generated At: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
        ]
        if not rounds:
            lines.append("_No rounds available yet._")
            return "\n".join(lines)

        for round_item in rounds:
            round_no = int(round_item.get("round_no", 0) or 0)
            signal = str(round_item.get("consensus_signal", "hold"))
            disagreement = float(round_item.get("disagreement_score", 0.0) or 0.0)
            lines.extend(
                [
                    f"## Round {round_no}",
                    "",
                    f"- Consensus Signal: `{signal}`",
                    f"- Disagreement Score: `{disagreement:.4f}`",
                    f"- Conflict Sources: {', '.join(map(str, round_item.get('conflict_sources', []))) or 'none'}",
                    "",
                    "### Opinions",
                ]
            )
            opinions = list(round_item.get("opinions", []))
            if not opinions:
                lines.append("- _No opinions recorded._")
            for op in opinions:
                agent = str(op.get("agent_id", op.get("agent", "")))
                op_signal = str(op.get("signal", "hold"))
                confidence = float(op.get("confidence", 0.0) or 0.0)
                reason = str(op.get("reason", "")).strip() or "N/A"
                lines.append(f"- **{agent}**: `{op_signal}` ({confidence:.2f}) - {reason}")
            lines.append("")
            summary = round_item.get("business_summary", {})
            if isinstance(summary, dict) and summary:
                lines.extend(["### Business Summary", ""])
                for key in (
                    "final_signal",
                    "confidence",
                    "thesis",
                    "risk_warning",
                    "invalidation_condition",
                ):
                    value = summary.get(key, "")
                    if str(value).strip():
                        lines.append(f"- {key}: {value}")
                lines.append("")
        return "\n".join(lines)

    def export_pdf_bytes(self, session: dict[str, Any]) -> bytes:
        """Export session report as minimal single-page/multi-line PDF bytes."""
        markdown = self.export_markdown(session)
        text_lines = markdown.splitlines()
        return self._build_minimal_pdf(text_lines[:180])

    def _build_minimal_pdf(self, lines: list[str]) -> bytes:
        """Build a tiny valid PDF document with Helvetica text lines."""
        # Escape PDF string special chars.
        def esc(value: str) -> str:
            return value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

        y = 800
        content_rows = ["BT", "/F1 10 Tf", "72 820 Td", "14 TL"]
        for idx, raw in enumerate(lines):
            if idx > 0:
                content_rows.append("T*")
            line = esc(str(raw))
            # PDF text operators prefer reasonably short lines.
            content_rows.append(f"({line[:180]}) Tj")
            y -= 14
            if y < 60:
                break
        content_rows.append("ET")
        stream_data = "\n".join(content_rows).encode("latin-1", errors="replace")

        objects: list[bytes] = []
        objects.append(b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n")
        objects.append(b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n")
        objects.append(
            b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] "
            b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >> endobj\n"
        )
        objects.append(b"4 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n")
        objects.append(
            b"5 0 obj << /Length " + str(len(stream_data)).encode("ascii") + b" >> stream\n" + stream_data + b"\nendstream endobj\n"
        )

        header = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"
        body = bytearray(header)
        offsets = [0]
        for obj in objects:
            offsets.append(len(body))
            body.extend(obj)

        xref_pos = len(body)
        body.extend(f"xref\n0 {len(offsets)}\n".encode("ascii"))
        body.extend(b"0000000000 65535 f \n")
        for offset in offsets[1:]:
            body.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
        body.extend(
            (
                f"trailer << /Size {len(offsets)} /Root 1 0 R >>\n"
                f"startxref\n{xref_pos}\n%%EOF\n"
            ).encode("ascii")
        )
        return bytes(body)

