from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path


class DocConvertEngine:
    """Convert legacy .doc files into .docx for downstream parsing."""

    def __init__(self) -> None:
        self._cmd = self._detect_soffice_cmd()

    @staticmethod
    def _detect_soffice_cmd() -> str:
        for cmd in ("soffice", "libreoffice"):
            path = shutil.which(cmd)
            if path:
                return path
        return ""

    @property
    def available(self) -> bool:
        return bool(self._cmd)

    def convert_doc_to_docx(self, *, raw_bytes: bytes, filename: str) -> tuple[bytes | None, str]:
        if not self.available:
            return None, "doc_convert_unavailable"
        suffix = ".doc"
        try:
            with tempfile.TemporaryDirectory(prefix="rag-doc-convert-") as tmp_dir:
                tmp_root = Path(tmp_dir)
                src = tmp_root / (Path(filename).stem + suffix)
                src.write_bytes(raw_bytes)
                proc = subprocess.run(
                    [self._cmd, "--headless", "--convert-to", "docx", "--outdir", str(tmp_root), str(src)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=45,
                    check=False,
                )
                if proc.returncode != 0:
                    return None, "doc_convert_failed"
                dst = tmp_root / (src.stem + ".docx")
                if not dst.exists():
                    return None, "doc_convert_missing_output"
                return dst.read_bytes(), "doc_converted_to_docx"
        except Exception:  # noqa: BLE001
            return None, "doc_convert_exception"

