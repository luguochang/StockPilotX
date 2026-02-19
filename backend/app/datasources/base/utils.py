from __future__ import annotations


def normalize_stock_code(code: str) -> str:
    """Normalize stock code into SH/SZ-prefixed uppercase format."""

    value = (code or "").upper().replace(".", "").strip()
    if value.startswith(("SH", "SZ")):
        return value
    if value.startswith("6"):
        return f"SH{value}"
    return f"SZ{value}"


def decode_response(content: bytes, encoding: str | None = None) -> str:
    """Decode response body with practical fallback chain.

    Data providers frequently return mixed encodings (UTF-8/GBK/GB18030).
    """

    if encoding:
        return content.decode(encoding, errors="ignore")
    for candidate in ("utf-8", "gbk", "gb18030"):
        try:
            return content.decode(candidate)
        except UnicodeDecodeError:
            continue
    return content.decode("latin-1", errors="ignore")

