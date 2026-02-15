from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from typing import Any


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def verify_password(password: str, hashed: str) -> bool:
    return hmac.compare_digest(hash_password(password), hashed)


def create_token(payload: dict[str, Any], secret: str, expire_seconds: int) -> str:
    body = dict(payload)
    body["exp"] = int(time.time()) + expire_seconds
    raw = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    msg = base64.urlsafe_b64encode(raw).decode("utf-8").rstrip("=")
    sig = hmac.new(secret.encode("utf-8"), msg.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"{msg}.{sig}"


def decode_token(token: str, secret: str) -> dict[str, Any]:
    try:
        msg, sig = token.split(".", 1)
    except ValueError as ex:
        raise ValueError("invalid token format") from ex
    expected = hmac.new(secret.encode("utf-8"), msg.encode("utf-8"), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(sig, expected):
        raise ValueError("invalid token signature")
    padded = msg + "=" * ((4 - len(msg) % 4) % 4)
    payload = json.loads(base64.urlsafe_b64decode(padded.encode("utf-8")).decode("utf-8"))
    if int(payload.get("exp", 0)) < int(time.time()):
        raise ValueError("token expired")
    return payload

