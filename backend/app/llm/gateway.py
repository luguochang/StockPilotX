from __future__ import annotations

from dataclasses import dataclass, field
import json
import time
from typing import Any, Iterator
from urllib.request import Request, urlopen

from backend.app.config import Settings
from backend.app.state import AgentState


def _join_url(base: str, path: str) -> str:
    return base.rstrip("/") + "/" + path.lstrip("/")


@dataclass(slots=True)
class ProviderConfig:
    """Single model provider config."""

    name: str
    api_base: str
    model: str
    api_style: str = "anthropic_messages"  # anthropic_messages | openai_chat | openai_responses
    enabled: bool = True
    api_key: str = ""
    api_key_header: str = "Authorization"
    api_key_prefix: str = "Bearer "
    anthropic_version: str = "2023-06-01"
    max_tokens: int = 1024
    temperature: float = 0.2
    top_p: float = 0.95
    stream: bool = False
    timeout_seconds: float = 20.0
    extra_headers: dict[str, str] = field(default_factory=dict)
    extra_body: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any], default_timeout: float) -> "ProviderConfig":
        key = str(payload.get("api_key", "")).strip()
        key_env = str(payload.get("api_key_env", "")).strip()
        if not key and key_env:
            import os

            key = os.getenv(key_env, "")
        return cls(
            name=str(payload.get("name", "unnamed")),
            api_base=str(payload.get("api_base", "")).strip(),
            model=str(payload.get("model", "")).strip(),
            api_style=str(payload.get("api_style", "anthropic_messages")).strip(),
            enabled=bool(payload.get("enabled", True)),
            api_key=key,
            api_key_header=str(payload.get("api_key_header", "Authorization")),
            api_key_prefix=str(payload.get("api_key_prefix", "Bearer ")),
            anthropic_version=str(payload.get("anthropic_version", "2023-06-01")),
            max_tokens=max(1, int(payload.get("max_tokens", 1024))),
            temperature=float(payload.get("temperature", 0.2)),
            top_p=float(payload.get("top_p", 0.95)),
            stream=bool(payload.get("stream", False)),
            timeout_seconds=float(payload.get("timeout_seconds", default_timeout)),
            extra_headers=dict(payload.get("extra_headers", {})),
            extra_body=dict(payload.get("extra_body", {})),
        )


class MultiProviderLLMGateway:
    """Multi-provider LLM gateway with ordered failover."""

    def __init__(self, settings: Settings, trace_emit: callable | None = None) -> None:
        self.settings = settings
        self.trace_emit = trace_emit
        self.providers = self._load_configs()

    def _load_configs(self) -> list[ProviderConfig]:
        rows = self.settings.load_llm_provider_configs()
        return [ProviderConfig.from_dict(x, self.settings.llm_request_timeout_seconds) for x in rows]

    def generate(
        self,
        state: AgentState,
        prompt: str,
        request_overrides: dict[str, Any] | None = None,
    ) -> str:
        """Call external LLM and return text; raise if all providers fail.

        request_overrides 用于本次请求临时注入 body 字段（例如 Responses 的 tools），
        不会改写 provider 的持久配置。
        """
        if not self.settings.llm_external_enabled:
            raise RuntimeError("external llm disabled")
        if not self.providers:
            raise RuntimeError("no llm providers configured")

        errors: list[str] = []
        for p in self.providers:
            if not p.enabled:
                continue
            try:
                text = self._call_with_retry(p, prompt, request_overrides=request_overrides)
                if text.strip():
                    # 将实际成功的模型信息回写到 state，供上层事件透传给前端展示。
                    state.analysis["llm_provider"] = p.name
                    state.analysis["llm_model"] = p.model
                    state.analysis["llm_api_style"] = p.api_style
                    self._emit(state.trace_id, "llm_provider_success", {"provider": p.name, "model": p.model})
                    return text
                errors.append(f"{p.name}: empty response")
            except Exception as ex:  # noqa: BLE001
                errors.append(f"{p.name}: {ex}")
                self._emit(state.trace_id, "llm_provider_error", {"provider": p.name, "error": str(ex)})
                continue
        raise RuntimeError("all llm providers failed: " + "; ".join(errors))

    def stream_generate(
        self,
        state: AgentState,
        prompt: str,
        request_overrides: dict[str, Any] | None = None,
    ) -> Iterator[str]:
        """Stream external model delta chunks.

        request_overrides 的语义与 generate 保持一致。
        """
        if not self.settings.llm_external_enabled:
            raise RuntimeError("external llm disabled")
        if not self.providers:
            raise RuntimeError("no llm providers configured")

        errors: list[str] = []
        for p in self.providers:
            if not p.enabled:
                continue
            try:
                emitted = False
                for chunk in self._stream_with_retry(p, prompt, request_overrides=request_overrides):
                    if chunk:
                        emitted = True
                        yield chunk
                if emitted:
                    # 流式链路同样记录本次命中的 provider/model。
                    state.analysis["llm_provider"] = p.name
                    state.analysis["llm_model"] = p.model
                    state.analysis["llm_api_style"] = p.api_style
                    self._emit(state.trace_id, "llm_provider_success", {"provider": p.name, "model": p.model})
                    return
                errors.append(f"{p.name}: empty stream")
            except Exception as ex:  # noqa: BLE001
                errors.append(f"{p.name}: {ex}")
                self._emit(state.trace_id, "llm_provider_error", {"provider": p.name, "error": str(ex)})
                continue
        raise RuntimeError("all llm providers failed: " + "; ".join(errors))

    def _call_with_retry(
        self,
        provider: ProviderConfig,
        prompt: str,
        request_overrides: dict[str, Any] | None = None,
    ) -> str:
        attempts = max(1, self.settings.llm_retry_count)
        last_err: Exception | None = None
        for _ in range(attempts):
            try:
                return self._call_provider(provider, prompt, request_overrides=request_overrides)
            except Exception as ex:  # noqa: BLE001
                last_err = ex
        if last_err:
            raise last_err
        raise RuntimeError("unknown llm call error")

    def _call_provider(
        self,
        provider: ProviderConfig,
        prompt: str,
        request_overrides: dict[str, Any] | None = None,
    ) -> str:
        if provider.api_style == "anthropic_messages":
            return self._call_anthropic_messages(provider, prompt, request_overrides=request_overrides)
        if provider.api_style == "openai_chat":
            return self._call_openai_chat(provider, prompt, request_overrides=request_overrides)
        if provider.api_style == "openai_responses":
            return self._call_openai_responses(provider, prompt, request_overrides=request_overrides)
        raise RuntimeError(f"unsupported api_style: {provider.api_style}")

    def _stream_with_retry(
        self,
        provider: ProviderConfig,
        prompt: str,
        request_overrides: dict[str, Any] | None = None,
    ) -> Iterator[str]:
        attempts = max(1, self.settings.llm_retry_count)
        last_err: Exception | None = None
        for idx in range(attempts):
            try:
                yield from self._stream_provider(provider, prompt, request_overrides=request_overrides)
                return
            except Exception as ex:  # noqa: BLE001
                last_err = ex
                if idx < attempts - 1:
                    # ?                    wait_s = self.settings.llm_retry_backoff_seconds * (2**idx)
                    if wait_s > 0:
                        time.sleep(wait_s)
        if last_err:
            raise last_err
        raise RuntimeError("unknown llm stream error")

    def _stream_provider(
        self,
        provider: ProviderConfig,
        prompt: str,
        request_overrides: dict[str, Any] | None = None,
    ) -> Iterator[str]:
        if provider.api_style == "anthropic_messages":
            yield from self._stream_anthropic_messages(provider, prompt, request_overrides=request_overrides)
            return
        if provider.api_style == "openai_chat":
            yield from self._stream_openai_chat(provider, prompt, request_overrides=request_overrides)
            return
        if provider.api_style == "openai_responses":
            yield from self._stream_openai_responses(provider, prompt, request_overrides=request_overrides)
            return
        raise RuntimeError(f"unsupported api_style: {provider.api_style}")

    def _call_anthropic_messages(
        self,
        provider: ProviderConfig,
        prompt: str,
        request_overrides: dict[str, Any] | None = None,
    ) -> str:
        endpoint = _join_url(provider.api_base, "messages")
        body: dict[str, Any] = {
            "model": provider.model,
            "max_tokens": provider.max_tokens,
            "temperature": provider.temperature,
            "top_p": provider.top_p,
            "messages": [{"role": "user", "content": prompt}],
            "stream": provider.stream,
        }
        self._merge_request_body(body, provider.extra_body)
        self._merge_request_body(body, request_overrides)
        headers = self._build_headers(provider)
        headers["anthropic-version"] = provider.anthropic_version
        payload = self._post_json(endpoint, body, headers, provider.timeout_seconds, stream=provider.stream)
        if provider.stream:
            return self._parse_anthropic_stream(payload)
        return self._parse_anthropic_response(payload)

    def _call_openai_chat(
        self,
        provider: ProviderConfig,
        prompt: str,
        request_overrides: dict[str, Any] | None = None,
    ) -> str:
        endpoint = _join_url(provider.api_base, "chat/completions")
        body: dict[str, Any] = {
            "model": provider.model,
            "temperature": provider.temperature,
            "top_p": provider.top_p,
            "max_tokens": provider.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "stream": provider.stream,
        }
        self._merge_request_body(body, provider.extra_body)
        self._merge_request_body(body, request_overrides)
        headers = self._build_headers(provider)
        payload = self._post_json(endpoint, body, headers, provider.timeout_seconds, stream=provider.stream)
        if provider.stream:
            return self._parse_openai_stream(payload)
        return self._parse_openai_response(payload)

    def _call_openai_responses(
        self,
        provider: ProviderConfig,
        prompt: str,
        request_overrides: dict[str, Any] | None = None,
    ) -> str:
        endpoint = _join_url(provider.api_base, "responses")
        body: dict[str, Any] = {
            "model": provider.model,
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": prompt,
                        }
                    ],
                }
            ],
            "stream": False,
        }
        self._merge_request_body(body, provider.extra_body)
        self._merge_request_body(body, request_overrides)
        headers = self._build_headers(provider)
        payload = self._post_json(endpoint, body, headers, provider.timeout_seconds, stream=False)
        return self._parse_openai_responses_response(payload)

    def _stream_anthropic_messages(
        self,
        provider: ProviderConfig,
        prompt: str,
        request_overrides: dict[str, Any] | None = None,
    ) -> Iterator[str]:
        endpoint = _join_url(provider.api_base, "messages")
        body: dict[str, Any] = {
            "model": provider.model,
            "max_tokens": provider.max_tokens,
            "temperature": provider.temperature,
            "top_p": provider.top_p,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
        }
        self._merge_request_body(body, provider.extra_body)
        self._merge_request_body(body, request_overrides)
        headers = self._build_headers(provider)
        headers["anthropic-version"] = provider.anthropic_version
        yielded = False
        for event in self._iter_sse_events(endpoint, body, headers, provider.timeout_seconds):
            if event.get("type") == "content_block_delta":
                delta = event.get("delta", {})
                text = delta.get("text")
                if isinstance(text, str) and text:
                    yielded = True
                    yield text
        if not yielded:
            raise RuntimeError("anthropic stream missing delta text")

    def _stream_openai_chat(
        self,
        provider: ProviderConfig,
        prompt: str,
        request_overrides: dict[str, Any] | None = None,
    ) -> Iterator[str]:
        endpoint = _join_url(provider.api_base, "chat/completions")
        body: dict[str, Any] = {
            "model": provider.model,
            "temperature": provider.temperature,
            "top_p": provider.top_p,
            "max_tokens": provider.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
        }
        self._merge_request_body(body, provider.extra_body)
        self._merge_request_body(body, request_overrides)
        headers = self._build_headers(provider)
        yielded = False
        for event in self._iter_sse_events(endpoint, body, headers, provider.timeout_seconds):
            for choice in event.get("choices", []):
                delta = choice.get("delta", {})
                text = delta.get("content")
                if isinstance(text, str) and text:
                    yielded = True
                    yield text
        if not yielded:
            raise RuntimeError("openai stream missing delta content")

    def _stream_openai_responses(
        self,
        provider: ProviderConfig,
        prompt: str,
        request_overrides: dict[str, Any] | None = None,
    ) -> Iterator[str]:
        endpoint = _join_url(provider.api_base, "responses")
        body: dict[str, Any] = {
            "model": provider.model,
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": prompt,
                        }
                    ],
                }
            ],
            "stream": True,
        }
        self._merge_request_body(body, provider.extra_body)
        self._merge_request_body(body, request_overrides)
        headers = self._build_headers(provider)
        yielded = False
        for event in self._iter_sse_events(endpoint, body, headers, provider.timeout_seconds):
            # esponse.output_text.delta
            if event.get("type") == "response.output_text.delta":
                delta = event.get("delta")
                if isinstance(delta, str) and delta:
                    yielded = True
                    yield delta
                continue
            # ?delta ?text 
            delta2 = event.get("delta")
            text2 = event.get("text")
            if isinstance(delta2, str) and delta2:
                yielded = True
                yield delta2
            elif isinstance(text2, str) and text2:
                yielded = True
                yield text2
        if not yielded:
            raise RuntimeError("openai responses stream missing delta content")

    def _build_headers(self, provider: ProviderConfig) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "User-Agent": "StockPilotX/1.0",
        }
        if provider.api_key:
            headers[provider.api_key_header] = f"{provider.api_key_prefix}{provider.api_key}"
            # Some anthropic proxy gateways require x-api-key in addition to Bearer auth.
            if provider.api_style == "anthropic_messages" and "x-api-key" not in headers:
                headers["x-api-key"] = provider.api_key
        headers.update(provider.extra_headers)
        return headers

    @staticmethod
    def _merge_request_body(base: dict[str, Any], patch: dict[str, Any] | None) -> None:
        """递归合并请求 body，支持按请求覆盖 provider 默认参数。"""
        if not patch:
            return
        for key, value in patch.items():
            # 仅当双方都是 dict 才递归，其他类型直接覆盖，保持行为可预测。
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                MultiProviderLLMGateway._merge_request_body(base[key], value)
            else:
                base[key] = value

    @staticmethod
    def _post_json(
        url: str,
        body: dict[str, Any],
        headers: dict[str, str],
        timeout: float,
        stream: bool = False,
    ) -> str:
        req = Request(
            url=url,
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urlopen(req, timeout=timeout) as resp:  # noqa: S310 - 
            text = resp.read().decode("utf-8", errors="ignore")
            if not text and not stream:
                raise RuntimeError("empty http response body")
            return text

    @staticmethod
    def _iter_sse_events(
        url: str,
        body: dict[str, Any],
        headers: dict[str, str],
        timeout: float,
    ) -> Iterator[dict[str, Any]]:
        req = Request(
            url=url,
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urlopen(req, timeout=timeout) as resp:  # noqa: S310 - 
            buffer = ""
            while True:
                raw = resp.readline()
                if not raw:
                    break
                line = raw.decode("utf-8", errors="ignore")
                if line == "\n":
                    block = buffer.strip()
                    buffer = ""
                    if not block:
                        continue
                    for row in block.splitlines():
                        row = row.strip()
                        if not row.startswith("data:"):
                            continue
                        data = row[5:].strip()
                        if not data or data == "[DONE]":
                            continue
                        try:
                            yield json.loads(data)
                        except json.JSONDecodeError:
                            continue
                    continue
                buffer += line

    @staticmethod
    def _parse_anthropic_response(payload: str) -> str:
        data = json.loads(payload)
        content = data.get("content", [])
        texts: list[str] = []
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    texts.append(str(item.get("text", "")))
        if not texts:
            raise RuntimeError("anthropic response missing content.text")
        return "\n".join(texts).strip()

    @staticmethod
    def _parse_openai_response(payload: str) -> str:
        data = json.loads(payload)
        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError("openai response missing choices")
        message = choices[0].get("message", {})
        text = message.get("content", "")
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError("openai response missing choices[0].message.content")
        return text.strip()

    @staticmethod
    def _parse_openai_responses_response(payload: str) -> str:
        # :
        # 1)  JSON 
        # 2)  stream=false  SSE(event/data)
        try:
            data = json.loads(payload)
            return MultiProviderLLMGateway._extract_openai_responses_text(data)
        except json.JSONDecodeError:
            pass

        chunks: list[str] = []
        final_text: str = ""
        for line in payload.splitlines():
            row = line.strip()
            if not row.startswith("data:"):
                continue
            data_str = row[5:].strip()
            if not data_str or data_str == "[DONE]":
                continue
            try:
                event = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            event_type = str(event.get("type", ""))
            if event_type == "response.output_text.delta":
                delta = event.get("delta")
                if isinstance(delta, str) and delta:
                    chunks.append(delta)
                continue
            if event_type == "response.output_text.done":
                done_text = event.get("text")
                if isinstance(done_text, str) and done_text.strip():
                    final_text = done_text.strip()
                continue
            if event_type == "response.completed":
                response_obj = event.get("response")
                if isinstance(response_obj, dict):
                    try:
                        final_text = MultiProviderLLMGateway._extract_openai_responses_text(response_obj)
                    except Exception:
                        pass

        merged = "".join(chunks).strip() or final_text.strip()
        if not merged:
            raise RuntimeError("openai responses payload missing output_text")
        return merged

    @staticmethod
    def _extract_openai_responses_text(data: dict[str, Any]) -> str:
        if isinstance(data.get("output_text"), str) and data["output_text"].strip():
            return data["output_text"].strip()
        texts: list[str] = []
        for item in data.get("output", []):
            if not isinstance(item, dict):
                continue
            for content in item.get("content", []):
                if not isinstance(content, dict):
                    continue
                if content.get("type") in {"output_text", "text"} and isinstance(content.get("text"), str):
                    texts.append(content["text"])
        merged = "".join(texts).strip()
        if not merged:
            raise RuntimeError("openai responses payload missing output_text")
        return merged

    @staticmethod
    def _parse_anthropic_stream(payload: str) -> str:
        # SSE  content_block_delta.delta.text
        lines = payload.splitlines()
        chunks: list[str] = []
        for line in lines:
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if not data or data == "[DONE]":
                continue
            try:
                event = json.loads(data)
            except json.JSONDecodeError:
                continue
            if event.get("type") == "content_block_delta":
                delta = event.get("delta", {})
                text = delta.get("text")
                if isinstance(text, str):
                    chunks.append(text)
        if not chunks:
            raise RuntimeError("anthropic stream missing delta text")
        return "".join(chunks).strip()

    @staticmethod
    def _parse_openai_stream(payload: str) -> str:
        # SSE  choices[].delta.content
        lines = payload.splitlines()
        chunks: list[str] = []
        for line in lines:
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if not data or data == "[DONE]":
                continue
            try:
                event = json.loads(data)
            except json.JSONDecodeError:
                continue
            for choice in event.get("choices", []):
                delta = choice.get("delta", {})
                text = delta.get("content")
                if isinstance(text, str):
                    chunks.append(text)
        if not chunks:
            raise RuntimeError("openai stream missing delta content")
        return "".join(chunks).strip()

    def _emit(self, trace_id: str, event: str, payload: dict[str, Any]) -> None:
        if self.trace_emit:
            self.trace_emit(trace_id, event, payload)

