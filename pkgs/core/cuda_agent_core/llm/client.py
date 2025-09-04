from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Dict, Generator, Iterable, List, Optional

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential_jitter,
    retry_if_exception_type,
)
from litellm import completion


# Simple typed structures for messages and responses
@dataclass
class ChatMessage:
    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass
class ChatResponse:
    model: str
    content: str
    role: str = "assistant"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: int = 0
    raw: Optional[dict] = None


@dataclass
class StreamChunk:
    content_delta: str
    done: bool = False


class ChatError(Exception):
    pass


def _env(key: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(key)
    return v if v is not None else default


class ChatClient:
    """LiteLLM-backed chat client with retries and exponential backoff.

    Env configuration:
    - OPENAI_API_KEY: required for OpenAI models via LiteLLM
    - LLM_MODEL: default model (e.g., "gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo")
    - LLM_TIMEOUT: request timeout in seconds (default 60)
    - LLM_RETRY_MAX: max retry attempts (default 5)
    - LLM_RETRY_BASE: base backoff seconds for exponential jitter (default 0.5)
    """

    def __init__(
        self,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
        retry_max: Optional[int] = None,
        retry_base: Optional[float] = None,
    ) -> None:
        self.model = model or _env("LLM_MODEL", "gpt-4o-mini")
        self.timeout = float(timeout or _env("LLM_TIMEOUT", "60"))
        self.retry_max = int(retry_max or _env("LLM_RETRY_MAX", "5"))
        self.retry_base = float(retry_base or _env("LLM_RETRY_BASE", "0.5"))

        if not os.getenv("OPENAI_API_KEY"):
            # LiteLLM reads OPENAI_API_KEY; fail fast with a clear message
            raise ChatError("OPENAI_API_KEY not set in environment")

    def _retry_decorator(self):
        return retry(
            reraise=True,
            stop=stop_after_attempt(self.retry_max),
            wait=wait_exponential_jitter(initial=self.retry_base, max=8),
            retry=retry_if_exception_type(Exception),
        )

    def chat(
        self,
        messages: Iterable[ChatMessage | Dict[str, str]],
        *,
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        extra: Optional[Dict] = None,
    ) -> ChatResponse:
        """Non-streaming chat completion with retries.

        Returns a ChatResponse with content and token usage.
        """

        msgs = [m if isinstance(m, dict) else {"role": m.role, "content": m.content} for m in messages]
        chosen_model = model or self.model

        @self._retry_decorator()
        def _invoke():
            t0 = time.time()
            resp = completion(
                model=chosen_model,
                messages=msgs,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=self.timeout,
                **(extra or {}),
            )
            dt_ms = int((time.time() - t0) * 1000)
            content = resp["choices"][0]["message"]["content"] if resp.get("choices") else ""
            usage = resp.get("usage", {})
            return ChatResponse(
                model=chosen_model,
                content=content or "",
                role=resp["choices"][0]["message"].get("role", "assistant") if resp.get("choices") else "assistant",
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
                latency_ms=dt_ms,
                raw=resp,
            )

        return _invoke()

    def stream_chat(
        self,
        messages: Iterable[ChatMessage | Dict[str, str]],
        *,
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        extra: Optional[Dict] = None,
    ) -> Generator[StreamChunk, None, None]:
        """Streaming chat completion generator with retries.

        Yields StreamChunk(content_delta=str, done=bool).
        """
        msgs = [m if isinstance(m, dict) else {"role": m.role, "content": m.content} for m in messages]
        chosen_model = model or self.model

        @self._retry_decorator()
        def _invoke_stream():
            return completion(
                model=chosen_model,
                messages=msgs,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=self.timeout,
                stream=True,
                **(extra or {}),
            )

        # On retry, the generator itself will be re-created.
        for event in _invoke_stream():
            try:
                delta = event["choices"][0]["delta"].get("content", "")
                finished = event["choices"][0].get("finish_reason") is not None
                if delta:
                    yield StreamChunk(content_delta=delta, done=False)
                if finished:
                    yield StreamChunk(content_delta="", done=True)
            except Exception:
                # Defensive: if shape is unexpected, skip token
                continue


_default_client: Optional[ChatClient] = None


def default_client() -> ChatClient:
    global _default_client
    if _default_client is None:
        _default_client = ChatClient()
    return _default_client

