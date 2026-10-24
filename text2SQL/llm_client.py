"""LLM 客户端：同步 / 流式 / 文本生成。

相对早期版本的变化：
  1. **惰性初始化**：`OpenAI()` 只在首次真正调用时构造。
     早期实现会在 import / 构造 `ChatBISystem` 时立刻建客户端，
     导致单元测试无法离线运行（环境变量里一个畸形的 NO_PROXY 就能让测试全挂）。
  2. **超时 + 指数退避重试**：避免单次请求挂死占满 FastAPI 线程池。
  3. **统一 markdown 清理**：同步与流式路径共用 `clean_sql`。
"""

from __future__ import annotations

import logging
import re
import time
from typing import Generator

from openai import OpenAI

from config import LLM_CONFIG

logger = logging.getLogger("chatbi.llm")

_FENCE_RE = re.compile(r"```(?:sql)?", re.IGNORECASE)

# 这些异常值得重试（网络抖动、限流、服务端 5xx）
_RETRYABLE_MARKERS = (
    "timeout",
    "timed out",
    "rate limit",
    "429",
    "500",
    "502",
    "503",
    "504",
    "connection",
    "temporarily",
    "overloaded",
)


def clean_sql(raw: str) -> str:
    """去掉 markdown 代码块围栏并修剪。"""
    cleaned = _FENCE_RE.sub("", raw or "")
    return cleaned.strip().rstrip(";").strip()


def _is_retryable(exc: Exception) -> bool:
    message = str(exc).lower()
    if any(marker in message for marker in _RETRYABLE_MARKERS):
        return True
    # OpenAI SDK 的 APIConnectionError / APITimeoutError 均属于可重试
    return type(exc).__name__ in {
        "APIConnectionError",
        "APITimeoutError",
        "InternalServerError",
        "RateLimitError",
    }


class LLMClient:
    """LLM API 客户端。"""

    def __init__(self, config: dict | None = None):
        self._config = dict(config or LLM_CONFIG)
        self._client: OpenAI | None = None
        self.model = self._config["model"]
        self.embedding_model = self._config.get("embedding_model", "text-embedding-v3")
        self.temperature = self._config.get("temperature", 0.2)
        self.max_tokens = self._config.get("max_tokens", 4000)
        self.timeout = float(self._config.get("timeout", 30))
        self.max_retries = int(self._config.get("max_retries", 2))
        self.retry_backoff = float(self._config.get("retry_backoff", 0.5))

    @property
    def client(self) -> OpenAI:
        """惰性构造底层 SDK 客户端。"""
        if self._client is None:
            self._client = OpenAI(
                api_key=self._config.get("api_key"),
                base_url=self._config.get("base_url"),
                timeout=self.timeout,
                max_retries=0,  # 重试由本类统一控制，避免双重退避
            )
        return self._client

    # ---------- 生成 ----------
    def generate_sql(self, system_msg: str, prompt: str) -> str:
        """生成 SQL（同步，一次性返回）。"""
        return clean_sql(self.generate_text(system_msg=system_msg, prompt=prompt))

    def generate_text(self, system_msg: str, prompt: str) -> str:
        """生成普通文本，带超时与指数退避重试。"""
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    timeout=self.timeout,
                )
                content = response.choices[0].message.content or ""
                return content.strip()
            except Exception as exc:
                last_exc = exc
                if attempt >= self.max_retries or not _is_retryable(exc):
                    raise
                delay = self.retry_backoff * (2**attempt)
                logger.warning(
                    "LLM 调用失败（第 %d/%d 次），%.1fs 后重试：%s",
                    attempt + 1,
                    self.max_retries + 1,
                    delay,
                    exc,
                )
                time.sleep(delay)

        raise last_exc  # pragma: no cover - 循环必然在此前 return 或 raise

    def generate_sql_stream(self, system_msg: str, prompt: str) -> Generator[str, None, None]:
        """流式生成 SQL（逐 chunk 产出）。

        流式请求不做自动重试：一旦已向客户端推送过内容，重试会造成重复输出。
        首次建立连接前失败时才重试一次（由 SDK 层之外的本方法处理成本较高，
        因此这里只在建立 stream 失败时重试，产出阶段失败直接抛出）。
        """
        stream = None
        for attempt in range(self.max_retries + 1):
            try:
                stream = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    stream=True,
                    timeout=self.timeout,
                )
                break
            except Exception as exc:
                if attempt >= self.max_retries or not _is_retryable(exc):
                    raise
                delay = self.retry_backoff * (2**attempt)
                logger.warning("流式请求建立失败，%.1fs 后重试：%s", delay, exc)
                time.sleep(delay)

        if stream is None:  # pragma: no cover
            raise RuntimeError("流式请求未能建立")

        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            content = delta.content if delta and delta.content is not None else None
            if content is not None:
                yield content
