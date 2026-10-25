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


def _is_timeout(exc: Exception) -> bool:
    """判断是否为超时类失败。

    超时要和"连接被拒/5xx/限流"区分开：后者重试常常立刻成功，
    而超时意味着服务端就是慢，同一个 prompt 再跑一遍大概率还是超时——
    所以按 max_retries 重试 3 次纯属把 30 秒浪费成 90 秒。
    """
    if type(exc).__name__ in {"APITimeoutError", "TimeoutException", "ReadTimeout"}:
        return True
    message = str(exc).lower()
    return "timeout" in message or "timed out" in message


class LLMClient:
    """LLM API 客户端。"""

    def __init__(self, config: dict | None = None):
        self._config = dict(config or LLM_CONFIG)
        self._client: OpenAI | None = None
        self.model = self._config["model"]
        self.embedding_model = self._config.get("embedding_model", "text-embedding-v3")
        self.temperature = self._config.get("temperature", 0.2)
        self.max_tokens = self._config.get("max_tokens", 4000)
        self.timeout = float(self._config.get("timeout", 60))
        self.max_retries = int(self._config.get("max_retries", 2))
        self.retry_backoff = float(self._config.get("retry_backoff", 0.5))
        self.timeout_retries = int(self._config.get("timeout_retries", 1))
        self.enable_thinking = str(self._config.get("enable_thinking", "false")).lower()
        # 供应商不支持 enable_thinking 时，失败一次后自动降级为不传
        self._thinking_supported = self.enable_thinking in {"true", "false"}

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

    def _build_kwargs(self, **kwargs) -> dict:
        """组装请求参数，并在配置要求时关闭思考模式。"""
        if self._thinking_supported:
            kwargs["extra_body"] = {"enable_thinking": self.enable_thinking == "true"}
        return kwargs

    def _should_retry(self, exc: Exception, attempt: int, timeouts: int) -> bool:
        """超时按 timeout_retries 限次，其它可重试错误按 max_retries。

        timeouts 在调用前已自增，因此 timeout_retries=1 表示"最多再试 1 次超时"。
        """
        if _is_timeout(exc):
            return timeouts <= self.timeout_retries
        return attempt < self.max_retries and _is_retryable(exc)

    @staticmethod
    def _looks_like_unsupported_param(exc: Exception) -> bool:
        message = str(exc).lower()
        return "enable_thinking" in message or (
            "invalid_request" in message and "parameter" in message
        )

    # ---------- 生成 ----------
    def generate_sql(self, system_msg: str, prompt: str) -> str:
        """生成 SQL（同步，一次性返回）。"""
        return clean_sql(self.generate_text(system_msg=system_msg, prompt=prompt))

    def generate_text(self, system_msg: str, prompt: str) -> str:
        """生成普通文本，带超时与指数退避重试。"""
        last_exc: Exception | None = None
        attempt = 0
        timeouts = 0

        while True:
            try:
                response = self.client.chat.completions.create(
                    **self._build_kwargs(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": system_msg},
                            {"role": "user", "content": prompt},
                        ],
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        timeout=self.timeout,
                    )
                )
                content = response.choices[0].message.content or ""
                return content.strip()
            except Exception as exc:
                last_exc = exc

                # 供应商不认识 enable_thinking：降级后立刻重试一次
                if self._thinking_supported and self._looks_like_unsupported_param(exc):
                    logger.warning(
                        "当前端点不接受 enable_thinking 参数，已降级为不传：%s", exc
                    )
                    self._thinking_supported = False
                    continue

                if _is_timeout(exc):
                    timeouts += 1

                if not self._should_retry(exc, attempt, timeouts):
                    raise

                attempt += 1
                delay = self.retry_backoff * (2 ** (attempt - 1))
                logger.warning(
                    "LLM 调用失败（第 %d 次尝试，%s），%.1fs 后重试：%s",
                    attempt,
                    "超时" if _is_timeout(exc) else "可重试错误",
                    delay,
                    exc,
                )
                time.sleep(delay)

    def generate_sql_stream(self, system_msg: str, prompt: str) -> Generator[str, None, None]:
        """流式生成 SQL（逐 chunk 产出）。

        流式请求只在"建立连接"阶段重试：一旦已向客户端推送过内容，
        重试会造成重复输出。此外流式的读超时是逐 chunk 计的，
        对长生成比一次性返回更耐受。
        """
        stream = None
        attempt = 0
        timeouts = 0
        while True:
            try:
                stream = self.client.chat.completions.create(
                    **self._build_kwargs(
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
                )
                break
            except Exception as exc:
                if self._thinking_supported and self._looks_like_unsupported_param(exc):
                    self._thinking_supported = False
                    continue
                if _is_timeout(exc):
                    timeouts += 1
                if not self._should_retry(exc, attempt, timeouts):
                    raise
                attempt += 1
                delay = self.retry_backoff * (2 ** (attempt - 1))
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
