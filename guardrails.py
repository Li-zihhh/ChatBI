"""限流与审计。

ChatBI 的两个企业级刚需，早期版本完全缺失：
  1. **限流**：一次查询会消耗 1~3 次 LLM 调用 + 若干次 Embedding，
     无限制意味着任何调用方都能把账单打穿。这里用进程内滑动窗口实现，
     单实例足够；多实例部署时应换成 Redis 计数。
  2. **审计**：BI 系统的合规底线是"谁在什么时候查了什么"。
     JSONL 追加写入，便于后续用 jq / pandas 分析。
"""

from __future__ import annotations

import json
import logging
import threading
from collections import defaultdict, deque
from datetime import date, datetime
from pathlib import Path
from time import time
from typing import Any

logger = logging.getLogger("chatbi.guardrails")


class RateLimitError(Exception):
    """请求超出配额。"""

    def __init__(self, message: str, retry_after: int = 60):
        super().__init__(message)
        self.retry_after = retry_after


class RateLimiter:
    """进程内滑动窗口限流器（按 key 维度）。

    同时限制每分钟与每日请求数：
      - 分钟窗口用 deque 存时间戳，滑动过期；
      - 每日窗口用 (date, count) 计数，跨天自动重置。
    """

    def __init__(self, per_minute: int = 20, per_day: int = 2000):
        self.per_minute = max(1, per_minute)
        self.per_day = max(1, per_day)
        self._minute_windows: dict[str, deque[float]] = defaultdict(deque)
        self._day_counters: dict[str, tuple[date, int]] = {}
        self._lock = threading.Lock()

    def check(self, key: str) -> None:
        """检查并记账；超限抛 RateLimitError。"""
        now = time()
        today = date.today()

        with self._lock:
            window = self._minute_windows[key]
            cutoff = now - 60
            while window and window[0] < cutoff:
                window.popleft()

            if len(window) >= self.per_minute:
                retry_after = max(1, int(60 - (now - window[0])) + 1)
                raise RateLimitError(
                    f"请求过于频繁，请 {retry_after} 秒后重试。", retry_after=retry_after
                )

            current_date, count = self._day_counters.get(key, (today, 0))
            if current_date != today:
                current_date, count = today, 0
            if count >= self.per_day:
                raise RateLimitError(
                    f"今日查询配额已用完（{self.per_day} 次），请明日再试。",
                    retry_after=3600,
                )

            window.append(now)
            self._day_counters[key] = (current_date, count + 1)

    def snapshot(self, key: str) -> dict[str, int]:
        """返回当前用量，便于 /usage 类接口展示。"""
        with self._lock:
            now = time()
            window = self._minute_windows.get(key, deque())
            used_minute = sum(1 for ts in window if ts >= now - 60)
            _, used_day = self._day_counters.get(key, (date.today(), 0))
        return {
            "used_last_minute": used_minute,
            "limit_per_minute": self.per_minute,
            "used_today": used_day,
            "limit_per_day": self.per_day,
        }


class AuditLogger:
    """把每次查询的关键事实落到 JSONL。

    只记录"事实"，不记录敏感值：记录 question 与 SQL 文本，
    但响应体中的具体数据不落盘（避免审计日志本身变成数据泄露渠道）。
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._lock = threading.Lock()
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as exc:  # pragma: no cover
            logger.warning("审计日志目录创建失败：%s", exc)

    def log(self, **record: Any) -> None:
        payload = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            **record,
        }
        line = json.dumps(payload, ensure_ascii=False, default=str)
        try:
            with self._lock:
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
        except Exception as exc:  # pragma: no cover
            # 审计失败不能影响主流程，但必须留下痕迹
            logger.error("审计日志写入失败：%s | record=%s", exc, line[:500])
