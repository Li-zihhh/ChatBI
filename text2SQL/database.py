"""数据库访问层：连接池 + 异常翻译 + 资源治理 + 会话级只读。

相对早期版本的关键变化：
  1. **连接池**（LIFO + 存活探活 + 溢出控制）：早期实现每次查询都
     `pymysql.connect()`，一次 TCP 握手 + 认证在跨网环境要 30~80ms。
  2. **errno 翻译**：把 pymysql 原始错误映射为分类异常，并标注 `repairable`，
     供上层"错误自修复"决定是否值得重新生成 SQL。
  3. **资源治理**：结果集行数上限（服务端 `sql_select_limit` + 客户端 fetchmany 双保险）、
     语句超时（`MAX_EXECUTION_TIME`，由 MySQL 主动中断）。
  4. **会话级只读**：`SET SESSION TRANSACTION READ ONLY`。
     这是"只有 root 账号"场景下的关键第二道防线——
     即使应用层 SQL 校验被绕过，写操作也会被数据库拒绝。
  5. **慢查询诊断**：超过阈值记录 EXPLAIN，写入 metadata，不阻塞主响应。
"""

from __future__ import annotations

import logging
import queue
import threading
from time import perf_counter
from typing import Any, Callable, Tuple, List

import pymysql

from config import DB_CONFIG, APP_CONFIG

logger = logging.getLogger("chatbi.db")


# ==================== 异常 ====================
class QueryExecutionError(RuntimeError):
    """数据库执行失败后的统一异常。

    Attributes:
        error_type: 分类标识，直接用于 API 状态码映射。
        repairable: 是否值得把错误回灌给 LLM 重新生成 SQL。
        errno: MySQL 原始错误码（若有）。
    """

    def __init__(
        self,
        error_type: str,
        message: str,
        metadata: dict[str, Any] | None = None,
        repairable: bool = False,
        errno: int | None = None,
    ):
        super().__init__(message)
        self.error_type = error_type
        self.metadata = metadata or {}
        self.repairable = repairable
        self.errno = errno


# errno → (error_type, repairable)
# repairable=True 表示这是"SQL 写法/字段选择"层面的问题，重新生成有意义；
# False 表示重试只会放大故障（连接、超时、权限）。
_ERRNO_MAP: dict[int, tuple[str, bool]] = {
    # 语法 / 字段 / 表名 —— 可修复
    1064: ("database_sql_syntax", True),      # You have an error in your SQL syntax
    1146: ("database_sql_syntax", True),      # Table doesn't exist
    1054: ("database_sql_syntax", True),      # Unknown column
    1052: ("database_sql_syntax", True),      # Ambiguous column
    1109: ("database_sql_syntax", True),      # Unknown table in field list
    1055: ("database_sql_syntax", True),      # ONLY_FULL_GROUP_BY
    1111: ("database_sql_syntax", True),      # Invalid use of group function
    1241: ("database_sql_syntax", True),      # Operand should contain N column(s)
    1247: ("database_sql_syntax", True),      # Reference not supported
    1582: ("database_sql_syntax", True),      # Incorrect parameter count
    3065: ("database_sql_syntax", True),      # ORDER BY not in select list
    # 权限
    1045: ("database_permission_denied", False),
    1142: ("database_permission_denied", False),
    1143: ("database_permission_denied", False),
    1227: ("database_permission_denied", False),
    1792: ("database_permission_denied", False),   # read-only transaction 被拒
    1290: ("database_permission_denied", False),   # --read-only 选项
    # 超时 / 中断
    3024: ("database_query_timeout", False),  # MAX_EXECUTION_TIME 超限
    1317: ("database_query_timeout", False),  # Query interrupted (KILL)
    # 连接
    1044: ("database_connection_error", False),
    1049: ("database_connection_error", False),
    2003: ("database_connection_error", False),
    2006: ("database_connection_error", False),
    2013: ("database_connection_error", False),
    2055: ("database_connection_error", False),
}

_TIMEOUT_HINTS = ("maximum statement execution time exceeded", "execution time exceeded")

# 使用占位符替换值，避免 psycopg 类驱动对 '%' 的转义问题（pymysql 用 %s）
class _ConnectionPool:
    """轻量 LIFO 连接池。

    LIFO 而非 FIFO：刚归还的连接热乎、认证缓存有效，命中率更高。
    溢出策略：idle 队列满 + 已达上限时，等待 `pool_timeout` 秒后抛错，
    而不是无限创建连接把 MySQL 的 max_connections 打满。
    """

    def __init__(
        self,
        connect_kwargs: dict[str, Any],
        size: int,
        max_overflow: int,
        timeout: float,
        validate: bool = True,
        on_acquire: Callable[[Any], None] | None = None,
        connect_fn: Callable[..., Any] | None = None,
    ):
        self._connect_kwargs = dict(connect_kwargs)
        self._size = max(1, size)
        self._max_overflow = max(0, max_overflow)
        self._timeout = timeout
        self._validate = validate
        self._on_acquire = on_acquire
        self._connect_fn = connect_fn or pymysql.connect
        self._idle: queue.LifoQueue = queue.LifoQueue(maxsize=self._size)
        self._created = 0
        self._lock = threading.Lock()
        self._closed = False

    @property
    def created(self) -> int:
        return self._created

    def acquire(self) -> Any:
        if self._closed:
            raise QueryExecutionError(
                "database_connection_error", "连接池已关闭。", repairable=False
            )

        # 1) 直接取一个空闲连接
        while True:
            try:
                conn = self._idle.get_nowait()
            except queue.Empty:
                break
            if self._is_usable(conn):
                return conn
            self._discard_idle(conn)

        # 2) 未达上限则新建（含溢出额度）
        with self._lock:
            if self._created < self._size + self._max_overflow:
                self._created += 1
                try:
                    conn = self._connect_fn(**self._connect_kwargs)
                    self._init_session(conn)
                    return conn
                except pymysql.err.MySQLError as exc:
                    self._created -= 1
                    raise self._translate(exc) from exc
                except Exception as exc:
                    self._created -= 1
                    raise QueryExecutionError(
                        "database_connection_error",
                        f"建立数据库连接失败：{exc}",
                        metadata={"host": self._connect_kwargs.get("host")},
                    ) from exc

        # 3) 已达上限，等待其他请求归还
        try:
            conn = self._idle.get(timeout=self._timeout)
        except queue.Empty:
            raise QueryExecutionError(
                "database_pool_exhausted",
                "数据库连接池已耗尽，请稍后重试。",
                metadata={"pool_size": self._size, "max_overflow": self._max_overflow},
            ) from None

        if self._is_usable(conn):
            return conn
        self._discard_idle(conn)
        raise QueryExecutionError(
            "database_connection_error", "取到的连接已失效，请重试。"
        )

    def release(self, conn: Any, discard: bool = False) -> None:
        if conn is None:
            return
        if discard or self._closed:
            self._close_conn(conn)
            with self._lock:
                self._created = max(0, self._created - 1)
            return
        try:
            self._idle.put_nowait(conn)
        except queue.Full:
            # 空闲队列已满，说明并发回落，直接关掉多余的连接
            self._close_conn(conn)
            with self._lock:
                self._created = max(0, self._created - 1)

    def close(self) -> None:
        self._closed = True
        while True:
            try:
                conn = self._idle.get_nowait()
            except queue.Empty:
                break
            self._close_conn(conn)
        with self._lock:
            self._created = 0

    def _is_usable(self, conn: Any) -> bool:
        if conn is None or not getattr(conn, "open", False):
            return False
        if not self._validate:
            return True
        try:
            conn.ping(reconnect=False)
            return True
        except Exception:
            return False

    def _discard_idle(self, conn: Any) -> None:
        self._close_conn(conn)
        with self._lock:
            self._created = max(0, self._created - 1)

    def _init_session(self, conn: Any) -> None:
        if self._on_acquire is not None:
            self._on_acquire(conn)

    @staticmethod
    def _close_conn(conn: Any) -> None:
        try:
            conn.close()
        except Exception:
            pass

    @staticmethod
    def _translate(exc: Exception) -> QueryExecutionError:
        return translate_db_error(exc)


def translate_db_error(exc: Exception) -> QueryExecutionError:
    """把 pymysql 异常翻译为分类异常。"""
    errno = None
    message = str(exc)

    if isinstance(exc, pymysql.err.MySQLError) and exc.args:
        first = exc.args[0]
        if isinstance(first, int):
            errno = first
        if len(exc.args) > 1:
            message = str(exc.args[1])

    if isinstance(exc, (pymysql.err.OperationalError, pymysql.err.InterfaceError)):
        if not errno:
            error_type, repairable = "database_connection_error", False
        else:
            error_type, repairable = _ERRNO_MAP.get(
                errno, ("database_connection_error", False)
            )
    elif isinstance(exc, (pymysql.err.ProgrammingError, pymysql.err.DataError)):
        error_type, repairable = _ERRNO_MAP.get(errno, ("database_sql_syntax", True))
    else:
        error_type, repairable = _ERRNO_MAP.get(
            errno, ("database_execution_error", False)
        )

    lowered = message.lower()
    if any(hint in lowered for hint in _TIMEOUT_HINTS):
        error_type, repairable = "database_query_timeout", False

    return QueryExecutionError(
        error_type, message, metadata={"errno": errno}, repairable=repairable, errno=errno
    )


class DatabaseClient:
    """MySQL 客户端（池化 + 治理）。

    对外契约保持向后兼容：
        execute(sql) -> (columns, rows)          # 旧调用方继续可用
        execute_query(sql) -> (columns, rows, meta)
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        runtime_config: dict[str, Any] | None = None,
        connection_factory: Callable[[], Any] | None = None,
        time_fn: Callable[[], float] = perf_counter,
    ):
        self.config = dict(config or DB_CONFIG)
        self.runtime = dict(
            runtime_config or APP_CONFIG.get("database", {}).get("runtime", {})
        )
        self._time_fn = time_fn
        self._custom_factory = connection_factory

        self.max_rows = int(self.runtime.get("max_rows", 5000))
        self.statement_timeout_ms = int(self.runtime.get("statement_timeout_ms", 8000))
        self.slow_query_threshold_ms = float(
            self.runtime.get("slow_query_threshold_ms", 200)
        )
        self.explain_on_slow = bool(self.runtime.get("explain_on_slow", True))
        self.read_only_session = bool(self.runtime.get("read_only_session", True))

        # 最近一次查询的诊断信息，保持旧字段名兼容
        self.last_query_info: dict[str, Any] = {}

        if connection_factory is not None:
            # 测试注入：退化为"每次新建"，但保留全部治理逻辑
            self._pool: _ConnectionPool | None = None
        else:
            self._pool = _ConnectionPool(
                connect_kwargs=self._connect_kwargs(),
                size=int(self.runtime.get("pool_size", 5)),
                max_overflow=int(self.runtime.get("max_overflow", 5)),
                timeout=float(self.runtime.get("pool_timeout", 3)),
                validate=bool(self.runtime.get("validate_on_acquire", True)),
                on_acquire=self._init_session,
            )

    # ---------- 连接管理 ----------
    def _connect_kwargs(self) -> dict[str, Any]:
        kwargs = dict(self.config)
        kwargs.setdefault("autocommit", True)
        return kwargs

    def _new_connection(self) -> Any:
        if self._custom_factory is not None:
            conn = self._custom_factory()
            self._init_session(conn)
            return conn
        return pymysql.connect(**self._connect_kwargs())

    def _init_session(self, conn: Any) -> None:
        """会话级加固：只读 + 语句超时 + 行数上限。"""
        statements: list[str] = []
        if self.read_only_session:
            # 关键：即使使用 root 账号，写操作也会在此被数据库拒绝
            statements.append("SET SESSION TRANSACTION READ ONLY")
        if self.statement_timeout_ms > 0:
            statements.append(
                f"SET SESSION MAX_EXECUTION_TIME = {self.statement_timeout_ms}"
            )
        if self.max_rows > 0:
            # 服务端限制返回行数（多取 1 行用于判定截断）
            statements.append(f"SET SESSION sql_select_limit = {self.max_rows + 1}")
        if not statements:
            return
        try:
            with conn.cursor() as cursor:
                for statement in statements:
                    cursor.execute(statement)
        except Exception as exc:  # pragma: no cover - 取决于 MySQL 版本/权限
            logger.warning("会话初始化部分失败（将依赖客户端侧治理）：%s", exc)

    def _acquire(self) -> Any:
        if self._pool is not None:
            return self._pool.acquire()
        return self._new_connection()

    def _release(self, conn: Any, discard: bool = False) -> None:
        if self._pool is not None:
            self._pool.release(conn, discard=discard)
        else:
            try:
                conn.close()
            except Exception:
                pass

    # ---------- 查询执行 ----------
    def execute(self, sql: str) -> Tuple[List[str], List[tuple]]:
        """兼容旧调用方：只返回 (columns, rows)。"""
        columns, rows, _meta = self.execute_query(sql)
        return columns, rows

    def execute_query(
        self,
        sql: str,
        max_rows: int | None = None,
        security: Any = None,
        user: Any = None,
    ) -> Tuple[List[str], List[tuple], dict[str, Any]]:
        """执行只读查询，返回 (columns, rows, metadata)。

        Args:
            security: 可选的安全管理器；传入时会在此层再做一次校验（纵深防御）。
            user: 权限上下文，配合 security 使用。
        """
        executed_sql = sql
        secured = None
        if security is not None and user is not None:
            secured = security.prepare(sql, user)
            executed_sql = secured.sql

        limit = self.max_rows if max_rows is None else max_rows
        conn = self._acquire()
        discard = True
        try:
            columns: List[str] = []
            rows: List[tuple] = []
            started = self._time_fn()
            try:
                with conn.cursor() as cursor:
                    cursor.execute(executed_sql)
                    columns = (
                        [desc[0] for desc in cursor.description]
                        if cursor.description
                        else []
                    )
                    fetched = (
                        cursor.fetchmany(limit + 1) if limit > 0 else cursor.fetchall()
                    )
            except pymysql.err.MySQLError as exc:
                raise translate_db_error(exc) from exc

            elapsed_ms = (self._time_fn() - started) * 1000
            truncated = bool(limit > 0 and len(fetched) > limit)
            rows = list(fetched[:limit]) if truncated else list(fetched)

            meta: dict[str, Any] = {
                "duration_ms": round(elapsed_ms, 2),
                "row_count": len(rows),
                "truncated": truncated,
                "row_limit": limit,
                "slow_query": elapsed_ms >= self.slow_query_threshold_ms,
                "statement_timeout_ms": self.statement_timeout_ms,
            }
            if meta["slow_query"] and self.explain_on_slow and executed_sql.lstrip().lower().startswith(("select", "with")):
                meta["explain"] = self._safe_explain(conn, executed_sql)

            self.last_query_info = dict(meta)
            discard = False
            return columns, rows, meta
        finally:
            self._release(conn, discard=discard)

    def _safe_explain(self, conn: Any, sql: str) -> Any:
        """对慢查询取执行计划，失败不影响主流程。"""
        try:
            with conn.cursor() as cursor:
                cursor.execute(f"EXPLAIN {sql}")
                plan_rows = cursor.fetchall()
                plan_columns = (
                    [desc[0] for desc in cursor.description] if cursor.description else []
                )
                return [dict(zip(plan_columns, row)) for row in plan_rows]
        except Exception as exc:  # pragma: no cover
            logger.debug("EXPLAIN 失败：%s", exc)
            return None

    def validate_connection(self) -> bool:
        """验证数据库连接是否有效。"""
        conn = None
        try:
            conn = self._acquire()
            with conn.cursor() as cursor:
                cursor.execute("SELECT 1")
                cursor.fetchall()
            return True
        except Exception:
            return False
        finally:
            if conn is not None:
                self._release(conn)

    def close(self) -> None:
        if self._pool is not None:
            self._pool.close()
