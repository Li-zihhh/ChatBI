"""SQL 安全校验、行级权限与结果脱敏。

设计原则（针对"个人项目只有 root 账号"这一现实约束）：
  1. **不依赖数据库账号权限**：所有校验都在应用层完成，root 账号也能被约束；
     另在连接池侧配合 `SET SESSION TRANSACTION READ ONLY` 做第二道防线。
  2. **白名单而非黑名单**：用 SQL AST（sqlglot）校验，只允许"单条 SELECT + 表白名单内的表"。
     黑名单永远漏（`INTO OUTFILE` / `LOAD_FILE` / `SLEEP` / `information_schema` ...），
     而且会被字符串字面量误伤（`WHERE name = 'update'`）。
  3. **fail-close**：未知角色、缺失权限上下文、无法解析的 SQL 一律拒绝，
     绝不"降级到 admin"或"跳过过滤条件"。
  4. **行级权限用子查询包裹而非拼接 WHERE**：把目标表替换为
     `(SELECT * FROM t WHERE region = 'X') AS 原别名`，
     这样无论 LLM 生成多复杂的 CTE / 派生表 / UNION，过滤都必然生效。
  5. **脱敏按列血缘判定**：解析 SELECT 投影，追踪输出别名引用了哪些底层列，
     使 `SELECT net_amount AS revenue` / `SELECT SUM(net_amount)` 都能被识别。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError

from config import APP_CONFIG


# ==================== 异常 ====================
class SecurityError(Exception):
    """权限校验或 SQL 安全检查失败。

    code 用于区分"可修复的生成错误"与"终止性的权限拒绝"：
      - unparsable_sql   : LLM 生成的 SQL 无法解析 → 可回灌 Prompt 重新生成
      - 其余             : 终止性拒绝，重试没有意义
    """

    # 可修复：说明是 LLM 生成质量问题，而不是越权意图
    REPAIRABLE_CODES = frozenset({"unparsable_sql"})

    def __init__(
        self,
        message: str,
        code: str = "security_violation",
        metadata: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.code = code
        self.metadata = metadata or {}

    @property
    def is_repairable(self) -> bool:
        return self.code in self.REPAIRABLE_CODES

    @property
    def error_type(self) -> str:
        return "security" if not self.is_repairable else "database_sql_syntax"


# ==================== 权限上下文 ====================
@dataclass(slots=True)
class UserContext:
    """当前请求的最小权限上下文。

    注意：默认角色是**最小权限**（sales），不是 admin。
    admin 只能由显式配置或受信身份来源注入。
    """

    user_id: str
    role: str = "sales"
    region: str | None = None
    allowed_regions: list[str] = field(default_factory=list)

    def regions(self) -> list[str]:
        """返回用户可访问的区域列表（去重、去空）。"""
        values = [r for r in ([self.region] + list(self.allowed_regions)) if r]
        # dict.fromkeys 保持顺序去重
        return list(dict.fromkeys(str(v).strip() for v in values if str(v).strip()))

    @classmethod
    def demo_admin(cls) -> "UserContext":
        return cls(user_id="demo_admin", role="admin")


# ==================== 角色策略 ====================
@dataclass(slots=True)
class SecurityPolicy:
    """单个角色的数据访问策略。

    row_level_filters: [(表名, 列名)]，命中后对该表的每次出现做子查询包裹过滤。
    masked_columns   : 需要脱敏的底层列名（小写）。
    allowed_tables   : 该角色可见的表；None 表示"不额外限制"（仅受系统白名单约束）。
                       ——仅靠行级过滤不够：finance_expenses 没有 region 列，
                       销售角色能直接读到全公司费用，这属于数据域越权。
    """

    role: str
    row_level_filters: list[tuple[str, str]] = field(default_factory=list)
    masked_columns: set[str] = field(default_factory=set)
    allowed_tables: set[str] | None = None


_SENSITIVE_PERSONAL = {"customer_phone", "customer_email"}
_SENSITIVE_FINANCE = {
    "gross_amount",
    "net_amount",
    "material_cost",
    "standard_cost",
    "labor_cost",
    "profit",
}

# 角色注册表：新增角色只需在此登记。
# 注意 sales 同时对 dim_customers 与 sales_orders 的 region 做过滤——
# 后者是关键：事实表自身也带 region 列，只过滤维度表会被直接绕过。
_SALES_TABLES = {
    "dim_customers",
    "dim_products",
    "sales_orders",
    "exchange_rates",
}

ROLE_POLICIES: dict[str, SecurityPolicy] = {
    "admin": SecurityPolicy(role="admin"),
    "finance": SecurityPolicy(
        role="finance",
        masked_columns=set(_SENSITIVE_PERSONAL),
    ),
    "sales": SecurityPolicy(
        role="sales",
        row_level_filters=[("dim_customers", "region"), ("sales_orders", "region")],
        masked_columns=set(_SENSITIVE_PERSONAL) | set(_SENSITIVE_FINANCE),
        # 销售不应看到期间费用（该表没有 region 列，行级过滤覆盖不到）
        allowed_tables=_SALES_TABLES,
    ),
}


def get_policy(role: str | None) -> SecurityPolicy:
    """按角色取策略。未知角色 fail-close，绝不回退到 admin。"""
    normalized = (role or "").strip().lower()
    if not normalized:
        raise SecurityError("缺少角色信息，拒绝访问。", code="missing_role")
    policy = ROLE_POLICIES.get(normalized)
    if policy is None:
        raise SecurityError(
            f"未知角色：{role!r}，拒绝访问。",
            code="unknown_role",
            metadata={"known_roles": sorted(ROLE_POLICIES)},
        )
    return policy


# ==================== SQL 防护 ====================
# 明确禁止的语句节点（白名单只放行 SELECT / UNION，这里是纵深防御）
_FORBIDDEN_NODES: tuple[type[exp.Expression], ...] = tuple(
    node
    for node in (
        getattr(exp, "Insert", None),
        getattr(exp, "Update", None),
        getattr(exp, "Delete", None),
        getattr(exp, "Drop", None),
        getattr(exp, "Create", None),
        getattr(exp, "Alter", None),
        getattr(exp, "TruncateTable", None),
        getattr(exp, "Grant", None),
        getattr(exp, "Command", None),
    )
    if node is not None
)

# 危险函数：文件读写、延时注入、UDF 提权
_FORBIDDEN_FUNCTIONS = frozenset(
    {
        "load_file",
        "sleep",
        "benchmark",
        "get_lock",
        "release_lock",
        "master_pos_wait",
        "updatexml",
        "extractvalue",
        "sys_exec",
        "sys_eval",
        "load_extension",
    }
)

_FENCE_RE = re.compile(r"^```(?:sql)?\s*|\s*```$", re.IGNORECASE)
_WS_RE = re.compile(r"\s+")


def _normalize_sql(sql: str) -> str:
    """清理 markdown 围栏与首尾分号，保留 SQL 主体。"""
    text = (sql or "").strip()
    text = _FENCE_RE.sub("", text).strip()
    # 去掉结尾的多个分号（合法场景：LLM 习惯性加分号）
    text = text.rstrip(";").strip()
    return text


@dataclass(slots=True)
class SecuredQuery:
    """已通过安全校验的查询，携带 AST 供脱敏使用。"""

    sql: str
    ast: exp.Expression
    policy: SecurityPolicy
    user: UserContext

    def mask(
        self,
        columns: list[str],
        rows: Iterable[tuple | list],
    ) -> tuple[list[str], list[tuple]]:
        return mask_rows(self.ast, self.policy, columns, rows)

    def masked_column_count(self, columns: list[str]) -> int:
        """有多少列会被脱敏（用于 metadata 观测）。"""
        return len(sensitive_indexes(self.ast, self.policy, columns))


class QuerySecurityManager:
    """负责 SQL 白名单校验、行级过滤和结果脱敏。"""

    def __init__(self, allowed_tables: list[str] | None = None):
        configured = allowed_tables or APP_CONFIG.get("security", {}).get("allowed_tables") or []
        self.allowed_tables = {t.lower() for t in configured}
        # 允许显式引用当前库（如 chatbi_mvp.sales_orders），跨库访问一律拒绝
        self.allowed_databases: set[str] = set()
        db_cfg = APP_CONFIG.get("database", {})
        source_cfg = (db_cfg.get("sources") or {}).get(db_cfg.get("default_source", ""), {})
        db_name = source_cfg.get("database")
        if db_name:
            self.allowed_databases.add(str(db_name).lower())

    # ---------- 对外主入口 ----------
    def prepare(
        self,
        sql: str,
        user: UserContext | None = None,
        policy: SecurityPolicy | None = None,
    ) -> SecuredQuery:
        """校验 SQL 并注入行级过滤，返回可执行语句 + AST。"""
        user_context = user or UserContext.demo_admin()
        resolved_policy = policy or get_policy(user_context.role)

        normalized = _normalize_sql(sql)
        if not normalized:
            raise SecurityError("SQL 不能为空。", code="empty_sql")

        ast = self._parse(normalized)
        self._assert_safe(ast, resolved_policy)
        ast = self._enforce_row_level(ast, user_context, resolved_policy)

        return SecuredQuery(
            sql=self._render(ast),
            ast=ast,
            policy=resolved_policy,
            user=user_context,
        )

    def secure_sql(self, sql: str, user: UserContext | None = None) -> str:
        """兼容入口：只返回加固后的 SQL 文本。"""
        return self.prepare(sql, user).sql

    def mask_result(
        self,
        columns: list[str],
        rows: Iterable[tuple | list],
        user: UserContext | None = None,
        secured: SecuredQuery | None = None,
    ) -> tuple[list[str], list[tuple]]:
        """兼容入口：无 AST 时退化为按列名脱敏（有 AST 时按血缘，更严格）。"""
        user_context = user or UserContext.demo_admin()
        policy = secured.policy if secured else get_policy(user_context.role)
        ast = secured.ast if secured else None
        return mask_rows(ast, policy, columns, rows)

    # ---------- 解析与校验 ----------
    @staticmethod
    def _parse(sql: str) -> exp.Expression:
        # MySQL 版本注释 /*!50000 ... */ 可绕过解析，直接拒绝
        if "/*!" in sql:
            raise SecurityError("检测到 MySQL 版本注释，拒绝执行。", code="statement_not_allowed")
        # 高信号危险关键字：sqlglot 可能解析失败（如 INTO OUTFILE），
        # 这里先做分类修正，避免被误判成"可修复的生成错误"而去重试。
        lowered = sql.lower()
        for keyword in ("into outfile", "into dumpfile"):
            if keyword in lowered:
                raise SecurityError(
                    f"禁止 {keyword.upper()}。", code="statement_not_allowed"
                )
        try:
            statements = [s for s in sqlglot.parse(sql, dialect="mysql") if s is not None]
        except ParseError as exc:
            raise SecurityError(
                f"SQL 解析失败：{exc}", code="unparsable_sql"
            ) from exc

        if not statements:
            raise SecurityError("SQL 解析结果为空。", code="unparsable_sql")
        if len(statements) > 1:
            raise SecurityError("检测到多语句执行风险。", code="statement_not_allowed")

        ast = statements[0]
        if not isinstance(ast, (exp.Select, exp.Union)):
            raise SecurityError(
                f"只允许执行查询语句，当前为 {type(ast).__name__}。",
                code="statement_not_allowed",
            )
        return ast

    def _assert_safe(
        self,
        ast: exp.Expression,
        policy: SecurityPolicy | None = None,
    ) -> None:
        cte_names = {
            (cte.alias_or_name or "").lower() for cte in ast.find_all(exp.CTE)
        }

        for node in ast.walk():
            if isinstance(node, _FORBIDDEN_NODES):
                raise SecurityError(
                    f"禁止的语句类型：{type(node).__name__}",
                    code="statement_not_allowed",
                )
            # SELECT ... INTO OUTFILE / DUMPFILE
            if isinstance(node, exp.Into):
                raise SecurityError("禁止 INTO OUTFILE / DUMPFILE。", code="statement_not_allowed")

        for func in list(ast.find_all(exp.Func)) + list(ast.find_all(exp.Anonymous)):
            name = (getattr(func, "name", "") or "").lower()
            if name in _FORBIDDEN_FUNCTIONS:
                raise SecurityError(
                    f"禁止的函数：{name}", code="function_not_allowed"
                )

        for table in ast.find_all(exp.Table):
            # CTE 引用不是真实表（e.g. WITH c AS (...) SELECT * FROM c），跳过；
            # CTE 体内的真实表仍会在此被校验，无法借 CTE 名称绕过白名单。
            if not table.db and (table.name or "").lower() in cte_names:
                continue
            self._assert_table_allowed(table, policy)

    def _assert_table_allowed(
        self,
        table: exp.Table,
        policy: SecurityPolicy | None = None,
    ) -> None:
        name = (table.name or "").lower()
        database = (table.db or "").lower()

        # 子查询别名不会成为 Table 节点，这里拿到的一定是真实表名
        if database and database not in self.allowed_databases:
            raise SecurityError(
                f"禁止跨库访问：{database}.{name}",
                code="table_not_allowed",
                metadata={"database": database},
            )
        if name not in self.allowed_tables:
            raise SecurityError(
                f"未授权的表：{name}",
                code="table_not_allowed",
                metadata={"table": name},
            )
        # 角色级数据域限制（如 sales 不可访问 finance_expenses）
        if policy is not None and policy.allowed_tables is not None:
            if name not in policy.allowed_tables:
                raise SecurityError(
                    f"角色 {policy.role} 无权访问表：{name}",
                    code="table_not_allowed",
                    metadata={
                        "table": name,
                        "role": policy.role,
                        "allowed_tables": sorted(policy.allowed_tables),
                    },
                )

    # ---------- 行级过滤 ----------
    def _enforce_row_level(
        self,
        ast: exp.Expression,
        user: UserContext,
        policy: SecurityPolicy,
    ) -> exp.Expression:
        if not policy.row_level_filters:
            return ast

        for table_name, column_name in policy.row_level_filters:
            regions = user.regions()
            if not regions:
                # fail-close：需要区域约束却拿不到区域，直接拒绝而不是放行
                raise SecurityError(
                    f"角色 {policy.role} 缺少区域权限信息，拒绝访问。",
                    code="missing_permission_context",
                    metadata={"table": table_name, "column": column_name},
                )

            target = table_name.lower()
            # 先收集再替换，避免遍历过程中修改 AST
            matches = [
                table
                for table in ast.find_all(exp.Table)
                if (table.name or "").lower() == target
            ]
            for table in matches:
                alias = table.alias_or_name or table_name
                source = table.copy()
                source.set("alias", None)

                predicate = self._region_predicate(column_name, regions)
                inner = exp.select("*").from_(source).where(predicate)
                table.replace(inner.subquery(alias))

        return ast

    @staticmethod
    def _region_predicate(column_name: str, regions: list[str]) -> exp.Expression:
        column = exp.column(column_name)
        if len(regions) == 1:
            return exp.EQ(this=column, expression=exp.Literal.string(regions[0]))
        return exp.In(
            this=column,
            expressions=[exp.Literal.string(r) for r in regions],
        )

    @staticmethod
    def _render(ast: exp.Expression) -> str:
        rendered = ast.sql(dialect="mysql")
        return rendered.rstrip(";")


# ==================== 结果脱敏 ====================
def _norm_label(value: Any) -> str:
    return _WS_RE.sub("", str(value or "")).lower()


def _sensitive_labels(ast: exp.Expression, masked: set[str]) -> set[str]:
    """收集"输出标签"中引用了敏感列的那些，用于覆盖别名与聚合表达式。

    覆盖三类写法：
      SELECT net_amount                  → 标签 net_amount（列名兜底也能命中）
      SELECT net_amount AS revenue       → 标签 revenue
      SELECT SUM(net_amount)             → 标签 sum(net_amount)
    """
    if not masked:
        return set()

    labels: set[str] = set()
    for select in ast.find_all(exp.Select):
        for projection in select.expressions:
            sources = {(col.name or "").lower() for col in projection.find_all(exp.Column)}
            if not (sources & masked):
                continue
            alias = projection.alias
            if alias:
                labels.add(_norm_label(alias))
            labels.add(_norm_label(projection.sql(dialect="mysql")))
    return labels


def sensitive_indexes(
    ast: exp.Expression | None,
    policy: SecurityPolicy,
    columns: list[str],
) -> set[int]:
    """返回结果集中应被脱敏的列下标。

    有 AST 时按"列名 + 血缘标签"双通道判定，防止别名/聚合绕过；
    无 AST 时退化为列名匹配。
    """
    if not policy.masked_columns:
        return set()

    labels = _sensitive_labels(ast, policy.masked_columns) if ast is not None else set()
    indexes: set[int] = set()
    for index, column in enumerate(columns):
        name = _norm_label(column)
        if name in policy.masked_columns or name in labels:
            indexes.add(index)
    return indexes


def mask_rows(
    ast: exp.Expression | None,
    policy: SecurityPolicy,
    columns: list[str],
    rows: Iterable[tuple | list],
) -> tuple[list[str], list[tuple]]:
    """按策略把敏感列替换为 ***。"""
    materialized = [tuple(row) for row in rows]
    masked_indexes = sensitive_indexes(ast, policy, columns)
    if not masked_indexes:
        return columns, materialized

    masked_rows: list[tuple] = []
    for row in materialized:
        values = list(row)
        for index in masked_indexes:
            if index < len(values):
                values[index] = "***"
        masked_rows.append(tuple(values))

    return columns, masked_rows
