"""SQL 对比指标引擎。

核心思路: 把 SQL 拆成可独立比较的组件（列、表、条件、GROUP BY、ORDER BY、LIMIT）,
分别计算每个维度的匹配度,再加权汇总为一个 0~1 的结构得分。

括号深度感知: 所有拆分函数都在 depth=0 时才切分,避免误拆子查询内部的逗号/AND。
"""

from __future__ import annotations

import datetime
import re
from decimal import Decimal, InvalidOperation
from typing import Dict, List, Tuple

# SQL 子句关键字,顺序决定了 _clause_spans 的扫描顺序
_CLAUSE_PATTERNS = [
    ("select", re.compile(r"\bselect\b", re.IGNORECASE)),
    ("from", re.compile(r"\bfrom\b", re.IGNORECASE)),
    ("join", re.compile(r"\bjoin\b", re.IGNORECASE)),
    ("where", re.compile(r"\bwhere\b", re.IGNORECASE)),
    ("group_by", re.compile(r"\bgroup\s+by\b", re.IGNORECASE)),
    ("having", re.compile(r"\bhaving\b", re.IGNORECASE)),
    ("order_by", re.compile(r"\border\s+by\b", re.IGNORECASE)),
    ("limit", re.compile(r"\blimit\b", re.IGNORECASE)),
]

_BOOL_OP_RE = re.compile(r"\b(and|or)\b", re.IGNORECASE)     # AND / OR 边界
_QUALIFIER_RE = re.compile(r"\b[a-z_][a-z0-9_]*\.")          # 表别名前缀,如 o. / e.
_SPACE_RE = re.compile(r"\s+")


def normalize_sql(sql: str) -> str:
    """归一化 SQL: 去首尾空白、去分号、合并空白、转小写。用于精确匹配判定。"""
    if sql is None:
        return ""
    s = sql.strip()
    if s.endswith(";"):
        s = s[:-1]
    s = _SPACE_RE.sub(" ", s)
    return s.strip().lower()


def _clause_spans(sql: str) -> Dict[str, Tuple[int, int]]:
    """定位各子句关键字在括号深度为 0 时的 (start, end) 位置。

    为什么要深度感知? 防止子查询里的 SELECT/FROM 被误识别为主子句。
    """
    spans: Dict[str, Tuple[int, int]] = {}
    lower = sql.lower()
    depth = 0
    i = 0
    n = len(sql)
    while i < n:
        ch = lower[i]
        if ch == "(":
            depth += 1
            i += 1
            continue
        if ch == ")":
            depth = max(0, depth - 1)
            i += 1
            continue
        if depth == 0:
            for key, pattern in _CLAUSE_PATTERNS:
                m = pattern.match(lower, i)
                if m:
                    spans.setdefault(key, (i, m.end()))
                    break
        i += 1
    return spans


def _extract_clause(sql: str, key: str) -> str:
    """取出某个子句关键字之后、下一个子句关键字之前的内容。"""
    spans = _clause_spans(sql)
    if key not in spans:
        return ""
    start = spans[key][1]
    next_starts = [s for k, (s, _) in spans.items() if s > spans[key][0]]
    end = min(next_starts) if next_starts else len(sql)
    return sql[start:end].strip()


def _split_top_level(text: str, sep: str = ",") -> List[str]:
    """在括号深度为 0 处按分隔符切分,用于 SELECT 列和 GROUP BY 字段的拆分。"""
    parts: List[str] = []
    depth = 0
    cur: List[str] = []
    for ch in text:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        if ch == sep and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    if cur:
        parts.append("".join(cur).strip())
    return [p for p in parts if p]


def _split_conditions(text: str) -> List[str]:
    """在括号深度为 0 处按 AND/OR 切分 WHERE 条件。

    注意: re.match 返回的 m.end() 已是绝对位置,直接赋值给 last 即可,
    之前有版本误写成 last = i + m.end(),导致从第 2 个 AND 起全部条件畸变。
    """
    parts: List[str] = []
    depth = 0
    last = 0
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        elif depth == 0:
            m = _BOOL_OP_RE.match(text, i)
            if m:
                parts.append(text[last:i].strip())
                last = m.end()        # ← 关键: m.end() 已经是绝对位置
                i = last
                continue
        i += 1
    parts.append(text[last:].strip())
    return [p for p in parts if p]


def _normalize_expr(expr: str) -> str:
    """归一化列/表达式: 去限定符 (o.)、去 AS 别名、压缩运算符空格。"""
    e = _SPACE_RE.sub(" ", expr.strip()).lower()
    e = _QUALIFIER_RE.sub("", e)
    e = re.sub(r"\s+as\s+[a-z_][a-z0-9_]*$", "", e)
    e = re.sub(r"\s*([*+/=<>,-])\s*", r"\1", e)
    e = re.sub(r"\s*\(\s*", "(", e)
    e = re.sub(r"\s*\)\s*", ")", e)
    return e.strip()


def _normalize_condition(cond: str) -> str:
    """归一化 WHERE 条件,与 _normalize_expr 类似但保留字面量完整性。"""
    c = _SPACE_RE.sub(" ", cond.strip()).lower()
    c = _QUALIFIER_RE.sub("", c)
    c = re.sub(r"\s*([=<>+*/])\s*", r"\1", c)
    c = re.sub(r"\s*\(\s*", "(", c)
    c = re.sub(r"\s*\)\s*", ")", c)
    return c.strip()


def _normalize_order(expr: str) -> str:
    """归一化 ORDER BY 项,去掉尾部的 ASC/DESC (方向独立比较)。"""
    e = _normalize_expr(expr)
    return re.sub(r"\s+(asc|desc)$", "", e)


def _first_table(text: str) -> str:
    """从 FROM/JOIN 子句中提取第一个表名 (跳过 LEFT/INNER 等修饰词)。"""
    text = text.strip()
    m = re.match(r"^(?:left|right|inner|outer|cross|full)\s+", text, re.IGNORECASE)
    if m:
        text = text[m.end():].strip()
    tokens = text.split()
    if not tokens:
        return ""
    return tokens[0].strip("`").lower()


def _parse_tables(sql: str) -> List[str]:
    """提取 SQL 中所有 FROM + JOIN 的表名,去重排序。"""
    tables: List[str] = []
    spans = _clause_spans(sql)
    if "from" in spans:
        table = _first_table(_extract_clause(sql, "from"))
        if table:
            tables.append(table)
    for m in re.finditer(r"\bjoin\b", sql, re.IGNORECASE):
        table = _first_table(sql[m.end():])
        if table:
            tables.append(table)
    return sorted(set(tables))


def parse_components(sql: str) -> Dict[str, object]:
    """把一条 SQL 拆成可比较的组件字典。"""
    comp: Dict[str, object] = {
        "columns": [],
        "tables": [],
        "conditions": [],
        "group_by": [],
        "order_by": [],
        "having": [],
        "limit": None,
    }

    select_part = _extract_clause(sql, "select")
    if select_part:
        comp["columns"] = [_normalize_expr(c) for c in _split_top_level(select_part)]

    comp["tables"] = _parse_tables(sql)

    where_part = _extract_clause(sql, "where")
    if where_part:
        comp["conditions"] = [_normalize_condition(c) for c in _split_conditions(where_part)]

    having_part = _extract_clause(sql, "having")
    if having_part:
        comp["having"] = [_normalize_condition(c) for c in _split_conditions(having_part)]

    group_part = _extract_clause(sql, "group_by")
    if group_part:
        comp["group_by"] = [_normalize_expr(c) for c in _split_top_level(group_part)]

    order_part = _extract_clause(sql, "order_by")
    if order_part:
        comp["order_by"] = [_normalize_order(c) for c in _split_top_level(order_part)]

    limit_part = _extract_clause(sql, "limit")
    if limit_part:
        m = re.search(r"\d+", limit_part)
        comp["limit"] = m.group(0) if m else limit_part

    return comp


def _token_set(text: str) -> set:
    """从文本中提取 token 集合 (字母数字下划线),用于 Jaccard 相似度计算。"""
    return set(re.findall(r"[a-z0-9_]+", text))


def _jaccard(a: set, b: set) -> float:
    """Jaccard 相似度 = |A∩B| / |A∪B|。两空集视为完全相同。"""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _condition_recall(gold_conds: List[str], gen_conds: List[str]) -> float:
    """条件召回率: 黄金条件中有多少被生成条件覆盖。

    匹配策略: 对每条黄金条件,在尚未被使用过的生成条件中找 Jaccard ≥ 0.7 的匹配。
    使用贪心 + 不重复匹配,确保一条生成条件不会被多条黄金条件复用。
    """
    if not gold_conds:
        return 1.0
    matched = 0
    used = [False] * len(gen_conds)
    for gc in gold_conds:
        gt = _token_set(gc)
        for idx, dc in enumerate(gen_conds):
            if used[idx]:
                continue
            if _jaccard(gt, _token_set(dc)) >= 0.7:
                used[idx] = True
                matched += 1
                break
    return matched / len(gold_conds)


def compare(golden_sql: str, generated_sql: str) -> Dict[str, object]:
    """对比黄金 SQL 与生成 SQL,返回完整结构指标。

    structure_score 加权公式 (权重反映各维度的重要性):
      0.30 × select_jaccard    ← 列/表达式正确最关键
      0.25 × table_match       ← 表漏了 JOIN 一定不对
      0.25 × condition_recall  ← 过滤条件漏了统计范围就错了
      0.08 × group_match
      0.07 × order_match
      0.05 × limit_match
    """
    exact = normalize_sql(golden_sql) == normalize_sql(generated_sql)

    g = parse_components(golden_sql)
    d = parse_components(generated_sql)

    select_jaccard = _jaccard(set(g["columns"]), set(d["columns"]))
    select_match = set(g["columns"]) == set(d["columns"])
    table_match = set(g["tables"]) == set(d["tables"])
    condition_recall = _condition_recall(g["conditions"], d["conditions"])
    condition_match = condition_recall >= 1.0
    group_match = set(g["group_by"]) == set(d["group_by"])
    order_match = set(g["order_by"]) == set(d["order_by"])
    limit_match = g["limit"] == d["limit"]

    score = (
        0.30 * select_jaccard
        + 0.25 * (1.0 if table_match else 0.0)
        + 0.25 * condition_recall
        + 0.08 * (1.0 if group_match else 0.0)
        + 0.07 * (1.0 if order_match else 0.0)
        + 0.05 * (1.0 if limit_match else 0.0)
    )

    return {
        "exact_match": exact,
        "select_match": select_match,
        "select_jaccard": round(select_jaccard, 4),
        "table_match": table_match,
        "condition_recall": round(condition_recall, 4),
        "condition_match": condition_match,
        "group_match": group_match,
        "order_match": order_match,
        "limit_match": limit_match,
        "structure_score": round(score, 4),
    }


def _to_decimal(value) -> "Decimal | None":
    """尽量把值转成 Decimal，用于数值等价比较。

    bool 不参与数值比较（True == 1 在业务上没有意义）。
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float, Decimal)):
        try:
            return Decimal(str(value))
        except (InvalidOperation, ValueError):
            return None
    if isinstance(value, str):
        text = value.strip().replace(",", "")
        if not text:
            return None
        try:
            return Decimal(text)
        except (InvalidOperation, ValueError):
            return None
    return None


def _as_date(value):
    """把 date/datetime 或 ISO 字符串统一成 date/datetime，便于跨类型比较。"""
    if isinstance(value, (datetime.date, datetime.datetime)):
        return value
    if isinstance(value, str):
        text = value.strip()
        for parser in (datetime.datetime.fromisoformat, datetime.date.fromisoformat):
            try:
                return parser(text)
            except ValueError:
                continue
    return None


def _to_date_only(value):
    """取出日期部分（datetime → date）。"""
    if isinstance(value, datetime.datetime):
        return value.date()
    return value


def cell_equal(a, b, tol: float = 1e-6) -> bool:
    """单元格级等价判定。

    处理三类历史上会产生假阴性的情况：
      1. `Decimal("100.00")` 与 `100.0` —— 数值等价
      2. 浮点误差 `0.1 + 0.2` 与 `0.3` —— 相对容差
      3. 数据库返回 date/datetime 而黄金 SQL 侧是字符串 —— 统一解析后比较

    日期比较策略：两侧都是 datetime 时比完整时间；类型不一致（date vs datetime）
    时只比日期部分——BI 场景下同一列的日期/时间混用通常来自驱动差异，不是语义差异。
    """
    if a is None or b is None:
        return a is b or a == b

    da, db = _to_decimal(a), _to_decimal(b)
    if da is not None and db is not None:
        if da == db:
            return True
        scale = max(abs(da), abs(db), Decimal(1))
        return abs(da - db) <= Decimal(str(tol)) * scale

    date_a, date_b = _as_date(a), _as_date(b)
    if date_a is not None and date_b is not None:
        if isinstance(date_a, datetime.datetime) and isinstance(date_b, datetime.datetime):
            return date_a == date_b
        return _to_date_only(date_a) == _to_date_only(date_b)

    return str(a).strip() == str(b).strip()


def rows_equal(
    rows_a: list,
    rows_b: list,
    order_sensitive: bool = False,
    tol: float = 1e-6,
) -> bool:
    """比较两个结果集是否等价。

    默认忽略行顺序（SQL 未写 ORDER BY 时顺序本就不保证）。
    早期实现用 `repr` 字符串比较，会让 `Decimal('100.00')` 与 `100.0` 判为不等，
    在财务场景下产生大量假阴性。
    """
    rows_a = [tuple(row) for row in (rows_a or [])]
    rows_b = [tuple(row) for row in (rows_b or [])]

    if len(rows_a) != len(rows_b):
        return False
    if not rows_a and not rows_b:
        return True

    def sort_key(row):
        return tuple("" if value is None else str(value) for value in row)

    if not order_sensitive:
        rows_a = sorted(rows_a, key=sort_key)
        rows_b = sorted(rows_b, key=sort_key)

    for row_a, row_b in zip(rows_a, rows_b):
        if len(row_a) != len(row_b):
            return False
        if not all(cell_equal(x, y, tol) for x, y in zip(row_a, row_b)):
            return False
    return True