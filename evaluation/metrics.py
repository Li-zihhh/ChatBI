"""SQL 对比指标。

提供 SQL 归一化、组件解析、结构对比与结果集比较，用于 Text2SQL 评估。
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple


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

_BOOL_OP_RE = re.compile(r"\b(and|or)\b", re.IGNORECASE)
_QUALIFIER_RE = re.compile(r"\b[a-z_][a-z0-9_]*\.")
_SPACE_RE = re.compile(r"\s+")


def normalize_sql(sql: str) -> str:
    """归一化 SQL：去首尾空白、去分号、合并空白、转小写。"""
    if sql is None:
        return ""
    s = sql.strip()
    if s.endswith(";"):
        s = s[:-1]
    s = _SPACE_RE.sub(" ", s)
    return s.strip().lower()


def _clause_spans(sql: str) -> Dict[str, Tuple[int, int]]:
    """定位各子句关键字在括号深度为 0 时的位置。"""
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
    """在括号深度为 0 处按分隔符切分。"""
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
    """在括号深度为 0 处按 AND/OR 切分条件。"""
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
                last = m.end()
                i = last
                continue
        i += 1
    parts.append(text[last:].strip())
    return [p for p in parts if p]


def _normalize_expr(expr: str) -> str:
    """归一化列/表达式：去限定符、去 AS 别名、压缩运算符两侧空格。"""
    e = _SPACE_RE.sub(" ", expr.strip()).lower()
    e = _QUALIFIER_RE.sub("", e)
    e = re.sub(r"\s+as\s+[a-z_][a-z0-9_]*$", "", e)
    e = re.sub(r"\s*([*+/=<>,-])\s*", r"\1", e)
    e = re.sub(r"\s*\(\s*", "(", e)
    e = re.sub(r"\s*\)\s*", ")", e)
    return e.strip()


def _normalize_condition(cond: str) -> str:
    """归一化 WHERE 条件，保留日期中的短横线等字面量。"""
    c = _SPACE_RE.sub(" ", cond.strip()).lower()
    c = _QUALIFIER_RE.sub("", c)
    c = re.sub(r"\s*([=<>+*/])\s*", r"\1", c)
    c = re.sub(r"\s*\(\s*", "(", c)
    c = re.sub(r"\s*\)\s*", ")", c)
    return c.strip()


def _normalize_order(expr: str) -> str:
    e = _normalize_expr(expr)
    return re.sub(r"\s+(asc|desc)$", "", e)


def _first_table(text: str) -> str:
    text = text.strip()
    m = re.match(r"^(?:left|right|inner|outer|cross|full)\s+", text, re.IGNORECASE)
    if m:
        text = text[m.end():].strip()
    tokens = text.split()
    if not tokens:
        return ""
    return tokens[0].strip("`").lower()


def _parse_tables(sql: str) -> List[str]:
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
    """把一条 SQL 拆成可比较的组件。"""
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
    return set(re.findall(r"[a-z0-9_]+", text))


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _condition_recall(gold_conds: List[str], gen_conds: List[str]) -> float:
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
    """对比黄金 SQL 与生成 SQL，返回结构指标。"""
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


def rows_equal(rows_a: list, rows_b: list) -> bool:
    """比较两个结果集是否等价（忽略行顺序）。"""
    a = sorted([repr(r) for r in (rows_a or [])])
    b = sorted([repr(r) for r in (rows_b or [])])
    return a == b