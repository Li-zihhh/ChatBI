"""SQL 错误分类与修复建议。

两层架构：
  第一层 启发式 —— 高频、信号明确的错误零成本归类；
  第二层 LLM   —— 启发式无法归类时做语义推理（可选，需注入 llm_client）。

相对早期版本的关键变化：
  1. **errno 优先**：数据库已经给出明确错误码时直接按 errno 归类，不再靠正则猜 SQL。
     早期实现里 `self.error_patterns` 定义了 5 组正则却从未被读取（死代码），
     实际生效的 `_check_*` 又过于宽松（问题里出现"订单"就判 filter_error），
     导致几乎所有失败都被归为同一类。
  2. **repairable 判定**：明确区分"重试有意义"（字段/语法/聚合）与
     "重试只会放大故障"（连接/超时/权限），供上层错误自修复闭环使用。
  3. **repair_hint**：产出可直接拼进 Prompt 的修复指引，而不只是给人看的建议。
"""

from __future__ import annotations

import json
import re
from typing import Optional


class ErrorAnalyzer:
    """SQL 生成错误分析器。"""

    FIELD_ERROR = "field_error"              # 字段选择错误
    JOIN_ERROR = "join_error"                # 关联路径错误
    TIME_ERROR = "time_error"                # 时间计算错误
    FILTER_ERROR = "filter_error"            # 过滤条件遗漏
    AGGREGATION_ERROR = "aggregation_error"  # 聚合逻辑错误
    SYNTAX_ERROR = "syntax_error"            # SQL 语法错误
    UNKNOWN = "unknown"                      # 未知错误

    ERROR_CATEGORIES = {
        FIELD_ERROR,
        JOIN_ERROR,
        TIME_ERROR,
        FILTER_ERROR,
        AGGREGATION_ERROR,
        SYNTAX_ERROR,
        UNKNOWN,
    }

    # 语义类错误（重试有意义）
    SEMANTIC_CATEGORIES = {
        FIELD_ERROR,
        JOIN_ERROR,
        TIME_ERROR,
        FILTER_ERROR,
        AGGREGATION_ERROR,
        SYNTAX_ERROR,
    }

    # MySQL errno → 业务错误类别
    ERRNO_TO_CATEGORY = {
        1064: SYNTAX_ERROR,        # 语法错误
        1146: JOIN_ERROR,          # 表不存在（多为关联表名写错或漏 JOIN）
        1054: FIELD_ERROR,         # 未知列
        1052: FIELD_ERROR,         # 列名歧义
        1109: FIELD_ERROR,         # 字段列表中的未知表
        1055: AGGREGATION_ERROR,   # ONLY_FULL_GROUP_BY
        1111: AGGREGATION_ERROR,   # 组函数用法非法
        1241: FIELD_ERROR,         # 操作数个数不匹配
        1247: JOIN_ERROR,          # 不支持的引用
        1582: FIELD_ERROR,         # 参数个数错误
        3065: AGGREGATION_ERROR,   # ORDER BY 列不在 SELECT 列表
    }

    LLM_ANALYZE_SYSTEM = "你是一个 SQL 错误分析专家，只输出合法 JSON，不要任何额外解释。"

    LLM_ANALYZE_PROMPT = """请根据以下信息，分析 SQL 生成错误的类型、根因和修复建议。

【错误类型标签】
- field_error: 字段选择错误（字段拼写错误、语义理解偏差、使用了错误的金额/成本口径）
- join_error: 关联路径错误（遗漏维度表、缺少必要 JOIN、外键条件错误）
- time_error: 时间计算错误（时间范围、时间函数、动态时间边界处理有误）
- filter_error: 过滤条件遗漏（缺少 order_status、统计范围过宽）
- aggregation_error: 聚合逻辑错误（GROUP BY 缺失、聚合函数使用不当、HAVING 与 WHERE 混淆）
- syntax_error: SQL 语法错误（关键字拼写、括号不匹配、引号错误、输出不完整）
- unknown: 无法明确归类的错误

【输出要求】严格按以下 JSON 格式输出：
{{
  "error_type": "上述标签之一",
  "root_cause": "一句话描述根本原因",
  "suggestion": "针对 prompt/schema/rules 的改进方向"
}}

【用户问题】
{question}

【生成的 SQL】
{sql}

【数据库错误信息（如有）】
{error_msg}

请直接输出 JSON："""

    # 修复指引：直接拼进重试 Prompt，告诉模型"这一次要改什么"
    REPAIR_HINTS = {
        FIELD_ERROR: (
            "上一条 SQL 使用了不存在或口径错误的字段。"
            "请严格只使用 Schema 中出现的字段名；"
            "收入/销售额默认用 net_amount，只有问题明确说“含税”时才用 gross_amount；"
            "成本用 material_cost + labor_cost，不要用 standard_cost。"
        ),
        JOIN_ERROR: (
            "上一条 SQL 的表关联不完整或表名错误。"
            "涉及收入且存在币种时，必须 JOIN exchange_rates "
            "ON sales_orders.order_date = exchange_rates.rate_date "
            "AND sales_orders.currency = exchange_rates.currency。"
        ),
        TIME_ERROR: (
            "上一条 SQL 的时间边界有误。"
            "“最近N个月”用 order_date >= DATE_SUB(CURDATE(), INTERVAL N MONTH)；"
            "“前N个月”用完整自然月边界；时间区间统一用 >= 与 < 的闭开区间写法。"
        ),
        FILTER_ERROR: (
            "上一条 SQL 缺少必要的业务过滤条件。"
            "统计收入、订单量、客单价、毛利时，必须加 order_status = 'completed'。"
        ),
        AGGREGATION_ERROR: (
            "上一条 SQL 的聚合与分组不一致。"
            "SELECT 中的非聚合字段必须全部出现在 GROUP BY 中；"
            "SELECT 全是聚合函数时不要加 GROUP BY。"
        ),
        SYNTAX_ERROR: (
            "上一条 SQL 存在语法问题，无法被 MySQL 解析。"
            "请输出一条完整闭合的 SELECT 语句：关键字拼写正确、括号与引号配对，"
            "不要输出解释文字或截断的半句。"
        ),
        UNKNOWN: (
            "上一条 SQL 执行失败。请重新检查字段名、表名、JOIN 条件与聚合写法后重新生成。"
        ),
    }

    def __init__(self, llm_client=None):
        self.llm_client = llm_client

    # ==================== 主入口 ====================
    def categorize_error(
        self,
        sql: str,
        error_msg: Optional[str] = None,
        question: Optional[str] = None,
        error_type: Optional[str] = None,
        errno: Optional[int] = None,
    ) -> dict:
        """对 SQL 错误进行分类。

        Args:
            sql: 生成（或执行）失败的 SQL。
            error_msg: 数据库返回的错误信息（如有）。
            question: 用户原始问题。
            error_type: 上层已分类的错误类型（如 database_sql_syntax），优先使用。
            errno: MySQL 原始错误码，最高优先级。

        Returns:
            {error_type, root_cause, suggestion, repair_hint, layer, repairable, sql}
        """
        category, layer, reason = self._from_structured_signal(error_type, errno)

        # 1) 结构化信号最可信，直接采纳
        if category is None and error_msg and self._is_syntax_error(error_msg):
            category, layer, reason = self.SYNTAX_ERROR, "database", "数据库报语法错误"

        # 2) 语义启发式（仅在无数据库结构化信号时使用）
        if category is None:
            category = self._detect_semantic_error(sql, question)
            if category is not None:
                layer, reason = "heuristic", "根据 SQL 与问题的语义匹配推断"

        # 3) LLM 兜底
        if category is None:
            llm_result = self._llm_analyze(sql, error_msg, question)
            if llm_result:
                llm_result["sql"] = sql
                return llm_result
            category, layer, reason = (
                self.UNKNOWN,
                "fallback",
                "无法从可用信号中识别错误模式，需人工审查",
            )

        return {
            "error_type": category,
            "root_cause": self._get_root_cause(category, error_msg, reason),
            "suggestion": self._get_suggestion(category),
            "repair_hint": self.REPAIR_HINTS.get(category, self.REPAIR_HINTS[self.UNKNOWN]),
            "layer": layer,
            "repairable": category in self.SEMANTIC_CATEGORIES,
            "sql": sql,
        }

    # ==================== 结构化信号 ====================
    def _from_structured_signal(
        self,
        error_type: Optional[str],
        errno: Optional[int],
    ) -> tuple[Optional[str], str, str]:
        if errno and errno in self.ERRNO_TO_CATEGORY:
            return self.ERRNO_TO_CATEGORY[errno], "database", f"MySQL errno {errno}"

        if error_type:
            if error_type == "database_sql_syntax":
                return self.SYNTAX_ERROR, "database", "上层错误类型 database_sql_syntax"
            if error_type.startswith("database_"):
                # 连接/超时/权限：不可修复，归类为 unknown 但保留 layer=database
                return self.UNKNOWN, "database", f"上层错误类型 {error_type}"

        return None, "", ""

    @staticmethod
    def _is_syntax_error(error_msg: str) -> bool:
        markers = (
            "syntax error",
            "parse error",
            "unexpected",
            "near ",
            "expected",
            "invalid",
            "you have an error in your sql",
        )
        lowered = error_msg.lower()
        return any(marker in lowered for marker in markers)

    # ==================== 语义启发式 ====================
    def _detect_semantic_error(self, sql: str, question: Optional[str]) -> Optional[str]:
        if not sql:
            return None
        sql_lower = sql.lower()
        q = question or ""

        # 顺序即优先级：口径错 > 聚合错 > 少 JOIN > 时间错 > 少过滤
        checks = (
            (self.FIELD_ERROR, self._check_field_error),
            (self.AGGREGATION_ERROR, self._check_aggregation_error),
            (self.JOIN_ERROR, self._check_join_error),
            (self.TIME_ERROR, self._check_time_error),
            (self.FILTER_ERROR, self._check_filter_error),
        )
        for category, check in checks:
            if check(sql_lower, q):
                return category
        return None

    @staticmethod
    def _check_field_error(sql: str, question: str) -> bool:
        """业务口径混淆：net vs gross、material_cost vs standard_cost。"""
        if ("收入" in question or "销售额" in question) and "含税" not in question:
            if "gross_amount" in sql and "net_amount" not in sql:
                return True
        if "成本" in question and "standard_cost" in sql and "material_cost" not in sql:
            return True
        # 引用了未建模字段
        if re.search(r"\bchannel\b", sql):
            return True
        return False

    @staticmethod
    def _check_join_error(sql: str, question: str) -> bool:
        """少关联：多币种收入缺 exchange_rates；客户/产品维度缺维度表。"""
        if ("收入" in question or "销售额" in question) and "currency" in sql:
            if "exchange_rates" not in sql:
                return True
        if ("客户" in question or "产品线" in question or "产品" in question):
            if "dim_customers" not in sql and "dim_products" not in sql:
                return True
        return False

    @staticmethod
    def _check_time_error(sql: str, question: str) -> bool:
        """动态时间边界：问了"最近/本月/上个/今年"却没有时间函数或日期条件。"""
        if not any(
            word in question for word in ("最近", "本月", "上个", "上季度", "今年", "近")
        ):
            return False
        signals = (
            "date_sub",
            "curdate",
            "current_date",
            "now()",
            "interval",
            "between",
            "date_format",
            "order_date >=",
            "expense_date >=",
        )
        return not any(signal in sql for signal in signals)

    @staticmethod
    def _check_filter_error(sql: str, question: str) -> bool:
        """漏过滤：统计类问题缺 order_status（"已取消/待支付"类问题除外）。"""
        if any(word in question for word in ("取消", "cancelled", "待支付", "pending")):
            return False
        metric_words = ("收入", "销售额", "订单量", "客单价", "毛利", "利润", "订单数")
        if not any(word in question for word in metric_words):
            return False
        # 只查维度表时不应要求 order_status
        if "sales_orders" not in sql:
            return False
        return "order_status" not in sql

    @staticmethod
    def _check_aggregation_error(sql: str, question: str) -> bool:
        """有聚合函数 + 多个顶层 SELECT 项 + 无 GROUP BY。"""
        has_agg = bool(re.search(r"\b(sum|avg|count|max|min)\s*\(", sql))
        if not has_agg or "group by" in sql:
            return False

        select_part = sql.split(" from ")[0]
        depth = 0
        top_level_commas = 0
        for char in select_part:
            if char == "(":
                depth += 1
            elif char == ")":
                depth = max(0, depth - 1)
            elif char == "," and depth == 0:
                top_level_commas += 1
        return top_level_commas > 0

    # ==================== 文案 ====================
    @staticmethod
    def _get_root_cause(
        error_type: str, error_msg: Optional[str], reason: str
    ) -> str:
        causes = {
            ErrorAnalyzer.FIELD_ERROR: "模型未能正确理解业务语义，选择了错误或不存在的字段",
            ErrorAnalyzer.JOIN_ERROR: "模型遗漏了必要的表关联或使用了错误的关联条件",
            ErrorAnalyzer.TIME_ERROR: "时间范围计算不符合业务预期，动态时间边界处理有误",
            ErrorAnalyzer.FILTER_ERROR: "遗漏了关键的业务过滤条件，导致统计范围不正确",
            ErrorAnalyzer.AGGREGATION_ERROR: "聚合维度与 SELECT 字段不匹配，或聚合逻辑有误",
            ErrorAnalyzer.SYNTAX_ERROR: "生成的 SQL 存在语法问题，无法被数据库解析",
        }
        base = causes.get(error_type, "无法明确归因")
        if error_msg:
            return f"{base}（{reason}；数据库报错：{error_msg[:160]}）"
        return f"{base}（{reason}）"

    @staticmethod
    def _get_suggestion(error_type: str) -> str:
        suggestions = {
            ErrorAnalyzer.FIELD_ERROR: "在 RULES 中强化字段语义定义，增加字段口径的 Few-shot 正反例",
            ErrorAnalyzer.JOIN_ERROR: "在 ERROR_GUARDS 中补充 Join 路径检查清单，增加多表 Few-shot 示例",
            ErrorAnalyzer.TIME_ERROR: "增加时间计算 Few-shot，明确滚动窗口与自然月的 SQL 写法差异",
            ErrorAnalyzer.FILTER_ERROR: "在 RULES 中明确各类查询的强制过滤条件，增加漏过滤负例",
            ErrorAnalyzer.AGGREGATION_ERROR: "增加 GROUP BY 正确用法示例，强调非聚合字段必须入 GROUP BY",
            ErrorAnalyzer.SYNTAX_ERROR: "检查 Prompt 中是否存在导致输出不完整 SQL 的干扰内容",
        }
        return suggestions.get(error_type, "人工审查该用例并补充到回归测试集")

    # ==================== 批量分析 ====================
    def analyze_batch(self, cases: list[dict]) -> list[dict]:
        """批量分析多个错误案例。"""
        results = []
        for case in cases:
            result = self.categorize_error(
                sql=case.get("sql", ""),
                error_msg=case.get("error_msg"),
                question=case.get("question"),
                error_type=case.get("error_type"),
                errno=case.get("errno"),
            )
            result["question"] = case.get("question", "")
            results.append(result)
        return results

    def generate_report(self, results: list[dict]) -> str:
        """生成错误分析汇总报告。"""
        from collections import Counter

        type_counts = Counter(r["error_type"] for r in results)
        lines = [
            "=" * 50,
            "SQL 生成错误分析汇总报告",
            "=" * 50,
            f"总案例数：{len(results)}",
            "",
            "错误类型分布：",
        ]
        for err_type, count in type_counts.most_common():
            lines.append(f"  {err_type}: {count} 例")

        lines.extend(["", "详细案例："])
        for index, item in enumerate(results, 1):
            lines.append(f"\n[{index}] 问题：{item.get('question', 'N/A')}")
            lines.append(f"    SQL：{(item.get('sql') or '')[:80]}...")
            lines.append(f"    类型：{item['error_type']}（layer={item.get('layer')}）")
            lines.append(f"    根因：{item['root_cause']}")
            lines.append(f"    建议：{item['suggestion']}")
        return "\n".join(lines)

    # ==================== 第二层：LLM 分析 ====================
    def _llm_analyze(
        self,
        sql: str,
        error_msg: Optional[str],
        question: Optional[str],
    ) -> Optional[dict]:
        """调用 LLM 做深度语义推理。仅在启发式无法归类且注入了 llm_client 时触发。"""
        if self.llm_client is None:
            return None

        prompt = self.LLM_ANALYZE_PROMPT.format(
            question=question or "(未提供用户问题)",
            sql=sql or "(未提供 SQL)",
            error_msg=error_msg or "(无数据库错误信息)",
        )

        try:
            raw_output = self.llm_client.generate_text(self.LLM_ANALYZE_SYSTEM, prompt)
            parsed = self._parse_llm_json(raw_output)
            if parsed is None:
                return None

            error_type = parsed.get("error_type", self.UNKNOWN)
            if error_type not in self.ERROR_CATEGORIES:
                error_type = self.UNKNOWN

            return {
                "error_type": error_type,
                "root_cause": (parsed.get("root_cause") or "").strip(),
                "suggestion": (parsed.get("suggestion") or "").strip(),
                "repair_hint": self.REPAIR_HINTS.get(
                    error_type, self.REPAIR_HINTS[self.UNKNOWN]
                ),
                "layer": "llm",
                "repairable": error_type in self.SEMANTIC_CATEGORIES,
            }
        except Exception:
            return None

    @staticmethod
    def _parse_llm_json(raw: str) -> Optional[dict]:
        """从 LLM 输出中解析 JSON，兼容 ```json 围栏与前后缀说明文本。"""
        if not raw:
            return None

        text = raw.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
            text = text.strip()

        if text.startswith("{") and text.endswith("}"):
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                pass

        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        return None
