"""
Schema Linking Pipeline 编排模块

将表召回（table_retriever）、字段匹配（field_matcher）、
Join 推理（join_resolver）三个模块串联为完整 Pipeline。

输入：用户的自然语言问题
输出：动态组装的精简 Schema（表+字段+Join 条件），可直接注入 Prompt

替代原 prompt_builder.py 中的硬编码全量 Schema 注入方式。
"""

import copy
import json
import logging
from functools import lru_cache

from .table_retriever import retrieve_tables, build_index
from .field_matcher import match_fields, build_field_index
from .join_resolver import (
    select_anchor,
    resolve_joins,
    _is_connected,
    _classify_intent,
    TABLE_RELATIONSHIPS,
    STRONG_METRIC_WORDS,
    TABLE_TYPES,
    TABLE_KEYWORDS,
)

logger = logging.getLogger("chatbi.schema_linking")


# ==================== Schema Linking Pipeline ====================
def _score_table_by_keywords(query: str, table_name: str) -> int:
    """基于17课 TABLE_KEYWORDS，计算查询与某张表的关键词匹配分数。"""
    keywords = TABLE_KEYWORDS.get(table_name, [])
    return sum(1 for kw in keywords if kw in query)


def _ensure_fact_table_for_metric(
    query: str,
    tables: list[dict],
    pool: list[dict] | None = None,
) -> list[dict]:
    """
    兜底策略：对于指标型问题，确保召回结果中包含语义最相关的事实表。

    背景：table_retriever 基于 Embedding 相似度召回，可能把"毛利率"这种问题
    错误地关联到 finance_expenses（费用表），而 sales_orders（订单事实表）
    才是计算毛利的正确来源。因此需要结合当前模块的关键词匹配规则做二次校验。

    逻辑：
    1. 无强指标词时跳过；
    2. 用 TABLE_KEYWORDS 给所有事实表打分；
    3. 若当前召回的事实表不是关键词得分最高的事实表，则把最高分事实表补充进来；
    4. 若当前未召回任何事实表，则放宽阈值检索并补充最相关事实表。

    Args:
        pool: 已放宽阈值召回的全量候选表。传入后不再重复触发一次
              Embedding 检索（早期实现对同一 query 会向量化两次）。
    """
    if not any(w in query for w in STRONG_METRIC_WORDS):
        return tables

    existing_names = {t["table_name"] for t in tables}

    # 当前已召回的事实表
    recalled_fact_tables = [
        t for t in tables if TABLE_TYPES.get(t["table_name"]) == "fact"
    ]

    # 复用外部传入的宽召回结果，避免重复 Embedding 调用
    all_tables = pool if pool is not None else retrieve_tables(
        query, top_k=10, score_threshold=0.0
    )
    all_fact_tables = [
        t for t in all_tables if TABLE_TYPES.get(t["table_name"]) == "fact"
    ]

    if not all_fact_tables:
        return tables

    # 按关键词匹配度选出最相关的事实表
    best_fact = max(
        all_fact_tables,
        key=lambda t: _score_table_by_keywords(query, t["table_name"]),
    )
    best_fact_name = best_fact["table_name"]

    # 如果最高分事实表不在召回结果中，则补充
    if best_fact_name not in existing_names:
        tables = tables + [best_fact]

    # 兜底：确保至少召回了一张事实表（兼容旧逻辑）
    if not recalled_fact_tables and not any(
        TABLE_TYPES.get(t["table_name"]) == "fact" for t in tables
    ):
        tables = tables + [all_fact_tables[0]]

    return tables


# ==================== 置信度评估（双通道核心） ====================
CONFIDENCE_THRESHOLD = 0.7

def assess_confidence(
    query: str,
    tables: list[dict],
    anchor: str,
    fields: list[dict],
    join_path: dict,
) -> float:
    """
    评估快速通道结果的置信度（严格遵循 docs/schema_linking_optimization.md 8.3 节设计）

    Returns:
        0.0 ~ 1.0，≥ CONFIDENCE_THRESHOLD 视为高置信度，可直接返回
    """
    score = 1.0

    # 1. 表召回置信度（权重最高）
    if not tables:
        score -= 0.5
    elif tables and tables[0].get("score", 0) < 0.5:
        score -= 0.3

    # 2. 意图分类置信度
    intent = _classify_intent(query)
    if intent == "ambiguous":
        score -= 0.3

    # 3. 锚表连通性（关键：连通性交叉校验）
    candidate_names = [t["table_name"] for t in tables]
    if anchor and candidate_names:
        if not _is_connected(anchor, candidate_names, TABLE_RELATIONSHIPS):
            score -= 0.3

    # 4. 字段匹配置信度
    if len(fields) < 3:
        score -= 0.2

    # 5. Join 可达性（关键：桥接表缺失检测）
    if join_path.get("unreachable"):
        unreachable_count = len(join_path["unreachable"])
        score -= 0.15 * unreachable_count

    # 6. 查询复杂度信号
    table_count = len(tables)
    if table_count > 3:
        score -= 0.1

    return max(score, 0.0)


# ==================== LLM 慢速通道 ====================
@lru_cache(maxsize=1)
def _get_llm_client():
    """慢速通道复用的 LLM 客户端（进程级单例，避免每次调用重新建连接池）。"""
    from text2SQL.llm_client import LLMClient
    from config import LLM_CONFIG

    return LLMClient(config=LLM_CONFIG)


def schema_link_llm(query: str) -> dict:
    """
    LLM 慢速通道：用一次调用完成表选择 + 字段匹配 + Join 推理
    （遵循 docs/schema_linking_optimization.md 8.4 节设计）
    """
    llm = _get_llm_client()

    prompt = f"""你是一个数据库 Schema 分析专家。已知以下数据库表结构：

表名: sales_orders（事实表-销售订单）
字段: order_id, order_no, customer_id, product_id, order_date, order_status,
      quantity, unit_price, discount_amount, gross_amount, net_amount, currency, region

表名: dim_customers（维度表-客户）
字段: customer_id, customer_name, customer_type, industry, country, region

表名: dim_products（维度表-产品）
字段: product_id, product_name, product_line, category, tech_route,
      standard_cost, material_cost, labor_cost

表名: exchange_rates（参考表-汇率）
字段: rate_date, currency, rate_to_cny

表名: finance_expenses（事实表-费用）
字段: expense_id, expense_date, department, rd_expense, selling_expense,
      admin_expense, finance_expense

表间关系：
- sales_orders.customer_id ↔ dim_customers.customer_id (JOIN)
- sales_orders.product_id ↔ dim_products.product_id (JOIN)
- sales_orders.(order_date, currency) ↔ exchange_rates.(rate_date, currency) (LEFT JOIN)
- finance_expenses 与其他表无直接关联，需独立查询

用户问题：{query}

请返回 JSON（不要输出其他内容）：
{{
    "anchor": "主表名",
    "tables": ["需要使用的表"],
    "fields": ["需要查询的字段（含表名前缀，如 o.net_amount）"],
    "joins": [
        {{"from": "表A", "to": "表B", "type": "JOIN/LEFT JOIN", "on": "条件"}}
    ],
    "reasoning": "选择理由（一句话）"
}}"""

    try:
        system_msg = "你是一个专业的数据库 Schema 分析专家，只输出 JSON 格式结果。"
        response_text = llm.generate_sql(system_msg, prompt)

        json_str = response_text.strip()
        if json_str.startswith('```'):
            json_str = json_str.split('\n', 1)[1].rsplit('```', 1)[0].strip()

        result = json.loads(json_str)

        tables_from_llm = [
            {"table_name": t, "score": 1.0, "description": f"LLM 选择: {t}", 
             "domain": TABLE_TYPES.get(t, "unknown"), "key_fields": ""}
            for t in result.get("tables", [])
        ]

        fields_from_llm = []
        for f in result.get("fields", []):
            parts = f.split('.')
            table = parts[0] if len(parts) > 1 else ""
            field = parts[-1] if parts else f
            fields_from_llm.append({
                "field_key": f,
                "table": table,
                "field": field,
                "score": 1.0,
                "description": "LLM 选择",
            })

        join_path_from_llm = {
            "anchor": result.get("anchor", ""),
            "joins": [
                {
                    "from_table": j.get("from", ""),
                    "to_table": j.get("to", ""),
                    "join_type": j.get("type", "JOIN"),
                    "on_clause": j.get("on", ""),
                }
                for j in result.get("joins", [])
            ],
            "unreachable": [],
            "sql_fragment": "",
        }

        dynamic_schema = _assemble_dynamic_schema(
            query, tables_from_llm, fields_from_llm, join_path_from_llm,
            result.get("reasoning", "LLM 慢速通道")
        )

        return {
            "tables": tables_from_llm,
            "fields": fields_from_llm,
            "anchor": result.get("anchor", ""),
            "join_path": join_path_from_llm,
            "dynamic_schema": dynamic_schema,
            "metadata": {
                "table_count": len(tables_from_llm),
                "field_count": len(fields_from_llm),
                "join_count": len(join_path_from_llm.get("joins", [])),
                "has_unreachable": False,
                "channel": "llm",
                "confidence": 1.0,
                "reasoning": result.get("reasoning", ""),
            },
        }

    except Exception as e:
        print(f"[LLM 慢速通道异常] {e}")
        raise


def schema_link(
    query: str,
    table_top_k: int = 3,
    field_top_k: int = 12,
    table_threshold: float = 0.3,
    field_threshold: float = 0.25,
    include_join: bool = True,
) -> dict:
    """
    Schema Linking 完整 Pipeline：
    用户问题 → 表召回 → 锚表选择 → 字段匹配 → Join 推理 → 动态 Schema 组装

    Args:
        query: 用户的自然语言问题
        table_top_k: 表召回数量
        field_top_k: 字段匹配数量
        table_threshold: 表召回相似度阈值
        field_threshold: 字段匹配最终得分阈值
        include_join: 是否生成 Join 路径

    Returns:
        {
            "tables": [{"table_name": str, "score": float, ...}],
            "fields": [{"field_key": str, "table": str, "field": str, "score": float, ...}],
            "anchor": str,                        # 选定的锚表
            "join_path": {"anchor": str, "joins": [...], "sql_fragment": str, "unreachable": [...]},
            "dynamic_schema": str,                # 可直接注入 Prompt 的精简 Schema 文本
            "metadata": {
                "table_count": int,
                "field_count": int,
                "join_count": int,
                "has_unreachable": bool,
            }
        }
    """
    # 结果只依赖问题文本与检索参数（行级权限在 SQL 生成后由 security 层施加），
    # 因此可以做进程级缓存；重复提问可直接省掉全部 Embedding 往返。
    cached = _schema_link_cached(
        query, table_top_k, field_top_k, table_threshold, field_threshold, include_join
    )
    return copy.deepcopy(cached)


@lru_cache(maxsize=256)
def _schema_link_cached(
    query: str,
    table_top_k: int,
    field_top_k: int,
    table_threshold: float,
    field_threshold: float,
    include_join: bool,
) -> dict:
    """带缓存的 Pipeline 实现；调用方拿到的应是 deepcopy。"""
    # Step 1: 一次宽召回，同时用于阈值筛选与指标兜底（避免同一 query 向量化两次）
    wide_tables = retrieve_tables(
        query, top_k=max(table_top_k, 10), score_threshold=0.0
    )
    tables = [t for t in wide_tables if t["score"] >= table_threshold][:table_top_k]

    # 兜底：指标型问题必须召回事实表，否则 SQL 无法生成
    tables = _ensure_fact_table_for_metric(query, tables, pool=wide_tables)

    candidate_table_names = [t["table_name"] for t in tables]

    # 无召回结果时直接返回空，触发 prompt_builder 的 fallback
    if not candidate_table_names:
        return {
            "tables": [],
            "fields": [],
            "anchor": "",
            "join_path": {"anchor": "", "joins": [], "sql_fragment": "", "unreachable": []},
            "dynamic_schema": "",
            "metadata": {
                "table_count": 0,
                "field_count": 0,
                "join_count": 0,
                "has_unreachable": False,
                "channel": "empty",
                "confidence": 0.0,
            },
        }

    # Step 2: 锚表选择（基于查询意图）
    anchor, anchor_reason = select_anchor(query, candidate_table_names)

    # Step 3: 字段匹配（限定在召回的表范围内）
    fields = match_fields(
        query,
        candidate_tables=candidate_table_names,
        top_k=field_top_k,
        score_threshold=field_threshold,
    )

    # Step 4: Join 路径推理
    join_path = {"anchor": anchor, "joins": [], "sql_fragment": "", "unreachable": []}
    if include_join:
        join_path = resolve_joins(anchor, candidate_table_names)

    # Step 5: 动态 Schema 组装
    dynamic_schema = _assemble_dynamic_schema(query, tables, fields, join_path, anchor_reason)

    # Step 6: 置信度评估（双通道核心决策点）
    confidence = assess_confidence(query, tables, anchor, fields, join_path)

    fast_result = {
        "tables": tables,
        "fields": fields,
        "anchor": anchor,
        "join_path": join_path,
        "dynamic_schema": dynamic_schema,
        "metadata": {
            "table_count": len(tables),
            "field_count": len(fields),
            "join_count": len(join_path.get("joins", [])),
            "has_unreachable": len(join_path.get("unreachable", [])) > 0,
            "channel": "fast",
            "confidence": round(confidence, 4),
        },
    }

    if confidence >= CONFIDENCE_THRESHOLD:
        logger.debug(
            "Schema Linking 快速通道通过（置信度 %.2f ≥ %.2f）",
            confidence,
            CONFIDENCE_THRESHOLD,
        )
        return fast_result

    logger.info(
        "Schema Linking 快速通道置信度不足（%.2f < %.2f），升级到 LLM 慢速通道",
        confidence,
        CONFIDENCE_THRESHOLD,
    )

    # 升级到 LLM 慢速通道
    try:
        llm_result = schema_link_llm(query)
        logger.info(
            "Schema Linking LLM 慢速通道完成：%s",
            llm_result["metadata"].get("reasoning", ""),
        )
        return llm_result
    except Exception as exc:
        logger.warning("Schema Linking LLM 慢速通道失败（%s），降级返回快速通道结果", exc)
        fast_result["metadata"]["channel"] = "fast_fallback"
        return fast_result


def _assemble_dynamic_schema(
    query: str,
    tables: list[dict],
    fields: list[dict],
    join_path: dict,
    anchor_reason: str,
) -> str:
    """
    将召回结果组装为可注入 Prompt 的动态 Schema 文本。

    格式设计原则：
    - 清晰列出涉及的表和字段
    - 包含字段的关键业务说明
    - 明确标注锚表和 Join 条件
    - 保持紧凑，尽量减少 Token 消耗
    """
    lines = []

    # 1. 按表分组字段
    table_fields_map: dict[str, list[dict]] = {}
    for t in tables:
        table_fields_map[t["table_name"]] = []
    for f in fields:
        if f["table"] in table_fields_map:
            table_fields_map[f["table"]].append(f)

    # 2. 生成各表的精简 Schema
    for table_name, field_list in table_fields_map.items():
        table_info = next((t for t in tables if t["table_name"] == table_name), None)
        domain = table_info.get("domain", "") if table_info else ""
        anchor_tag = " [锚表]" if table_name == join_path.get("anchor") else ""

        lines.append(f"表：{table_name}（{domain}）{anchor_tag}")
        if field_list:
            for f in field_list:
                short_desc = f.get("description", "").split("。")[0]
                rule_tag = ""
                if f.get("rule_applied"):
                    rule_tag = f" ★{f['rule_applied']}"
                lines.append(f"  - {f['field']} {short_desc}{rule_tag}")
        else:
            lines.append("  （未匹配到特定字段，参考全部字段）")
        lines.append("")

    # 3. 生成 Join 关系说明
    if join_path.get("joins"):
        lines.append("表间关联：")
        for j in join_path["joins"]:
            lines.append(f"  {j['join_type']} {j['to_table']} ON {j['on_clause']}")
        lines.append("")

    # 4. 标注不可连通的表
    if join_path.get("unreachable"):
        lines.append(
            f"注意：以下表与其他表无直接关联，需独立查询：{join_path['unreachable']}"
        )
        lines.append("")

    return "\n".join(lines).strip()


# ==================== Prompt 集成接口 ====================
def build_dynamic_prompt_schema(query: str) -> str:
    """
    供 prompt_builder.py 调用的简化接口。
    输入用户问题，输出可直接替换 SCHEMA 常量的动态文本。

    如果 Schema Linking 失败（如无召回结果），返回空字符串，
    调用方应回退到全量 Schema。
    """
    try:
        result = schema_link(query)
        if result["metadata"]["table_count"] == 0:
            return ""  # 无召回，触发 fallback
        return result["dynamic_schema"]
    except Exception as exc:
        logger.warning("Schema Linking 异常（%s），将回退到全量 Schema", exc)
        return ""


# ==================== 索引初始化 ====================
def ensure_indexes(force_rebuild: bool = False):
    """确保表索引和字段索引都已构建"""
    print("检查向量索引状态...")
    build_index(force_rebuild=force_rebuild)
    build_field_index(force_rebuild=force_rebuild)
    print("索引就绪。")


# ==================== 主程序：演示 ====================
if __name__ == "__main__":
    print("=" * 60)
    print("Schema Linking 完整链路演示")
    print("=" * 60)

    # 初始化索引（如果已存在则跳过，避免重复调用 Embedding API）
    print("\n--- 初始化向量索引 ---")
    ensure_indexes(force_rebuild=False)

    # 测试问题集
    test_questions = [
        "按客户类型统计各产品线的收入，需要换算成人民币",
        "各产品线的毛利率",
        "上个月的研发费用和销售费用对比",
        "查询德国客户的订单总额",
        "哪些客户没有下过订单",
    ]

    for question in test_questions:
        print(f"\n{'='*60}")
        print(f"问题：{question}")
        print("-" * 60)

        result = schema_link(question)

        # 打印元数据
        meta = result["metadata"]
        print(f"召回：{meta['table_count']} 张表, "
              f"{meta['field_count']} 个字段, "
              f"{meta['join_count']} 个 Join")
        print(f"锚表：{result['anchor']}")

        # 打印动态 Schema
        print(f"\n动态 Schema（注入 Prompt）：")
        print("-" * 40)
        print(result["dynamic_schema"])
        print("-" * 40)

        # 打印 Join SQL
        if result["join_path"].get("sql_fragment"):
            print(f"\nJoin SQL 片段：")
            print(f"  {result['join_path']['sql_fragment'].replace(chr(10), chr(10) + '  ')}")

    # 对比 Token 估算
    print(f"\n\n{'='*60}")
    print("Token 消耗对比（粗略估算）")
    print("=" * 60)
    full_schema_tokens = 1500  # 全量 Schema 约 1500 tokens
    for question in test_questions[:3]:
        result = schema_link(question)
        dynamic_tokens = len(result["dynamic_schema"]) // 2  # 粗略估算：2字符≈1token
        savings = (1 - dynamic_tokens / full_schema_tokens) * 100
        print(f"  {question[:20]}... → ~{dynamic_tokens} tokens (节省 {savings:.0f}%)")