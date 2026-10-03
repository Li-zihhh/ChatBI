
# ==================== 手写 Schema（完全靠人工整理）====================
SCHEMA = """
表：dim_customers（客户维度表）
- customer_id INT 主键
- customer_name VARCHAR(100) 客户名称
- customer_type VARCHAR(50) 客户类型：OEM整车厂 / 储能集成商 / 电网集团 / 工商业用户 / 换电运营商 / 经销商
- industry VARCHAR(50) 客户行业：交通 / 能源 / 工业 / 特种交通
- country VARCHAR(50) 具体国家，如 Germany
- region VARCHAR(50) 大区，如 欧洲、北美

表：dim_products（产品维度表）
- product_id INT 主键
- product_name VARCHAR(100) 产品名称
- product_line VARCHAR(50) 产品线：动力电池-乘用车 / 动力电池-商用车 / 储能系统-电网级 / 储能系统-工商业 / 电池材料与回收
- category VARCHAR(50) 产品分类：高能量密度型 / 超快充型 / 混动专用型 / 低温适配型 / 商用车标准型 / 电网级储能型 / 工商业储能型
- tech_route VARCHAR(50) 技术路线：三元锂 / 磷酸铁锂 / 钠离子 / 固态电池
- standard_cost DECIMAL(10,2) 标准成本
- material_cost DECIMAL(10,2) 材料成本
- labor_cost DECIMAL(10,2) 人工成本

表：sales_orders（销售订单表）
- order_id BIGINT 主键
- order_no VARCHAR(50) 订单编号
- customer_id INT 外键 → dim_customers.customer_id
- product_id INT 外键 → dim_products.product_id
- region VARCHAR(50) 销售区域
- order_date DATE 订单日期
- order_status VARCHAR(20) 订单状态：completed / cancelled / pending
- quantity DECIMAL(10,2) 数量（MWh 或套数）
- unit_price DECIMAL(10,2) 单价（每 MWh 或每套价格，不含税）
- discount_amount DECIMAL(10,2) 折扣金额
- gross_amount DECIMAL(12,2) 含税总额（仅在明确要求"含税"时使用）
- net_amount DECIMAL(12,2) 不含税收入（默认销售额字段）
- currency VARCHAR(10) 币种

表：exchange_rates（汇率表）
- rate_date DATE 日期
- currency VARCHAR(10) 币种
- rate_to_cny DECIMAL(10,4) 兑人民币汇率

表：finance_expenses（费用表）
- expense_id BIGINT 主键
- expense_date DATE 费用日期
- department VARCHAR(50) 部门
- rd_expense DECIMAL(12,2) 研发费用（新能源企业研发投入大）
- selling_expense DECIMAL(12,2) 销售费用
- admin_expense DECIMAL(12,2) 管理费用
- finance_expense DECIMAL(12,2) 财务费用
- marketing_expense DECIMAL(12,2) 市场费用（属于销售费用子项）
- logistics_expense DECIMAL(12,2) 物流费用
- warranty_expense DECIMAL(12,2) 质保费用
"""

# ==================== Few-shot 示例 ====================
FEW_SHOT_EXAMPLES = """
示例1：
问题：查询已完成订单的总数量
SQL：SELECT COUNT(*) FROM sales_orders WHERE order_status = 'completed';

示例2：
问题：按客户类型统计订单数量
SQL：SELECT c.customer_type, COUNT(*) AS order_count FROM sales_orders o JOIN dim_customers c ON o.customer_id = c.customer_id WHERE o.order_status = 'completed' GROUP BY c.customer_type;

示例3：
问题：查询2026年第一季度的总费用
SQL：SELECT SUM(rd_expense + selling_expense + admin_expense + finance_expense) AS total_expense FROM finance_expenses WHERE expense_date >= '2026-01-01' AND expense_date < '2026-04-01';

示例4：
问题：查询含税销售额最高的产品名称
SQL：SELECT p.product_name, SUM(o.gross_amount) AS gross_revenue FROM sales_orders o JOIN dim_products p ON o.product_id = p.product_id WHERE o.order_status = 'completed' GROUP BY p.product_name ORDER BY gross_revenue DESC LIMIT 1;

示例5：
问题：统计各销售区域的订单金额（人民币）
SQL：SELECT o.region, SUM(o.net_amount * e.rate_to_cny) AS revenue FROM sales_orders o JOIN exchange_rates e ON o.order_date = e.rate_date AND o.currency = e.currency WHERE o.order_status = 'completed' GROUP BY o.region;

示例6：
问题：查询销售费用最高的三个部门
SQL：SELECT department, SUM(selling_expense) AS total_selling FROM finance_expenses GROUP BY department ORDER BY total_selling DESC LIMIT 3;

示例7：
问题：北美市场最近六个月的销售额（人民币）
SQL：SELECT SUM(o.net_amount * e.rate_to_cny) AS revenue FROM sales_orders o JOIN exchange_rates e ON o.order_date = e.rate_date AND o.currency = e.currency WHERE o.region = '北美' AND o.order_status = 'completed' AND o.order_date >= DATE_SUB(CURDATE(), INTERVAL 6 MONTH);
"""

# ==================== COT 引导与结构化约束 ====================
COT_INSTRUCTION = """
【思考步骤】
在生成 SQL 之前，请按以下步骤思考：
1. 识别问题中涉及的核心表和字段
2. 判断是否需要 JOIN 以及 JOIN 的条件
3. 确认金额口径（net_amount vs gross_amount）和成本口径
4. 确认是否需要过滤 order_status = 'completed'
5. 确认时间范围和汇率转换需求
6. 最后生成 SQL
"""

# ==================== 输出约束 ====================
OUTPUT_CONSTRAINTS = """
【输出约束】
1. 只输出 SQL 语句，不需要解释
2. 使用标准 MySQL 语法
3. 确保字段名和表名与 Schema 一致
4. 如果涉及多表查询，使用 JOIN 连接
5. SQL 必须完整闭合，CTE、SELECT、GROUP BY、ORDER BY 不得省略，不要输出截断的半句 SQL
"""

# ==================== 规则注入层 ====================
RULES = """
【关键业务规则】
1. 收入口径："销售额""收入"均指不含税收入，统一使用 sales_orders.net_amount，禁止使用 gross_amount
2. 成本口径："成本"指实际销售成本，计算公式为 dim_products.material_cost + dim_products.labor_cost
3. 订单统计范围：统计收入、订单量、客单价等指标时，必须过滤 order_status = 'completed'，排除 cancelled 和 pending
4. 汇率转换：涉及多币种收入汇总时，必须通过 order_date 和 currency 关联 exchange_rates 表，使用 rate_to_cny 折算为人民币
5. 时间范围语义：
   - "最近N个月""近N个月"：滚动窗口，从 N 个月前到今天，用 DATE_SUB(CURDATE(), INTERVAL N MONTH) 作为起始边界（10月8号提问 → 7月8号~10月8号）
   - "前N个月"：指截止到上个月的 N 个完整自然月，用 DATE_FORMAT(DATE_SUB(CURDATE(), INTERVAL N MONTH), '%Y-%m-01') 作为起始边界、DATE_FORMAT(CURDATE(), '%Y-%m-01') 作为上界（10月8号提问 → 7月1号~9月30号）
   - "本月"：当月1日至当前日期，用 DATE_FORMAT(CURDATE(), '%Y-%m-01') 作为起始边界
   - "上季度""Q1/Q2/Q3/Q4"：日历周期，使用固定的自然月边界
6. 费用层级：selling_expense 是销售费用总项，包含 marketing_expense、logistics_expense、warranty_expense，汇总时不得重复计算
7. 毛利与毛利率：
   - 毛利（绝对值）= net_amount - (material_cost + labor_cost) * quantity
   - 毛利率（比率）= (SUM(net_amount) - SUM((material_cost + labor_cost) * quantity)) / SUM(net_amount)
8. 不要引用 Schema 中不存在的字段，例如 channel
"""

# ==================== 错误防护层 ====================
ERROR_GUARDS = """
【常见错误防护】
- 字段选择：
  * "不含税收入""净收入""销售额"（未明确说含税）→ 用 net_amount
  * "含税销售额""含税总额""价税合计" → 用 gross_amount
  * 典型错误：问题问"不含税收入"却用了 gross_amount；问题问"含税销售额"却用了 net_amount
- Join 遗漏：只要查询涉及"收入"且存在 currency 字段，必须关联 exchange_rates 表做汇率转换
- 过滤遗漏：所有收入类统计必须包含 WHERE order_status = 'completed'
- 时间边界：使用 >= 和 < 组合表示闭开区间；“最近N个月”按 DATE_SUB(CURDATE(), INTERVAL N MONTH) 处理
- 聚合维度：GROUP BY 字段必须与 SELECT 中的非聚合字段完全一致
- 字段合法性：不要输出 Schema 中不存在的字段；如果问题里出现未建模维度，优先回退到产品线、区域、客户、月份等已有维度
"""


def build_prompt(
        user_question: str,
        use_few_shot: bool = True,
        use_cot: bool = False,
        use_rules: bool = False,
        use_guards: bool = False,
        ) -> tuple[str, str]:
    """构建提示词"""

    system_msg = "你是一个专业的 SQL 生成助手，擅长根据业务问题生成标准 MySQL 查询语句。"
    if use_rules or use_guards:
        system_msg = "你是一个专业的 SQL 生成助手，擅长根据业务问题生成标准 MySQL 查询语句。请严格遵守给定的业务规则，避免常见错误。"

    prompt = f"""【数据库Schema】
    {SCHEMA}
    """

    if use_cot:
        prompt += f"""
    {COT_INSTRUCTION}
    """

    if use_rules:
        prompt += f"""
    {RULES}
    """

    if use_few_shot:
        prompt += f"""
    【示例】
    {FEW_SHOT_EXAMPLES}
    """

    if use_guards:
        prompt += f"""
    {ERROR_GUARDS}
    """


    prompt += f"""
    【用户问题】
    {user_question}


    【要求】
    1. 只输出 SQL 语句，不需要解释
    2. 使用标准 MySQL 语法
    3. 确保字段名和表名与 Schema 一致
    4. 如果涉及多表查询，使用 JOIN 连接
    5. SQL 必须完整闭合，CTE、SELECT、GROUP BY、ORDER BY 不得省略，不要输出截断的半句 SQL
    """
    if use_rules or use_guards:
        prompt += """6. 优先遵循【关键业务规则】和【常见错误防护】中的约束
    """

    prompt += """
    请直接输出 SQL：
    """
    return system_msg,prompt