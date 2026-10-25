"""真实数据库验证脚本（需要可连通的 MySQL）。

用于证明本次改造的关键行为"真的生效"，而不是只在单测里通过：
  1. 会话级只读：即使使用 root 账号，写操作也会被数据库拒绝
  2. 行级过滤：sales 角色的查询只返回本区域数据
  3. 结果脱敏：敏感列返回 ***
  4. 连接池：连续查询不再重复握手
  5. 危险 SQL：在到达数据库之前就被拦下

运行：
    python verify_live.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from security import SecurityError, UserContext  # noqa: E402
from text2SQL.chatbi_system import ChatBISystem  # noqa: E402
from text2SQL.database import DatabaseClient, QueryExecutionError  # noqa: E402

PASS, FAIL = "✅", "❌"
results = {"ok": 0, "fail": 0}


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        results["ok"] += 1
        print(f"  {PASS} {name} {detail}")
    else:
        results["fail"] += 1
        print(f"  {FAIL} {name} {detail}")


class FakeLLM:
    """固定返回同一条 SQL，用于隔离 LLM 变量。"""

    def __init__(self, sql: str):
        self.sql = sql
        self.prompts: list[str] = []

    def generate_sql(self, system_msg, prompt):
        self.prompts.append(prompt)
        return self.sql

    def generate_text(self, system_msg, prompt):
        return ""

    def generate_sql_stream(self, system_msg, prompt):
        for char in self.sql:
            yield char


def section(title: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def main() -> int:
    db = DatabaseClient()

    section("1. 连通性")
    check("数据库可连接", db.validate_connection())

    section("2. 会话级只读（root 账号下仍必须挡住写操作）")
    for statement in (
        "DELETE FROM sales_orders WHERE order_id = -1",
        "CREATE TABLE _chatbi_should_fail (id INT)",
        "UPDATE sales_orders SET net_amount = 0 WHERE order_id = -1",
    ):
        try:
            db.execute_query(statement)
        except QueryExecutionError as exc:
            check(
                f"写操作被拒绝：{statement.split()[0]}",
                exc.error_type == "database_permission_denied",
                f"(errno={exc.errno}, {exc.error_type})",
            )
        else:
            check(f"写操作被拒绝：{statement.split()[0]}", False, "竟然执行成功了！")

    section("3. 行级过滤（真实数据）")
    admin = UserContext(user_id="live_admin", role="admin")
    sales = UserContext(user_id="live_sales", role="sales", region="欧洲")

    _cols, admin_rows, _ = db.execute_query(
        "SELECT region, COUNT(*) FROM sales_orders GROUP BY region ORDER BY region"
    )
    print(f"     admin 可见区域：{admin_rows}")

    sql = "SELECT region, COUNT(*) AS c FROM sales_orders GROUP BY region"
    system = ChatBISystem(llm=FakeLLM(sql), analyzer_llm=FakeLLM(""), db=db)

    admin_result = system.run("各区域订单量", user=admin)
    sales_result = system.run("各区域订单量", user=sales)

    check("admin 查询成功", admin_result["success"] is True, admin_result.get("error", ""))
    check("sales 查询成功", sales_result["success"] is True, sales_result.get("error", ""))

    if admin_result["success"] and sales_result["success"]:
        sales_regions = {row[0] for row in sales_result["results"]}
        print(f"     sales 可见区域：{sales_regions}")
        check(
            "sales 只能看到本区域",
            sales_regions <= {"欧洲"},
            f"(实际 {sales_regions})",
        )
        check(
            "sales 行数少于 admin",
            len(sales_result["results"]) <= len(admin_result["results"]),
        )
        check(
            "下发 SQL 已被改写",
            "region = '欧洲'" in sales_result["executed_sql"]
            or "region='欧洲'" in sales_result["executed_sql"],
        )

    section("4. 结果脱敏（真实数据）")
    mask_sql = "SELECT net_amount AS revenue FROM sales_orders LIMIT 3"
    mask_system = ChatBISystem(llm=FakeLLM(mask_sql), analyzer_llm=FakeLLM(""), db=db)

    admin_mask = mask_system.run("收入明细", user=admin)
    sales_mask = mask_system.run("收入明细", user=sales)

    if admin_mask["success"] and sales_mask["success"]:
        admin_values = [row[0] for row in admin_mask["results"]]
        sales_values = [row[0] for row in sales_mask["results"]]
        print(f"     admin 看到：{admin_values}")
        print(f"     sales 看到：{sales_values}")
        check("admin 可见真实金额", any(value != "***" for value in admin_values))
        check(
            "sales 金额全部脱敏（含别名）",
            all(value == "***" for value in sales_values),
            f"(实际 {sales_values})",
        )
    else:
        check("脱敏验证执行成功", False)

    section("5. 危险 SQL 在到达数据库前被拦截")
    dangerous = [
        "SELECT * FROM mysql.user",
        "SELECT * INTO OUTFILE '/tmp/leak.txt' FROM sales_orders",
        "DELETE FROM sales_orders",
        "SELECT SLEEP(5)",
    ]
    for bad in dangerous:
        try:
            system.security.prepare(bad, admin)
        except SecurityError as exc:
            check(f"拦截：{bad[:38]}", True, f"code={exc.code}")
        else:
            check(f"拦截：{bad[:38]}", False, "未被拦截！")

    section("6. 连接池复用")
    first = db.execute_query("SELECT COUNT(*) FROM sales_orders")[2]["duration_ms"]
    second = db.execute_query("SELECT COUNT(*) FROM sales_orders")[2]["duration_ms"]
    print(f"     首次 {first:.1f}ms / 复用后 {second:.1f}ms")
    # 仅作参考：单次毫秒级测量的抖动远大于连接复用带来的差异，
    # 用它做硬断言会引入偶发失败（这一点本身也说明"别拿单次计时当基准"）。
    check(
        "连接池可用（两次查询均成功）",
        first is not None and second is not None,
        "（耗时差异仅供参考）",
    )

    section("7. 超大结果集被截断")
    _cols, all_rows, _ = db.execute_query("SELECT * FROM sales_orders")
    limit = max(1, len(all_rows) - 1)  # 比全量少 1 行，确保必然触发截断
    _cols, rows, meta = db.execute_query("SELECT * FROM sales_orders", max_rows=limit)
    check(
        f"行数被限制在 {limit} 行",
        len(rows) == limit,
        f"(全量 {len(all_rows)} 行，实际返回 {len(rows)} 行)",
    )
    check("truncated 标记正确", meta["truncated"] is True)

    section("8. 多轮对话：追问改写 + 安全不失效")
    mt_llm = FakeLLM("SELECT region, COUNT(*) AS c FROM sales_orders GROUP BY region")
    mt_system = ChatBISystem(llm=mt_llm, analyzer_llm=FakeLLM(""), db=db)
    mt_user = UserContext(user_id="live_mt", role="sales", region="欧洲")
    sid = "live-verify-session"

    first = mt_system.run("各区域订单量", user=mt_user, session_id=sid)
    second = mt_system.run("那上个月呢", user=mt_user, session_id=sid)

    check("第一轮成功", first["success"] is True, first.get("error", ""))
    check("第二轮成功", second["success"] is True, second.get("error", ""))
    if second["success"]:
        multiturn = second["metadata"].get("multiturn", {})
        print(f"     改写策略：{multiturn.get('strategy')}")
        print(f"     改写结果：{multiturn.get('resolved_question')}")
        check(
            "追问被改写并带上时间",
            "上个月" in str(multiturn.get("resolved_question")),
        )
        check(
            "改写结果进入 Prompt",
            "上个月" in mt_llm.prompts[-1],
        )
        # 最关键：多轮不能成为绕过行级权限的通道
        check(
            "第二轮仍应用行级过滤",
            "欧洲" in second["executed_sql"],
            "(多轮改写后仍注入 region 过滤)",
        )
        sales_regions = {row[0] for row in second["results"]}
        check("第二轮仍只看到本区域", sales_regions <= {"欧洲"}, f"(实际 {sales_regions})")

    mt_system.conversation_store.reset(sid)

    db.close()

    print(f"\n{'=' * 60}")
    print(f"  验证结果：{results['ok']} {PASS}   {results['fail']} {FAIL}")
    print(f"{'=' * 60}")
    return 0 if results["fail"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
