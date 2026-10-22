import re
import sys
from datetime import datetime
from decimal import Decimal
from typing import Literal, Any, Callable

from pydantic import BaseModel, Field
import pymysql
from config import DB_CONFIG


#=================== 中间状态定义 ====================

class DecomposedTask(BaseModel):
    """单个子任务定义。
    包含任务ID、名称、类型、依赖任务ID列表、维度和指标。"""

    task_id: str
    task_name: str
    task_type: str
    description: str
    depends_on: list[str] = Field(default_factory=list)
    dimensions: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)


class DecompositionPlan(BaseModel):
    """复杂查询拆解结果。
    包含分析问题类型、分析目标和拆解后的子任务列表。每个子任务包含任务 ID、名称、类型、依赖、维度和指标。"""

    question_type: str
    analysis_goal: str
    subtasks: list[DecomposedTask]


class PlanStep(BaseModel):
    """单个执行步骤定义——一个步骤对应 QueryDecomposer 产出的一个 DecomposedTask。"""

    step_id: str                                          # 步骤唯一标识，如 "step_1"
    task_id: str                                          # 关联的原始任务 ID，如 "task_1"
    step_name: str                                        # 步骤名称
    task_type: str                                        # 任务类型：trend / diagnosis / dimension / summary
    action: str = "text2sql"                              # 执行动作类型，固定为 text2sql
    question: str                                         # 拼装后的自然语言问题，传给 ChatBISystem.run()
    description: str                                      # 子任务描述
    depends_on: list[str] = Field(default_factory=list)   # 前置步骤 ID 列表，形成 DAG 依赖
    metrics: list[str] = Field(default_factory=list)      # 关注的指标列表
    dimensions: list[str] = Field(default_factory=list)   # 分析维度列表
    expected_output: str                                  # 预期输出描述


class ExecutionPlan(BaseModel):
    """可执行计划——由 PlanGenerator 将 DecompositionPlan 转换而来。"""

    original_question: str           # 用户原始问题
    question_type: str               # 问题类型：trend / diagnosis / dimension
    analysis_goal: str               # 分析目标文本
    steps: list[PlanStep]            # 有序执行步骤列表（按 DAG 拓扑序排列）


class StepExecutionResult(BaseModel):
    """单步执行结果——StepExecutor 执行每个步骤后产出的结构化结果。"""

    step_id: str                                          # 步骤 ID
    task_id: str                                          # 关联任务 ID
    step_name: str                                        # 步骤名称
    success: bool                                         # 是否执行成功
    status: Literal["completed", "failed", "skipped"] = "completed"  # 执行状态
    attempts: int = 1                                     # 实际尝试次数（含重试）
    question: str                                         # 实际传给 ChatBISystem 的问题文本（含前置上下文）
    depends_on: list[str] = Field(default_factory=list)   # 依赖步骤
    context_used: str = ""                                # 注入本步骤的前置步骤上下文
    storage_backend: str | None = None                    # 中间结果存储方式: memory / temp_table
    result_reference: str | None = None                   # 中间结果引用地址，供后续步骤通过 result_store 读取
    sql: str | None = None                                # 执行的 SQL 语句
    columns: list[str] = Field(default_factory=list)      # 返回列名
    rows: list[dict[str, Any]] = Field(default_factory=list)  # 返回数据行（dict 格式）
    formatted: str = ""                                   # 格式化后的结果文本
    error: str | None = None                              # 错误信息（成功时为 None）


class ExecutionSummary(BaseModel):
    """执行摘要——ResultSummarizer 汇聚各步骤结果后的最终输出。"""

    original_question: str                                # 用户原始问题
    analysis_goal: str                                    # 分析目标
    completed_steps: int                                  # 成功完成的步骤数
    failed_steps: int                                     # 执行失败的步骤数
    skipped_steps: int = 0                                # 因依赖失败被跳过的步骤数
    key_findings: list[str] = Field(default_factory=list)  # 各步骤的关键发现摘要
    summary_text: str                                     # 汇总文本描述




#=================== 中间结果存储模块 ====================
class IntermediateResultStore:
    """中间结果存储抽象——定义 put/get/cleanup 接口，支持不同后端实现。"""

    def put(
        self,
        step_id: str,
        columns: list[str],
        rows: list[dict[str, Any]],
    ) -> str:
        """存入中间结果，返回引用地址。"""
        raise NotImplementedError

    def get(self, reference: str) -> list[dict[str, Any]]:
        """通过引用地址获取中间结果。"""
        raise NotImplementedError

    def cleanup(self) -> None:
        """释放临时资源。"""


class MemoryResultStore(IntermediateResultStore):
    """内存存储——进程内字典，适合小数据量、开发调试。"""

    def __init__(self):
        self.data: dict[str, list[dict[str, Any]]] = {}

    def put(
        self,
        step_id: str,
        columns: list[str],
        rows: list[dict[str, Any]],
    ) -> str:
        reference = f"memory://{step_id}"
        self.data[reference] = rows
        return reference

    def get(self, reference: str) -> list[dict[str, Any]]:
        return self.data.get(reference, [])

    def cleanup(self) -> None:
        self.data.clear()


class TempTableResultStore(IntermediateResultStore):
    """临时表存储——通过 MySQL CREATE TEMPORARY TABLE 实现，
    后续 SQL 步骤可直接 JOIN 临时表做关联分析，适合大数据量、跨步骤关联场景。"""

    def __init__(
        self,
        connection_factory: Callable[[], Any] | None = None,
    ):
        self.connection_factory = connection_factory or (
            lambda: pymysql.connect(**DB_CONFIG)
        )
        self.connection: Any | None = None
        self.reference_to_table: dict[str, str] = {}

    def put(
        self,
        step_id: str,
        columns: list[str],
        rows: list[dict[str, Any]],
    ) -> str:
        """创建临时表并插入数据，返回 temp_table:// 引用地址。"""
        conn = self._ensure_connection()
        table_name = f"tmp_agent_{self._sanitize_identifier(step_id)}"
        table_columns = columns or (list(rows[0].keys()) if rows else ["_empty_marker"])
        column_defs = ", ".join(
            f"{self._quote_identifier(column)} {self._infer_sql_type([row.get(column) for row in rows])}"
            for column in table_columns
        )

        with conn.cursor() as cursor:
            cursor.execute(
                f"DROP TEMPORARY TABLE IF EXISTS {self._quote_identifier(table_name)}"
            )
            cursor.execute(
                f"CREATE TEMPORARY TABLE {self._quote_identifier(table_name)} ({column_defs})"
            )
            if rows:
                placeholders = ", ".join(["%s"] * len(table_columns))
                column_names = ", ".join(
                    self._quote_identifier(column) for column in table_columns
                )
                values = [
                    tuple(row.get(column) for column in table_columns)
                    for row in rows
                ]
                cursor.executemany(
                    f"INSERT INTO {self._quote_identifier(table_name)} ({column_names}) VALUES ({placeholders})",
                    values,
                )
        conn.commit()

        reference = f"temp_table://{table_name}"
        self.reference_to_table[reference] = table_name
        return reference

    def get(self, reference: str) -> list[dict[str, Any]]:
        """通过 temp_table:// 引用获取临时表数据。"""
        conn = self._ensure_connection()
        table_name = self.reference_to_table.get(reference) or self._parse_reference(reference)

        with conn.cursor() as cursor:
            cursor.execute(f"SELECT * FROM {self._quote_identifier(table_name)}")
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            rows = cursor.fetchall()

        return [dict(zip(columns, row)) for row in rows]

    def cleanup(self) -> None:
        """释放临时表资源，关闭数据库连接。"""
        if self.connection is None:
            return

        try:
            with self.connection.cursor() as cursor:
                for table_name in set(self.reference_to_table.values()):
                    cursor.execute(
                        f"DROP TEMPORARY TABLE IF EXISTS {self._quote_identifier(table_name)}"
                    )
            self.connection.commit()
        finally:
            self.connection.close()
            self.connection = None
            self.reference_to_table.clear()

    def _ensure_connection(self) -> Any:
        """确保数据库连接已建立，返回连接对象。"""
        if self.connection is None or not getattr(self.connection, "open", True):
            self.connection = self.connection_factory()
        return self.connection

    @staticmethod
    def _sanitize_identifier(value: str) -> str:
        """对 SQL 标识符进行清理，确保符合 MySQL 规则。"""
        sanitized = re.sub(r"[^0-9a-zA-Z_]", "_", value)
        return sanitized.strip("_") or "step"

    @staticmethod
    def _quote_identifier(value: str) -> str:
        """对 SQL 标识符进行转义，确保在查询中安全使用。"""
        escaped = value.replace("`", "``")
        return f"`{escaped}`"

    @staticmethod
    def _infer_sql_type(values: list[Any]) -> str:
        """根据值列表推断 SQL 数据型，返回字符串表示。"""
        non_null = next((value for value in values if value is not None), None)
        if isinstance(non_null, bool):
            return "TINYINT(1)"
        if isinstance(non_null, int):
            return "BIGINT"
        if isinstance(non_null, (float, Decimal)):
            return "DOUBLE"
        return "TEXT"

    @staticmethod
    def _parse_reference(reference: str) -> str:
        """解析 temp_table:// 引用，返回实际表名。"""
        if not reference.startswith("temp_table://"):
            raise ValueError(f"非法的临时表引用：{reference}")
        return reference.split("://", 1)[1]




#=================== 进度打印 ====================
def print_progress(message: str) -> None:
    """统一进度日志打印，输出到 stderr，不影响 stdout 的 JSON 输出。"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", file=sys.stderr, flush=True)
