"""
结果格式化模块

将数据库查询结果格式化为可读的表格形式,支持控制台输出和错误提示。
"""

from typing import List, Tuple


class ResultFormatter:
    """结果格式化器"""

    def format(
        self, columns: List[str], results: List[tuple]
    ) -> str:
        """将查询结果格式化为字符串表格

        Args:
            columns: 列名列表
            results: 查询结果行列表

        Returns:
            格式化后的字符串表格
        """
        if not results:
            return "查询结果为空"

        col_widths = []
        for i, col in enumerate(columns):
            max_data_width = max((len(str(row[i])) for row in results), default=0)
            col_widths.append(max(len(col), max_data_width) + 2)

        header = "|" + "|".join(
            col.ljust(col_widths[i]) for i, col in enumerate(columns)
        ) + "|"
        separator = "+" + "+".join("-" * w for w in col_widths) + "+"

        rows = []
        for row in results:
            row_str = "|" + "|".join(
                str(val).ljust(col_widths[i]) for i, val in enumerate(row)
            ) + "|"
            rows.append(row_str)

        return (
            f"{separator}\n{header}\n{separator}\n"
            + "\n".join(rows)
            + f"\n{separator}"
        )

    def format_error(self, error_msg: str) -> str:
        """格式化错误信息"""
        return f"执行出错: {error_msg}"