import re
import sys
from decimal import Decimal

from indicator_knowledge import IndicatorKnowledge
from query_parser import QueryParser
from prompt_builder import build_prompt
from llm_client import LLMClient
from database import DatabaseClient, QueryExecutionError
from result_formatter import ResultFormatter
from error_analyzer import ErrorAnalyzer
from config import LLM_ANALYZER_CONFIG, LLM_CONFIG
import json
from typing import Generator


from security import SecurityError, UserContext

SYSTEM_PROMPT = '你是一个专业的 SQL 生成助手，只输出 SQL 语句，不要任何额外解释。'


class ChatBISystem:
    """ChatBI 系统主类"""

    def __init__(self):
        self.parser = QueryParser()
        self.llm = LLMClient()
        self.analyzer_llm = LLMClient(config=LLM_ANALYZER_CONFIG) #TODO 需要重新设计llm_client,
        self.db = DatabaseClient()
        self.formatter = ResultFormatter()
        self.indicator_knowledge = IndicatorKnowledge()

    def run(self,
            user_question: str,
            use_few_shot: bool = True,
            use_rules: bool = False,
            use_guards: bool = False,
            use_indicator_knowledge: bool = False,
            use_schema_linking: bool = False,
            use_indicator_rag: bool = False,
            ) -> dict:
        """运行完整链路"""
        # 1. 解析问题
        parsed = self.parser.parse(user_question)

        if not self.parser.validate(parsed):
            return {"success": False, "error": "输入问题为空"}

        # 2. 构造 Prompt
        detected_indicators = []
        indicator_block = ""
        if use_indicator_knowledge:
            detected_indicators, indicator_block = self.indicator_knowledge.get_indicator_context(user_question)
        system_msg, prompt_msg = build_prompt(
            user_question,
            use_few_shot=use_few_shot,
            use_rules=use_rules,
            use_guards=use_guards,
            indicator_knowledge=indicator_block,
            use_schema_linking=use_schema_linking,
        )
        # 3. 生成 SQL
        sql = self.llm.generate_sql(system_msg, prompt_msg)

        # 4. 执行 SQL
        try:
            columns, results = self.db.execute(sql)
            formatted = self.formatter.format(columns, results)

            return {
                "success": True,
                "sql": sql,
                "columns": columns,
                "results": results,
                "formatted": formatted
            }

        except Exception as e:
            # 5. sql 执行失败处理
            error_analyzer = ErrorAnalyzer(llm_client=self.analyzer_llm)
            error_result = error_analyzer.categorize_error(sql, str(e), user_question)

            return {
                "success": False,
                "sql": sql,
                "error": error_result
            }

    def run_stream(
            self,
            user_question: str,
            use_few_shot: bool = True,
            use_rules: bool = True,
            use_guards: bool = True,
            use_indicator_knowledge: bool = True,
            use_schema_linking: bool = False,
            use_indicator_rag: bool = False,
    ) -> Generator[str, None, None]:
        # 1. 解析问题
        parsed = self.parser.parse(user_question)

        if not self.parser.validate(parsed):
            yield _sse_event("error", {
                "error": "输入问题为空",
                "error_type": "validation"
            })
            return

        # 2. 构造 Prompt
        detected_indicators = []
        indicator_block = ""

        if use_indicator_knowledge:
            detected_indicators, indicator_block = self.indicator_knowledge.get_indicator_context(user_question)

        system_msg, prompt = build_prompt(
            user_question,
            use_few_shot=use_few_shot,
            use_rules=use_rules,
            use_guards=use_guards,
            indicator_knowledge=indicator_block,
            use_schema_linking=use_schema_linking,
        )

        # 3. 流式生成 SQL —— 逐 chunk 推送
        sql_parts = []

        try:
            for chunk_text in self.llm.generate_sql_stream(system_msg, prompt):
                sql_parts.append(chunk_text)
                yield _sse_event("sql_chunk", {"content": chunk_text})

        except Exception as e:
            yield _sse_event("error", {"error": str(e), "error_type": "llm"})
            return

        # 4. 拼接完整 SQL 并清理 markdown 标记
        raw_sql = "".join(sql_parts)
        sql = re.sub(r'```sql|```', '', raw_sql).strip()
        yield _sse_event("sql_done", {"sql": sql})

        # 5. 执行 SQL

        try:
            columns, results = self.db.execute(sql)
            rows_dict = [dict(zip(columns, row)) for row in results]

            yield _sse_event("result", {
                "columns": columns,
                "rows": rows_dict,
                "row_count": len(results),
            })

        except Exception as e:

            yield _sse_event("error", {
                "error": str(e),
                "error_type": "database",
                "sql": sql,
            })


def _sse_event(event_type: str, data: dict) -> str:
    """构造 SSE 格式的事件字符串"""
    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False, default=_json_serializer)}\n\n"


def _json_serializer(obj):
    """JSON 序列化补充：处理 Decimal 等非标准类型"""
    if isinstance(obj, Decimal):
        return float(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def main():
    system = ChatBISystem()
    print("=" * 60)
    print("ChatBI Text2SQL 系统")
    print("=" * 60)

    if len(sys.argv) > 1:
        # 命令行模式
        question = " ".join(sys.argv[1:])
        result = system.run(question)
        _print_result(question, result)
        return

    # 交互模式
    print("\n请输入问题（输入 exit/quit/q 退出）：")

    while True:

        try:
            question = input("\n> ")
            if question.strip().lower() in ["exit", "quit", "q"]:
                break
            for event in system.run_stream(question):
                print(event)

        except (KeyboardInterrupt, EOFError):
            break

        except Exception as e:
            print(f"系统错误：{e}")

    print("\n感谢使用！")


def _print_result(question: str, result: dict):
    print(f"\n问题：{question}")
    print(f"SQL：{result.get('sql', '')}")

    if result["success"]:
        print(f"\n{result['formatted']}")

    else:
        print(f"\n错误：{result['error']}")


if __name__ == "__main__":
    main()