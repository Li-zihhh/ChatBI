import sys
from query_parser import QueryParser
from prompt_builder import build_prompt
from llm_client import LLMClient
from database import DatabaseClient
from result_formatter import ResultFormatter
from error_analyzer import ErrorAnalyzer
from config import LLM_ANALYZER_CONFIG


SYSTEM_PROMPT = '你是一个专业的 SQL 生成助手，只输出 SQL 语句，不要任何额外解释。'


class ChatBISystem:
    """ChatBI 系统主类"""

    def __init__(self):
        self.parser = QueryParser()
        self.llm = LLMClient()
        self.analyzer_llm = LLMClient(config=LLM_ANALYZER_CONFIG)
        self.db = DatabaseClient()
        self.formatter = ResultFormatter()

    def run(self, user_question: str, use_few_shot: bool = True) -> dict:
        """运行完整链路"""
        # 1. 解析问题
        parsed = self.parser.parse(user_question)

        if not self.parser.validate(parsed):
            return {"success": False, "error": "输入问题为空"}

        # 2. 构造 Prompt
        system_msg, prompt_msg = build_prompt(user_question, use_few_shot)

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
            result = system.run(question)
            _print_result(question, result)

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