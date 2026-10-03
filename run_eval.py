"""SQL 生成评估命令行入口。

默认离线自检(用黄金 SQL 当作生成结果,验证评分逻辑):
    .venv\\Scripts\\python.exe run_eval.py

使用真实 LLM 生成:
    .venv\\Scripts\\python.exe run_eval.py --live

额外连接数据库比较执行结果集:
    .venv\\Scripts\\python.exe run_eval.py --live --db
"""

import argparse
import json

from evaluation.evaluator import Evaluator


def _load_cases(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data.get("cases", [])
    return data


def _make_mock_generator(cases: list):
    lookup = {case["question"]: case["golden_sql"] for case in cases}

    def generate(question: str) -> str:
        return lookup[question]

    return generate


def _make_llm_generator():
    from llm_client import LLMClient
    from config import LLM_ANALYZER_CONFIG
    from prompt_builder import build_prompt

    generator_llm = LLMClient()
    analyzer_llm = LLMClient(config=LLM_ANALYZER_CONFIG)

    def generate(question: str) -> str:
        system_msg, prompt = build_prompt(
            question, use_few_shot=True, use_rules=True, use_guards=True, use_cot=True
        )
        return generator_llm.generate_sql(system_msg, prompt)

    return generate, analyzer_llm


def _make_db_executor():
    from database import DatabaseClient

    db = DatabaseClient()

    def execute(sql: str) -> list:
        _, rows = db.execute(sql)
        return rows

    return execute


def main() -> None:
    parser = argparse.ArgumentParser(description="SQL 生成评估")
    parser.add_argument("--cases", default="evaluation/cases.json")
    parser.add_argument(
        "--live",
        action="store_true",
        help="使用真实 LLM 生成 SQL；否则用黄金 SQL 自检评分逻辑",
    )
    parser.add_argument(
        "--db",
        action="store_true",
        help="连接数据库执行 SQL 并比较结果集",
    )
    parser.add_argument("--limit", type=int, default=None, help="仅评估前 N 个用例")
    parser.add_argument("--out", default=None, help="把完整 JSON 结果写入该文件")
    args = parser.parse_args()

    cases = _load_cases(args.cases)
    if args.limit:
        cases = cases[: args.limit]

    if args.live:
        generate_fn, analyzer_llm = _make_llm_generator()
    else:
        generate_fn = _make_mock_generator(cases)
        analyzer_llm = None

    execute_fn = _make_db_executor() if args.db else None

    evaluator = Evaluator(
        generate_fn=generate_fn, execute_fn=execute_fn, llm_client=analyzer_llm
    )
    result = evaluator.evaluate(cases)
    print(evaluator.render(result))

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n完整结果已写入: {args.out}")


if __name__ == "__main__":
    main()