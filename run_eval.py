"""SQL 生成评估命令行入口。

默认离线自检(用黄金 SQL 当作生成结果,验证评分逻辑):
    .venv\\Scripts\\python.exe run_eval.py

使用真实 LLM 生成:
    .venv\\Scripts\\python.exe run_eval.py --live

额外连接数据库比较执行结果集:
    .venv\\Scripts\\python.exe run_eval.py --live --db

常用参数:
  --cases    指定用例文件 (默认 evaluation/test_cases.json)
  --limit    只跑前 N 个用例,方便快速迭代
  --workers  并发线程数 (默认 4),设为 1 退化为串行
  --out      把完整 JSON 结果落盘,便于后续分析
"""

import argparse
import json
import time
from typing import Optional

from evaluation.evaluator import Evaluator


def _load_cases(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data.get("cases", [])
    return data


def _make_mock_generator(cases: list):
    # 直接返回黄金 SQL,用于验证评测链路自身逻辑是否正常
    lookup = {case["question"]: case["golden_sql"] for case in cases}

    def generate(question: str) -> str:
        return lookup[question]

    return generate


def _make_llm_generator():
    # 创建生成器和分析器两个 LLM 客户端:前者生成 SQL,后者做错误深度分析
    from llm_client import LLMClient
    from config import LLM_ANALYZER_CONFIG
    from prompt_builder import build_prompt

    # 生成器和分析器分别使用不同的 LLM 配置
    generator_llm = LLMClient()
    analyzer_llm = LLMClient(config=LLM_ANALYZER_CONFIG)

    def generate(question: str) -> str:
        system_msg, prompt = build_prompt(
            question, use_few_shot=True, use_rules=True, use_guards=True, use_cot=True
        )
        return generator_llm.generate_sql(system_msg, prompt)

    return generate, analyzer_llm


def _make_db_executor():
    # 返回仅取结果集的执行函数,供 Evaluator 比较黄金 SQL 和生成 SQL 的输出
    from database import DatabaseClient

    db = DatabaseClient()

    def execute(sql: str) -> list:
        _, rows = db.execute(sql)
        return rows

    return execute


def _progress(done: int, total: int) -> None:
    print(f"[{done}/{total}] 已完成...")


def run_evaluation(
    cases_path: str = "evaluation/test_cases.json",
    live: bool = True,
    use_db: bool = True,
    limit: Optional[int] = None,
    out_path: Optional[str] = None,
    workers: int = 4,
) -> dict:
    """运行 SQL 生成评估,返回评估结果"""
    cases = _load_cases(cases_path)
    if limit:
        cases = cases[:limit]

    if live:
        generate_llm, analyzer_llm = _make_llm_generator()
    else:
        generate_llm = _make_mock_generator(cases)
        analyzer_llm = None

    execute_fn = _make_db_executor() if use_db else None

    evaluator = Evaluator(
        generate_llm=generate_llm,
        execute_fn=execute_fn,
        llm_client=analyzer_llm,
        max_workers=workers,
        progress_callback=_progress,
    )

    start = time.perf_counter()
    result = evaluator.evaluate(cases)
    elapsed = time.perf_counter() - start

    print(f"\n耗时: {elapsed:.1f}s (workers={workers})\n")
    print(evaluator.render(result))

    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n完整结果已写入: {out_path}")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="SQL 生成评估")
    parser.add_argument("--cases", default="evaluation/test_cases.json")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--db", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    run_evaluation(
        cases_path=args.cases,
        live=args.live,
        use_db=args.db,
        limit=args.limit,
        out_path=args.out,
        workers=args.workers,
    )


if __name__ == "__main__":
    run_evaluation(
        cases_path="evaluation/test_cases.json",
        live=True,
        use_db=True,
        limit=None,
        out_path=None,
        workers=4,
    )