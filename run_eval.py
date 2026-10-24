"""SQL 生成评估命令行入口。

三种模式（早期版本这里写的是"默认离线自检"，但 `__main__` 里硬编码了 live=True，
且 `from llm_client import ...` 因为模块已移入 text2SQL 包而必然 ImportError——
也就是说"100% 通过"这个结论当时根本无法复现）：

    # 1) 自检：用黄金 SQL 当作生成结果，验证"尺子"本身是否正确（不消耗 API）
    python run_eval.py --mode selfcheck

    # 2) 只测 Prompt + LLM（可选接数据库比结果集）
    python run_eval.py --mode llm --suite dev --db

    # 3) 端到端（含安全校验 / 自动修复 / 脱敏），这才是线上真实表现
    python run_eval.py --mode pipeline --suite holdout --db

常用参数：
    --suite           dev | holdout | edge，或直接给 json 路径
    --db              连接数据库按执行结果集判定（推荐，exact match 过于严格）
    --baseline        与基线对比，输出回归/修复明细
    --save-baseline   把本次结果存为基线
    --fail-under      通过率低于该值时退出码为 1（用于 CI 门禁）
    --limit / --workers / --out
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Callable, Optional

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from evaluation.evaluator import Evaluator  # noqa: E402

SUITE_DIR = BASE_DIR / "evaluation" / "suites"


# ==================== 用例加载 ====================
def resolve_suite(name_or_path: str) -> Path:
    candidate = Path(name_or_path)
    if candidate.exists():
        return candidate
    named = SUITE_DIR / f"{name_or_path}.json"
    if named.exists():
        return named
    available = sorted(p.stem for p in SUITE_DIR.glob("*.json"))
    raise SystemExit(f"找不到用例集 {name_or_path!r}，可用：{available}")


def load_suite(path: Path) -> tuple[list[dict], dict]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, list):
        return data, {}
    meta = {key: value for key, value in data.items() if key != "cases"}
    return data.get("cases", []), meta


# ==================== 生成器 ====================
def make_selfcheck_generator(cases: list[dict]) -> Callable[[str], str]:
    """把黄金 SQL 当作模型输出，验证评分逻辑自身是否正确。"""
    lookup = {case["question"]: case.get("golden_sql", "") for case in cases}

    def generate(question: str) -> str:
        return lookup.get(question, "")

    return generate


def make_llm_generator():
    """只测 Prompt → LLM，不经过安全层与数据库。"""
    from text2SQL.llm_client import LLMClient
    from text2SQL.prompt_builder import build_prompt

    generator_llm = LLMClient()

    def generate(question: str) -> str:
        system_msg, prompt = build_prompt(
            question, use_few_shot=True, use_rules=True, use_guards=True, use_cot=True
        )
        return generator_llm.generate_sql(system_msg, prompt)

    return generate


def make_pipeline_generator(role: str = "admin"):
    """端到端：走 ChatBISystem.run（含安全校验、自动修复、脱敏）。"""
    from security import UserContext
    from text2SQL.chatbi_system import ChatBISystem

    system = ChatBISystem()
    user = UserContext(user_id=f"eval_{role}", role=role)

    def generate(question: str) -> str:
        result = system.run(
            user_question=question,
            user=user,
            use_few_shot=True,
            use_rules=True,
            use_guards=True,
            use_schema_linking=True,
            use_indicator_knowledge=True,
        )
        if not result.get("success"):
            # 抛错让 Evaluator 记录为失败，并保留错误类型用于归因
            raise RuntimeError(
                f"[{result.get('error_type')}] {result.get('error')}"
            )
        return result["sql"]

    return generate


def make_db_executor():
    from text2SQL.database import DatabaseClient

    db = DatabaseClient()

    def execute(sql: str) -> list:
        _columns, rows = db.execute(sql)
        return rows

    return execute


# ==================== 基线与门禁 ====================
def load_baseline(path: Optional[str]) -> Optional[dict]:
    if not path:
        return None
    candidate = Path(path)
    if not candidate.exists():
        print(f"[warn] 基线文件不存在，跳过对比：{candidate}")
        return None
    with candidate.open(encoding="utf-8") as handle:
        return json.load(handle)


def diff_with_baseline(result: dict, baseline: dict | None) -> dict:
    """对比基线，找出回归（pass→fail）与修复（fail→pass）。"""
    if not baseline:
        return {"regressions": [], "fixes": [], "baseline_pass_rate": None}

    baseline_cases = baseline.get("cases", {})
    regressions, fixes = [], []
    for case in result["cases"]:
        case_id = str(case.get("id"))
        before = baseline_cases.get(case_id)
        after = "pass" if case["passed"] else "fail"
        if before == "pass" and after == "fail":
            regressions.append(case_id)
        elif before == "fail" and after == "pass":
            fixes.append(case_id)

    return {
        "regressions": regressions,
        "fixes": fixes,
        "baseline_pass_rate": (baseline.get("summary") or {}).get("pass_rate"),
    }


def as_baseline(result: dict, suite: str, mode: str) -> dict:
    return {
        "suite": suite,
        "mode": mode,
        "cases": {
            str(case.get("id")): ("pass" if case["passed"] else "fail")
            for case in result["cases"]
        },
        "summary": result["summary"],
    }


# ==================== 主流程 ====================
def run_evaluation(
    suite: str = "dev",
    mode: str = "selfcheck",
    use_db: bool = False,
    limit: Optional[int] = None,
    workers: int = 1,
    out_path: Optional[str] = None,
    baseline_path: Optional[str] = None,
    save_baseline: bool = False,
    fail_under: Optional[float] = None,
    role: str = "admin",
) -> dict:
    path = resolve_suite(suite)
    cases, meta = load_suite(path)
    if limit:
        cases = cases[:limit]

    print(f"用例集: {path.name}（{len(cases)} 条） 模式: {mode}  DB: {use_db}")
    if meta.get("frozen"):
        print("注意: 本用例集已冻结，禁止据其结果修改 Prompt（否则失去留出集意义）")

    if mode == "selfcheck":
        cases = [case for case in cases if case.get("golden_sql")]
        generate_llm = make_selfcheck_generator(cases)
        analyzer_llm = None
        if not cases:
            print("该用例集没有可用黄金 SQL（如 edge 拒答集），selfcheck 模式跳过。")
            return {"cases": [], "summary": {}, "_exit_code": 0, "_diff": {}}
    elif mode == "llm":
        generate_llm = make_llm_generator()
        from config import LLM_ANALYZER_CONFIG
        from text2SQL.llm_client import LLMClient

        analyzer_llm = LLMClient(config=LLM_ANALYZER_CONFIG)
    elif mode == "pipeline":
        generate_llm = make_pipeline_generator(role=role)
        analyzer_llm = None
    else:
        raise SystemExit(f"未知模式：{mode}")

    execute_fn = make_db_executor() if use_db else None

    evaluator = Evaluator(
        generate_llm=generate_llm,
        execute_fn=execute_fn,
        llm_client=analyzer_llm,
        max_workers=workers,
        progress_callback=lambda done, total: print(f"  [{done}/{total}]"),
    )

    started = perf_counter()
    result = evaluator.evaluate(cases)
    elapsed = perf_counter() - started

    print(f"\n耗时: {elapsed:.1f}s (workers={workers})\n")
    print(evaluator.render(result))

    baseline = load_baseline(baseline_path)
    diff = diff_with_baseline(result, baseline)
    if baseline:
        print("\n" + "=" * 72)
        print("与基线对比")
        print("=" * 72)
        print(f"基线通过率: {diff['baseline_pass_rate']}")
        print(f"本次通过率: {result['summary']['pass_rate']}")
        if diff["regressions"]:
            print(f"❌ 回归（{len(diff['regressions'])} 条）: {diff['regressions']}")
        if diff["fixes"]:
            print(f"✅ 修复（{len(diff['fixes'])} 条）: {diff['fixes']}")
        if not diff["regressions"] and not diff["fixes"]:
            print("无变化")

    if out_path:
        Path(out_path).write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n完整结果已写入: {out_path}")

    if save_baseline:
        target = Path(baseline_path or f"evaluation/baseline_{suite}_{mode}.json")
        target.write_text(
            json.dumps(as_baseline(result, suite, mode), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"基线已保存: {target}")

    # CI 门禁
    exit_code = 0
    if fail_under is not None:
        pass_rate = result["summary"]["pass_rate"]
        if pass_rate < fail_under:
            print(f"\n❌ 门禁失败：通过率 {pass_rate:.2%} < 阈值 {fail_under:.2%}")
            exit_code = 1
    if diff["regressions"]:
        print(f"\n❌ 门禁失败：检出 {len(diff['regressions'])} 条回归")
        exit_code = 1

    result["_exit_code"] = exit_code
    result["_diff"] = diff
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="SQL 生成评估")
    parser.add_argument("--suite", default="dev", help="dev | holdout | edge | <path>")
    parser.add_argument(
        "--mode",
        default="selfcheck",
        choices=["selfcheck", "llm", "pipeline"],
        help="selfcheck=验证评分逻辑; llm=只测Prompt+LLM; pipeline=端到端",
    )
    parser.add_argument("--db", action="store_true", help="连接数据库按结果集判定")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--out", default=None)
    parser.add_argument("--baseline", default=None, help="基线文件，用于回归对比")
    parser.add_argument("--save-baseline", action="store_true")
    parser.add_argument("--fail-under", type=float, default=None, help="通过率门禁阈值")
    parser.add_argument("--role", default="admin", help="pipeline 模式使用的角色")
    args = parser.parse_args()

    result = run_evaluation(
        suite=args.suite,
        mode=args.mode,
        use_db=args.db,
        limit=args.limit,
        workers=args.workers,
        out_path=args.out,
        baseline_path=args.baseline,
        save_baseline=args.save_baseline,
        fail_under=args.fail_under,
        role=args.role,
    )
    sys.exit(result.get("_exit_code", 0))


if __name__ == "__main__":
    main()
