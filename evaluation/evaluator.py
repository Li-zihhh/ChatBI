"""SQL 生成评估器：跑用例、算指标、出报告，并接入错误分析。"""

from __future__ import annotations

from collections import Counter
from typing import Any, Callable, Dict, List, Optional

from error_analyzer import ErrorAnalyzer

from . import metrics


class Evaluator:
    def __init__(
        self,
        generate_fn: Callable[[str], str],
        execute_fn: Optional[Callable[[str], list]] = None,
        llm_client=None,
    ):
        self.generate_fn = generate_fn
        self.execute_fn = execute_fn
        self.error_analyzer = ErrorAnalyzer(llm_client=llm_client)

    def evaluate(self, cases: List[dict]) -> Dict[str, Any]:
        results = [self._evaluate_one(case) for case in cases]
        summary = self._summarize(results)
        return {"cases": results, "summary": summary}

    def _evaluate_one(self, case: dict) -> dict:
        question = case["question"]
        golden = case.get("golden_sql", "")
        entry: Dict[str, Any] = {
            "id": case.get("id"),
            "question": question,
            "golden_sql": golden,
            "generated_sql": None,
            "metrics": {},
            "passed": False,
        }

        try:
            generated = self.generate_fn(question)
        except Exception as exc:  # noqa: BLE001
            entry["metrics"] = {"exact_match": False, "structure_score": 0.0}
            entry["error"] = {
                "error_type": "generation_error",
                "root_cause": "",
                "suggestion": "",
                "message": str(exc),
            }
            return entry

        entry["generated_sql"] = generated
        m = metrics.compare(golden, generated)
        entry["metrics"] = m

        if self.execute_fn is not None:
            try:
                gen_rows = self.execute_fn(generated)
                gold_rows = self.execute_fn(golden)
                exec_match = metrics.rows_equal(gold_rows, gen_rows)
            except Exception as exc:  # noqa: BLE001
                exec_match = False
                entry["error"] = self.error_analyzer.categorize_error(
                    generated, str(exc), question
                )
            entry["metrics"]["execution_match"] = exec_match
            entry["passed"] = bool(exec_match)
        else:
            entry["passed"] = bool(m["exact_match"])

        if not entry["passed"] and "error" not in entry:
            entry["error"] = self.error_analyzer.categorize_error(generated, None, question)

        return entry

    def _summarize(self, results: List[dict]) -> Dict[str, Any]:
        total = len(results)
        passed = sum(1 for r in results if r.get("passed"))
        exact = sum(1 for r in results if r["metrics"].get("exact_match"))
        scores = [r["metrics"].get("structure_score", 0.0) for r in results]
        avg_score = sum(scores) / total if total else 0.0

        exec_vals = [
            r["metrics"].get("execution_match")
            for r in results
            if "execution_match" in r["metrics"]
        ]
        exec_acc = sum(1 for v in exec_vals if v) / len(exec_vals) if exec_vals else None

        comp_avgs: Dict[str, Optional[float]] = {}
        for key in ["select_jaccard", "condition_recall"]:
            vals = [
                r["metrics"].get(key)
                for r in results
                if isinstance(r["metrics"].get(key), (int, float))
            ]
            comp_avgs[key] = sum(vals) / len(vals) if vals else None

        err_dist = Counter()
        for r in results:
            err = r.get("error")
            if err:
                err_dist[err.get("error_type", "unknown")] += 1

        return {
            "total": total,
            "passed": passed,
            "pass_rate": passed / total if total else 0.0,
            "exact_match": exact,
            "exact_match_rate": exact / total if total else 0.0,
            "avg_structure_score": round(avg_score, 4),
            "execution_accuracy": round(exec_acc, 4) if exec_acc is not None else None,
            "avg_select_jaccard": round(comp_avgs["select_jaccard"], 4)
            if comp_avgs["select_jaccard"] is not None
            else None,
            "avg_condition_recall": round(comp_avgs["condition_recall"], 4)
            if comp_avgs["condition_recall"] is not None
            else None,
            "error_distribution": dict(err_dist),
        }

    def render(self, result: Dict[str, Any]) -> str:
        cases = result["cases"]
        s = result["summary"]
        lines = [
            "=" * 72,
            "SQL 生成评估报告",
            "=" * 72,
            f"总用例数: {s['total']}",
            f"通过数: {s['passed']}",
            f"通过率: {s['pass_rate'] * 100:.2f}%",
            f"精确匹配率: {s['exact_match_rate'] * 100:.2f}%",
            f"平均结构得分: {s['avg_structure_score']:.4f}",
            f"平均 SELECT 相似度: {s['avg_select_jaccard'] if s['avg_select_jaccard'] is not None else 'N/A'}",
            f"平均条件覆盖率: {s['avg_condition_recall'] if s['avg_condition_recall'] is not None else 'N/A'}",
            f"执行准确率: {s['execution_accuracy'] if s['execution_accuracy'] is not None else 'N/A (未启用数据库执行)'}",
        ]

        if s["error_distribution"]:
            lines.append("错误类型分布:")
            for err_type, count in sorted(
                s["error_distribution"].items(), key=lambda x: -x[1]
            ):
                lines.append(f"  - {err_type}: {count}")
        else:
            lines.append("错误类型分布: 无")

        lines.append("")
        lines.append("用例明细:")
        for r in cases:
            m = r["metrics"]
            err = r.get("error")
            err_type = err.get("error_type", "-") if err else "-"
            lines.append(
                f"[{r['id']}] 通过={'Y' if r['passed'] else 'N'} "
                f"精确={'Y' if m.get('exact_match') else 'N'} "
                f"结构={m.get('structure_score', 0.0):.2f} "
                f"错误={err_type}"
            )
            lines.append(f"    问题: {r['question']}")
        return "\n".join(lines)