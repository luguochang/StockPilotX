from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from backend.app.prompt.evaluator import PromptRegressionRunner, default_prompt_generate_fn


def main() -> None:
    runner = PromptRegressionRunner()
    result = runner.run(default_prompt_generate_fn)
    out_dir = Path("docs/v1/prompt-evals")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    json_path = out_dir / f"prompt-regression-{ts}.json"
    md_path = out_dir / f"prompt-regression-{ts}.md"
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "result": result,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Prompt Regression Report",
        "",
        f"- generated_at: `{payload['generated_at']}`",
        f"- prompt_total_pass_rate: `{result.get('prompt_total_pass_rate', 0)}`",
        f"- prompt_redteam_pass_rate: `{result.get('prompt_redteam_pass_rate', 0)}`",
        f"- prompt_freshness_timestamp_rate: `{result.get('prompt_freshness_timestamp_rate', 0)}`",
        f"- prompt_failed_case_count: `{result.get('prompt_failed_case_count', 0)}`",
        f"- prompt_pass_gate: `{result.get('prompt_pass_gate', False)}`",
        "",
        "## Failed Cases",
        "",
        f"- ids: `{','.join(result.get('prompt_failed_case_ids', []))}`",
        "",
        "## Group Stats",
        "",
        "| group | total | passed | pass_rate |",
        "| --- | ---: | ---: | ---: |",
    ]
    for group, stat in dict(result.get("prompt_group_stats", {})).items():
        lines.append(
            f"| {group} | {int(stat.get('total', 0))} | {int(stat.get('passed', 0))} | {float(stat.get('pass_rate', 0.0)):.4f} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(str(json_path))
    print(str(md_path))


if __name__ == "__main__":
    main()
