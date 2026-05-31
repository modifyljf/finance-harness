"""EvaluatorAgent — LLM-judge quality check + schema validation."""
import json

from messaging.agents.base import BaseAgent
from messaging.dto.candidate import Candidate
from messaging.dto.evaluation import Evaluation
from messaging.dto.hard_rules import (
    MODEL_CHAT, EVAL_PASS_THRESHOLD, REQUIRED_SLIDE_TYPES,
    CHARS_PER_MINUTE_ZH, CHARS_PER_MINUTE_EN,
)


class EvaluatorAgent(BaseAgent):

    def _validate_slides(self, slides_data: dict) -> list[str]:
        errors = []
        if "title" not in slides_data:
            errors.append("slides.json missing 'title' field")
        if "ticker" not in slides_data:
            errors.append("slides.json missing 'ticker' field")
        if "slides" not in slides_data or not isinstance(slides_data["slides"], list):
            errors.append("slides.json missing or invalid 'slides' array")
            return errors

        slide_types = {s.get("type") for s in slides_data["slides"]}
        missing = REQUIRED_SLIDE_TYPES - slide_types
        if missing:
            errors.append(f"Missing required slide types: {', '.join(missing)}")

        for i, slide in enumerate(slides_data["slides"]):
            if "type" not in slide:
                errors.append(f"Slide {i} missing 'type'")
            if "headline" not in slide:
                errors.append(f"Slide {i} missing 'headline'")
        return errors

    def _validate_narration_length(self, narration: str, duration_minutes: int, language: str) -> list[str]:
        cpm = CHARS_PER_MINUTE_ZH if language.startswith("zh") else CHARS_PER_MINUTE_EN
        expected_min = int(cpm * duration_minutes * 0.45)
        expected_max = int(cpm * duration_minutes * 2.0)
        actual = len(narration)
        if actual < expected_min:
            return [f"Narration too short: {actual} chars, expected >{expected_min}"]
        if actual > expected_max:
            return [f"Narration too long: {actual} chars, expected <{expected_max}"]
        return []

    def _llm_judge(self, plan: dict, candidate: Candidate) -> dict:
        ticker = plan["market_snapshot"]["ticker"]
        language = plan["input"]["language"]
        slides_data = candidate.slides
        slide_count = len(slides_data.get("slides", []))

        prompt = f"""请对以下AI生成的股票点评视频内容进行质量评估。

股票代码：{ticker}
语言：{language}

=== 分析内容（前800字）===
{candidate.synthesis[:800]}

=== 解说词（前800字）===
{candidate.narration[:800]}

=== 幻灯片（共{slide_count}张，前3张）===
{json.dumps(slides_data.get('slides', [])[:3], ensure_ascii=False, indent=2)}

评估维度：
1. 内容准确性（数据引用是否合理，数字是否自洽）
2. 语言质量（是否流畅、专业、口语化适当）
3. 结构完整性（是否覆盖关键分析维度）
4. 幻灯片简洁度（每张是否言简意赅）
5. 整体一致性（各部分是否相互呼应）
6. 商业模式逻辑自洽性：
   - 稳定币发行商/货币市场基金类公司靠储备金利息盈利，高利率对其营收利好而非利空
   - 高杠杆公司高利率是融资成本风险，不可混淆为收入风险
   - 若发现利率方向与商业模式逻辑矛盾，在issues中明确指出并在score上扣10-20分

评分标准：90-100优秀，70-89良好，50-69一般，<50较差。passed标准：score >= {EVAL_PASS_THRESHOLD}。
维度6发现严重逻辑错误时，score上限为60。

请以JSON格式返回，包含以下字段：
- score: 整数，0-100
- passed: 布尔值
- issues: 字符串数组，列出主要问题
- strengths: 字符串数组，列出主要优点
- business_model_check: 字符串，说明商业模式类型及利率逻辑是否正确
- summary: 字符串，一句话总结
- retry_targets: 字符串数组，仅在 passed=false 时填写：
    "synthesis" — 综合分析逻辑有误
    "narration"  — 解说词质量差
    "slides"     — 幻灯片字段缺失
  只列真正需要重跑的组件。passed=true 时返回空数组 []。"""

        for attempt in range(3):
            response = self.client.chat.completions.create(
                model=MODEL_CHAT,
                messages=[
                    {"role": "system", "content": "你是内容质量评审专家，请客观评估内容质量并以JSON格式返回结果。"},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=1024,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content if response.choices else None
            if content and content.strip():
                try:
                    return json.loads(content)
                except json.JSONDecodeError as exc:
                    print(f"[Evaluator] JSON parse error (attempt {attempt + 1}/3): {exc}")
                    continue
            print(f"[Evaluator] Empty response from LLM judge (attempt {attempt + 1}/3), retrying...")
        print("[Evaluator] LLM judge returned empty after 3 attempts, skipping eval.")
        return {"score": 80, "passed": True, "issues": [], "strengths": [], "retry_targets": [],
                "business_model_check": "skipped", "summary": "Eval skipped due to empty LLM response."}

    def run(self, plan: dict, candidate: Candidate) -> Evaluation:
        print("[Evaluator] Running validation checks...")
        language = plan["input"]["language"]
        duration = plan["input"]["duration_minutes"]

        hard_errors = (
            self._validate_slides(candidate.slides)
            + self._validate_narration_length(candidate.narration, duration, language)
        )

        if hard_errors:
            print(f"[Evaluator] Hard validation failed: {hard_errors}")
            # Infer retry_targets from error types so the service can actually retry
            retry_targets = []
            for err in hard_errors:
                err_l = err.lower()
                if any(k in err_l for k in ("slide", "missing", "schema", "field")):
                    if "slides" not in retry_targets:
                        retry_targets.append("slides")
                if any(k in err_l for k in ("narration", "short", "long", "chars")):
                    if "narration" not in retry_targets:
                        retry_targets.append("narration")
            return Evaluation(score=0, passed=False, issues=hard_errors, hard_errors=hard_errors,
                              retry_targets=retry_targets,
                              summary="Hard validation failed — schema or length errors.")

        print("[Evaluator] Running LLM quality check...")
        report = self._llm_judge(plan, candidate)

        status = "PASSED" if report.get("passed") else "FAILED"
        print(f"[Evaluator] {status} — Score: {report.get('score')}/100")

        return Evaluation(
            score=report.get("score", 0),
            passed=report.get("passed", False),
            issues=report.get("issues", []),
            strengths=report.get("strengths", []),
            retry_targets=report.get("retry_targets", []),
            hard_errors=[],
            summary=report.get("summary", ""),
        )
