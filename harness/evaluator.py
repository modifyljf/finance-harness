"""
Stage 3: Evaluator
Validates outputs and runs an LLM-judge quality check using DeepSeek.
"""
import json
import os

from openai import OpenAI

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
MODEL_FAST = "deepseek-v4-flash"  # fast and cheap for evaluation

REQUIRED_SLIDE_TYPES = {"cover", "market_overview", "price_action", "key_points", "outlook", "summary"}


def validate_slides_schema(slides_data: dict) -> list[str]:
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


def validate_narration_length(narration: str, duration_minutes: int, language: str) -> list[str]:
    errors = []
    chars_per_minute = 350 if language.startswith("zh") else 140
    expected_min = int(chars_per_minute * duration_minutes * 0.45)
    expected_max = int(chars_per_minute * duration_minutes * 2.0)
    actual = len(narration)

    if actual < expected_min:
        errors.append(f"Narration too short: {actual} chars, expected >{expected_min}")
    elif actual > expected_max:
        errors.append(f"Narration too long: {actual} chars, expected <{expected_max}")

    return errors


def llm_judge(client: OpenAI, plan: dict, analysis: str, narration: str, slides_data: dict) -> dict:
    ticker = plan["market_snapshot"]["ticker"]
    language = plan["input"]["language"]
    slide_count = len(slides_data.get("slides", []))

    prompt = f"""请对以下AI生成的股票点评视频内容进行质量评估。

股票代码：{ticker}
语言：{language}

=== 分析内容（前800字）===
{analysis[:800]}

=== 解说词（前800字）===
{narration[:800]}

=== 幻灯片（共{slide_count}张，前3张）===
{json.dumps(slides_data.get('slides', [])[:3], ensure_ascii=False, indent=2)}

评估维度：
1. 内容准确性（数据引用是否合理，数字是否自洽）
2. 语言质量（是否流畅、专业、口语化适当）
3. 结构完整性（是否覆盖关键分析维度）
4. 幻灯片简洁度（每张是否言简意赅）
5. 整体一致性（各部分是否相互呼应）
6. 商业模式逻辑自洽性（重点检查以下常见错误）：
   - 稳定币发行商/货币市场基金类公司（如Circle/USDC）靠储备金利息盈利，高利率对其营收利好而非利空，若分析写反须标记为严重错误
   - 高杠杆公司高利率是融资成本风险，不可混淆为收入风险
   - 若发现利率方向与商业模式逻辑矛盾，在issues中明确指出并在score上扣10-20分

评分标准：90-100优秀，70-89良好，50-69一般，<50较差。passed标准：score >= 70。
维度6发现严重逻辑错误时，score上限为60。

请以JSON格式返回，包含以下字段：
- score: 整数，0-100
- passed: 布尔值
- issues: 字符串数组，列出主要问题（维度6的错误须单独列出）
- strengths: 字符串数组，列出主要优点
- business_model_check: 字符串，说明判断到的公司商业模式类型及利率逻辑是否正确
- summary: 字符串，一句话总结
- retry_targets: 字符串数组，仅在 passed=false 时填写，列出需要重新生成的组件：
    "synthesis" — 综合分析逻辑有误、深度不足、商业模式判断错误
    "narration"  — 解说词质量差、语言生硬、前后矛盾、内容与分析不符
    "slides"     — 幻灯片字段缺失、内容与分析脱节、关键数据未体现
  注意：只列真正需要重跑的组件，不要全部列出。passed=true 时返回空数组 []。"""

    response = client.chat.completions.create(
        model=MODEL_FAST,
        messages=[
            {"role": "system", "content": "你是内容质量评审专家，请客观评估内容质量并以JSON格式返回结果。"},
            {"role": "user", "content": prompt},
        ],
        max_tokens=1024,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content
    return json.loads(raw)


def run(plan: dict, generation_result: dict) -> dict:
    print("[Evaluator] Running validation checks...")
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise EnvironmentError("DEEPSEEK_API_KEY environment variable not set.")

    client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)
    language = plan["input"]["language"]
    duration = plan["input"]["duration_minutes"]

    slides_errors = validate_slides_schema(generation_result["slides"])
    narration_errors = validate_narration_length(generation_result["narration"], duration, language)
    hard_errors = slides_errors + narration_errors

    if hard_errors:
        print(f"[Evaluator] Hard validation failed: {hard_errors}")
        return {
            "score": 0,
            "passed": False,
            "issues": hard_errors,
            "strengths": [],
            "summary": "Hard validation failed — schema or length errors.",
            "hard_errors": hard_errors,
        }

    print("[Evaluator] Running LLM quality check (deepseek-v4-flash)...")
    report = llm_judge(
        client,
        plan,
        generation_result["analysis"],
        generation_result["narration"],
        generation_result["slides"],
    )
    report["hard_errors"] = []

    status = "PASSED" if report["passed"] else "FAILED"
    print(f"[Evaluator] {status} — Score: {report['score']}/100")
    return report
