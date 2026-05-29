"""
Stage 2: Generator — Multi-Agent Pipeline
4 specialized agents: Fundamental | Technical | Narrative | Synthesis
Then: Narration + Slides JSON generation.
"""
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
MODEL_REASONER = "deepseek-reasoner"   # DeepSeek-R1: deep reasoning for analysis
MODEL_CHAT = "deepseek-chat"           # DeepSeek-V3: fast, for JSON + synthesis

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


# ── Helpers ──────────────────────────────────────────────────────────────────

def _load_prompt(name: str) -> str:
    return (PROMPTS_DIR / f"{name}.txt").read_text(encoding="utf-8")


def _stream_call(client: OpenAI, system: str, user: str, model: str = MODEL_REASONER) -> str:
    stream = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=8000,
        stream=True,
    )
    parts = []
    for chunk in stream:
        delta = chunk.choices[0].delta if chunk.choices else None
        if delta and delta.content:
            parts.append(delta.content)
    return "".join(parts)


def _chat_call(client: OpenAI, system: str, user: str, json_mode: bool = False, model: str = MODEL_CHAT) -> str:
    kwargs = dict(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=8000,
    )
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content


# ── Agent 1: Fundamental ──────────────────────────────────────────────────────

def _agent_fundamental(client: OpenAI, plan: dict) -> str:
    print("[Agent:Fundamental] Analyzing...")
    md = plan["market_snapshot"]
    current_date = plan["current_date"]
    inp = plan["input"]
    val = plan["valuation_snapshot"]
    fin = plan["financial_snapshot"]
    analyst = plan["analyst_snapshot"]

    prompt = _load_prompt("fundamental").format(
        current_date=current_date,
        ticker=md["ticker"],
        company_name=md["company_name"],
        sector=md.get("sector", ""),
        industry=md.get("industry", ""),
        language=inp["language"],
        current_price=md.get("current_price", "N/A"),
        market_cap_str=md.get("market_cap_str", "N/A"),
        beta=md.get("beta", "N/A"),
        pe_trailing=val.get("pe_trailing", "N/A"),
        pe_forward=val.get("pe_forward", "N/A"),
        peg=val.get("peg", "N/A"),
        ps_ratio=val.get("ps_ratio", "N/A"),
        pb_ratio=val.get("pb_ratio", "N/A"),
        ev_ebitda=val.get("ev_ebitda", "N/A"),
        revenue_growth=fin.get("revenue_growth_yoy", "N/A"),
        earnings_growth=fin.get("earnings_growth_yoy", "N/A"),
        gross_margin=fin.get("gross_margin", "N/A"),
        operating_margin=fin.get("operating_margin", "N/A"),
        profit_margin=fin.get("profit_margin", "N/A"),
        roe=fin.get("roe", "N/A"),
        eps_trailing=fin.get("eps_trailing", "N/A"),
        eps_forward=fin.get("eps_forward", "N/A"),
        short_description=md.get("short_description", ""),
        recommendation=analyst.get("recommendation", "N/A"),
        target_mean=analyst.get("target_mean", "N/A"),
        target_low=analyst.get("target_low", "N/A"),
        target_high=analyst.get("target_high", "N/A"),
        analyst_count=analyst.get("analyst_count", "N/A"),
        upside_pct=analyst.get("upside_pct", "N/A"),
    )

    system = f"你是专业的基本面分析师。今天是{current_date}，分析必须基于提供的实时数据，禁止引用训练数据中的具体历史日期。"
    result = _stream_call(client, system, prompt)
    print("[Agent:Fundamental] Done.")
    return result


# ── Agent 2: Technical ────────────────────────────────────────────────────────

def _agent_technical(client: OpenAI, plan: dict) -> str:
    print("[Agent:Technical] Analyzing...")
    md = plan["market_snapshot"]
    current_date = plan["current_date"]
    inp = plan["input"]
    tech = plan["technical_indicators"]

    # Build price table (last 14 rows for display)
    history = plan["price_history"]["items"][-14:]
    price_table = "\n".join(f"{p['date']}: ${p['close']}" for p in history)

    bb = tech.get("bollinger") or {}

    prompt = _load_prompt("technical").format(
        current_date=current_date,
        ticker=md["ticker"],
        company_name=md["company_name"],
        language=inp["language"],
        current_price=md.get("current_price", "N/A"),
        prev_close=md.get("prev_close", "N/A"),
        price_change_pct=md.get("price_change_pct", "N/A"),
        high_52w=md.get("52w_high", "N/A"),
        low_52w=md.get("52w_low", "N/A"),
        range_position=md.get("52w_range_position_pct", "N/A"),
        rsi_14=tech.get("rsi_14", "N/A"),
        rsi_signal=tech.get("rsi_signal", "N/A"),
        ma20=tech.get("ma20", "N/A"),
        ma50=tech.get("ma50", "N/A"),
        price_vs_ma20=tech.get("price_vs_ma20_pct", "N/A"),
        price_vs_ma50=tech.get("price_vs_ma50_pct", "N/A"),
        bb_upper=bb.get("upper", "N/A"),
        bb_middle=bb.get("middle", "N/A"),
        bb_lower=bb.get("lower", "N/A"),
        ma_signal=tech.get("ma_signal", "N/A"),
        volume_trend=tech.get("volume_trend", "N/A"),
        beta=md.get("beta", "N/A"),
        price_table=price_table,
    )

    system = f"你是专业的技术分析师。今天是{current_date}，分析必须基于提供的数据，禁止引用训练数据中的具体历史日期。"
    result = _stream_call(client, system, prompt, model=MODEL_CHAT)
    print("[Agent:Technical] Done.")
    return result


# ── Agent 3: Narrative & Sentiment ───────────────────────────────────────────

def _agent_narrative(client: OpenAI, plan: dict) -> str:
    print("[Agent:Narrative] Analyzing...")
    md = plan["market_snapshot"]
    current_date = plan["current_date"]
    inp = plan["input"]
    tech = plan["technical_indicators"]
    analyst = plan["analyst_snapshot"]
    ns = plan["computed_signals"]

    # Format news evidence pack for prompt — include category and impact hint
    news_items = plan["news_evidence_pack"].get("items", [])
    if news_items:
        news_lines = "\n".join(
            f"- [{n['published_at']}] [{n.get('category', 'other')}] {n['publisher']}: {n['title']}"
            + (f"\n  → {n['why_relevant']}" if n.get("why_relevant") else "")
            for n in news_items
        )
    else:
        news_lines = "（本周暂无抓取到新闻，请基于技术和情绪数据进行分析）"

    from datetime import date, timedelta
    week_start = (date.fromisoformat(current_date) - timedelta(days=7)).isoformat()

    _momentum_zh = {"strong": "强势", "neutral": "中性", "weak": "弱势"}
    prompt = _load_prompt("narrative").format(
        current_date=current_date,
        week_start=week_start,
        ticker=md["ticker"],
        company_name=md["company_name"],
        sector=md.get("sector", ""),
        industry=md.get("industry", ""),
        language=inp["language"],
        current_price=md.get("current_price", "N/A"),
        price_change_pct=md.get("price_change_pct", "N/A"),
        range_position=md.get("52w_range_position_pct", "N/A"),
        rsi_14=tech.get("rsi_14", "N/A"),
        rsi_signal=tech.get("rsi_signal", "N/A"),
        volume_trend=tech.get("volume_trend", "N/A"),
        sentiment_score=ns.get("technical_sentiment_score", "N/A"),
        sentiment_label=_momentum_zh.get(ns.get("momentum_state", ""), "中性"),
        recommendation=analyst.get("recommendation", "N/A"),
        upside_pct=analyst.get("upside_pct", "N/A"),
        short_interest=md.get("short_interest", "N/A"),
        institutional_pct=md.get("institutional_ownership_pct", "N/A"),
        beta=md.get("beta", "N/A"),
        key_themes="; ".join(ns.get("signal_basis", [])),
        risk_level="; ".join(ns.get("risk_flags", [])) or "N/A",
        short_description=md.get("short_description", ""),
        weekly_news=news_lines,
    )

    system = f"你是专业的市场叙事与情绪分析师，专注周度市场分析。今天是{current_date}，分析必须基于提供的数据（包括本周新闻标题），禁止引用训练数据中的具体历史日期或事件。"
    result = _stream_call(client, system, prompt)
    print("[Agent:Narrative] Done.")
    return result


# ── Agent 4: Synthesis ────────────────────────────────────────────────────────

def _agent_synthesis(client: OpenAI, plan: dict, fundamental: str, technical: str, narrative: str) -> str:
    print("[Agent:Synthesis] Synthesizing all dimensions...")
    md = plan["market_snapshot"]
    current_date = plan["current_date"]
    inp = plan["input"]

    prompt = _load_prompt("synthesis").format(
        current_date=current_date,
        ticker=md["ticker"],
        company_name=md["company_name"],
        language=inp["language"],
        fundamental_analysis=fundamental,
        technical_analysis=technical,
        narrative_analysis=narrative,
    )

    system = f"你是首席投资分析师，今天是{current_date}。请将多维度分析整合为一份权威、简洁、可操作的综合报告。"
    result = _chat_call(client, system, prompt, model=MODEL_CHAT)
    print("[Agent:Synthesis] Done.")
    return result


# ── Narration & Slides ────────────────────────────────────────────────────────

def _normalize_slides(slides_data: dict) -> dict:
    slides = slides_data.get("slides")
    if isinstance(slides, list):
        return slides_data
    if isinstance(slides, dict):
        normalized = [
            {"type": k, **(v if isinstance(v, dict) else {"headline": str(v)})}
            for k, v in slides.items()
        ]
        slides_data["slides"] = normalized
    return slides_data


# Per-slide narration instructions (extracted from global prompt, per slide type)
_SLIDE_NARRATION_INSTRUCTIONS: dict[str, str] = {
    "cover": (
        "写30秒的狠Hook（约150字）：\n"
        "- 以反常识陈述句或悬念问句开场，禁止以'大家好''欢迎收看'开场\n"
        "- 引发恐惧/好奇/认知冲突，体现本周时效性\n"
        "- 结尾固定接：'今天我就带你把这周发生的事情掰开揉碎说清楚。'\n"
        "- Hook结束后，再用2-3句点出股价和核心矛盾"
    ),
    "market_overview": (
        "- 从宏观或行业背景切入，1-2句定性\n"
        "- 用具体数字说话（价格、涨跌幅、市值、PE），不要形容词堆砌\n"
        "- 引导观众注意最关键的1个数据异常点\n"
        "- 用类比让数字有感觉（如：这市值，相当于整个X行业的Y%）"
    ),
    "price_action": (
        "- 引导观众'看图'，描述近期走势特征和形态\n"
        "- 明确说出支撑位和阻力位的具体价格（不要模糊表达）\n"
        "- 给出一个明确的技术判断：偏多/偏空/震荡\n"
        "- 说明触发反转需要满足的具体条件"
    ),
    "key_points": (
        "这是全片核心段，必须写最多字：\n"
        "- 每个要点单独成段，用'第一点''第二点'引出\n"
        "- 每个要点：给结论 → 给具体数据 → 用类比或场景 → 说投资含义\n"
        "- 每个要点结尾用一句金句式总结（要有记忆点）\n"
        "- 各要点之间有逻辑关联，不要简单罗列"
    ),
    "financials": (
        "- 用比喻让财务数字有画面感（如：这毛利率，比卖茅台还高）\n"
        "- 触及盈利质量、增长可持续性、现金流三个维度\n"
        "- 用对比说明（如：同期行业平均是X，它是Y）\n"
        "- 结尾说明财务数据对估值的含义"
    ),
    "risk": (
        "- 用'但是'或'然而'硬转折引入风险主题\n"
        "- 每个风险说清楚：是什么 → 为什么有这个风险 → 影响有多大\n"
        "- 不要只列标题，要说影响机制和传导路径\n"
        "- 结尾必须是：'知道了风险，才能管好仓位。'（不要劝退）"
    ),
    "catalyst": (
        "- 说明具体时间节点（下周/本季度末/财报前后）\n"
        "- 每个催化剂说明潜在影响量级（小/中/大）及理由\n"
        "- 区分正向和负向催化剂，各自展开说"
    ),
    "outlook": (
        "- 给出明确的短期区间（1-4周的价格区间上下限）\n"
        "- 给出中期目标价（3-6个月）和触发条件\n"
        "- 区分多头和空头各自的入场/出场逻辑\n"
        "- 结尾一句话定性：看多/中性/谨慎"
    ),
    "summary": (
        "- 一句话核心结论，要有记忆点，像金句\n"
        "- 提醒仓位管理（1句，具体说多少仓位合适）\n"
        "- 引导点赞订阅不超过2句，要自然不硬推\n"
        "- 最后一句必须是：'投资有风险，以上内容仅供参考，不构成任何投资建议。'"
    ),
}


_POSITION_RULE_FIRST = (
    "这是视频的第一段（cover/hook）。\n"
    "必须以反常识陈述句或悬念问句开场，制造张力。\n"
    "可以引入股票代码和核心矛盾。\n"
    "结尾用：'今天我就带你把这周发生的事情掰开揉碎说清楚。'"
)

_POSITION_RULE_MIDDLE = (
    "这是视频的中间段，前面已经介绍过股票和日期。\n"
    "绝对禁止：\n"
    "  × '大家好' '朋友们' '欢迎收看' 等问候语\n"
    "  × 重新介绍股票代码、公司名称、今天日期\n"
    "  × 任何'开场白'式的引入句\n"
    "直接承接上一段节奏展开本段内容，用一句承上启下的话作为段首。"
)

_POSITION_RULE_LAST = (
    "这是视频的最后一段（summary）。\n"
    "绝对禁止重复问候语和股票介绍。\n"
    "直接收束全片，给出核心结论，引导点赞订阅，结尾必须有免责声明。"
)


def _generate_slide_narration(
    client: OpenAI,
    plan: dict,
    synthesis: str,
    slide: dict,
    target_chars: int,
    slide_index: int,
    total_slides: int,
) -> str:
    """Generate narration for a single slide with position-aware context."""
    md = plan["market_snapshot"]
    current_date = plan["current_date"]
    inp = plan["input"]

    slide_type = slide["type"]
    slide_goal = slide.get("goal", f"展开讲解 {slide_type} 相关内容")
    min_chars = int(target_chars * 0.85)
    max_chars = int(target_chars * 1.15)
    instructions = _SLIDE_NARRATION_INSTRUCTIONS.get(slide_type, "展开说明本幻灯片的内容，要说透而不是点到。")

    if slide_index == 0:
        position_rule = _POSITION_RULE_FIRST
    elif slide_index == total_slides - 1:
        position_rule = _POSITION_RULE_LAST
    else:
        position_rule = _POSITION_RULE_MIDDLE

    prompt_tpl = _load_prompt("narration_slide")
    user_msg = prompt_tpl.format(
        current_date=current_date,
        ticker=md["ticker"],
        slide_type=slide_type,
        slide_goal=slide_goal,
        target_chars=target_chars,
        min_chars=min_chars,
        max_chars=max_chars,
        language=inp["language"],
        analysis=synthesis,
        slide_instructions=instructions,
        slide_index=slide_index + 1,
        total_slides=total_slides,
        position_rule=position_rule,
    )

    system = (
        f"你是中文YouTube顶级财经主播，今天是{current_date}。"
        "以强Hook、高retention著称。稿子适配ElevenLabs TTS，"
        "句子短促有力，数字用中文读法。"
        "禁止引用训练数据中的具体历史日期。"
    )

    if slide_type in ("cover", "key_points", "risk"):
        return _stream_call(client, system, user_msg, model=MODEL_REASONER)
    return _chat_call(client, system, user_msg, model=MODEL_CHAT)


def _smooth_narration(client: OpenAI, plan: dict, narration: str) -> str:
    """
    One editor pass: fix cross-segment greeting repetition and improve flow.
    Uses fast MODEL_CHAT; input/output keeps [幻灯片: xxx] markers intact.
    """
    md = plan["market_snapshot"]
    current_date = plan["current_date"]

    system = (
        "你是专业的视频脚本编辑，负责将分段口播稿整合成一篇连贯的视频稿件。"
        "只修改影响连贯性的部分，不改变实质分析内容和数据。"
    )
    ticker = md["ticker"]
    company = md.get("company_name", "")
    segment_count = narration.count("[幻灯片:")
    user_msg = f"""以下是{ticker}（{company}）股票点评视频的分段口播稿，今天是{current_date}，总共{segment_count}段。

存在的问题需要修复：
1. 非首段（非cover）出现了问候语（大家好、朋友们等）→ 直接删除
2. 中间段重新介绍了日期、股票名称等已说过的内容 → 删除重复部分
3. 部分段落之间缺乏衔接感 → 加一句承上启下的过渡

修改要求：
- 保留所有 [幻灯片: xxx] 标注，位置不变
- 保持每段字数大致不变（允许±5%）
- 不改变任何价格、数据、分析结论
- 直接输出修改后的完整稿件，不加任何说明

原始稿件：
{narration}"""

    print("[Generator] Editor pass: smoothing cross-segment continuity...")
    result = _chat_call(client, system, user_msg, model=MODEL_CHAT)
    print("[Generator] Editor pass done.")
    return result


def generate_narration(client: OpenAI, plan: dict, synthesis: str) -> str:
    print("[Generator] Generating narration per slide (parallel)...")
    inp = plan["input"]
    outline = plan["slide_outline"]
    current_date = plan["current_date"]

    chars_per_minute = 350 if inp["language"].startswith("zh") else 140
    total_chars = chars_per_minute * inp["duration_minutes"]
    total_seconds = sum(s.get("approx_seconds", 60) for s in outline)

    def _target(slide: dict) -> int:
        secs = slide.get("approx_seconds", 60)
        return max(150, int(total_chars * secs / total_seconds))

    slide_results: dict[str, str] = {}
    total_slides = len(outline)

    def _gen(args: tuple) -> tuple[str, str]:
        idx, slide = args
        chars = _target(slide)
        text = _generate_slide_narration(client, plan, synthesis, slide, chars, idx, total_slides)
        print(f"[Narration:{slide['type']}] {len(text)}字 (目标{chars}字)")
        return slide["type"], text

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(_gen, (i, s)): s for i, s in enumerate(outline)}
        for future in as_completed(futures):
            stype, text = future.result()
            slide_results[stype] = text

    # Reassemble in outline order
    parts = [
        f"[幻灯片: {s['type']}]\n{slide_results.get(s['type'], '')}"
        for s in outline
    ]
    raw = "\n\n".join(parts)

    # Editor pass: fix greeting repetition and improve cross-segment flow
    full = _smooth_narration(client, plan, raw)

    total = len(full)
    print(f"[Generator] Narration complete. {total}字 ≈ {total/chars_per_minute:.1f}分钟")
    return full


def generate_slides(client: OpenAI, plan: dict, synthesis: str) -> dict:
    print("[Generator] Generating slides JSON...")
    md = plan["market_snapshot"]
    inp = plan["input"]
    outline = plan["slide_outline"]
    current_date = plan["current_date"]

    slide_outline_str = "\n".join(
        f"- {s['type']}（约{s['approx_seconds']}秒）：{s.get('goal', '')}"
        for s in outline
    )

    prompt_tpl = _load_prompt("slides")
    user_msg = prompt_tpl.format(
        current_date=current_date,
        ticker=md["ticker"],
        company_name=md["company_name"],
        language=inp["language"],
        slide_outline=slide_outline_str,
        analysis=synthesis,
    )

    system = (
        f"你是专业的财经演示文稿设计师，今天是{current_date}。"
        "严格按照 prompt 中定义的 JSON Schema 生成每张幻灯片，"
        "key_points 必须用 points 数组，financials 必须用 metrics 数组，"
        "risk 必须用 risks 数组，catalyst 必须用 catalysts 数组，"
        "outlook 必须包含 base_range / scenario_bull / scenario_bear。"
        "返回合法 JSON，根字段为 title、ticker、slides（array）。"
    )

    raw = _chat_call(client, system, user_msg, json_mode=True)
    slides_data = json.loads(raw)
    slides_data = _normalize_slides(slides_data)
    print(f"[Generator] Slides complete: {len(slides_data['slides'])} slides.")
    return slides_data


# ── Targeted Re-generation (called by run.py retry loop) ─────────────────────

def _make_client() -> OpenAI:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise EnvironmentError("DEEPSEEK_API_KEY environment variable not set.")
    return OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)


def regenerate_synthesis(plan: dict, gen_result: dict) -> str:
    """Re-run only the Synthesis agent using existing per-dimension analyses."""
    print("[Retry] Regenerating synthesis...")
    client = _make_client()
    result = _agent_synthesis(
        client, plan,
        gen_result["fundamental_analysis"],
        gen_result["technical_analysis"],
        gen_result["narrative_analysis"],
    )
    print("[Retry] Synthesis done.")
    return result


def regenerate_narration(plan: dict, gen_result: dict) -> str:
    """Re-run only the narration generator using the current synthesis."""
    print("[Retry] Regenerating narration...")
    client = _make_client()
    result = generate_narration(client, plan, gen_result["analysis"])
    print("[Retry] Narration done.")
    return result


def regenerate_slides(plan: dict, gen_result: dict) -> dict:
    """Re-run only the slides generator using the current synthesis."""
    print("[Retry] Regenerating slides...")
    client = _make_client()
    result = generate_slides(client, plan, gen_result["analysis"])
    print("[Retry] Slides done.")
    return result


# ── Main Entry ────────────────────────────────────────────────────────────────

def run(plan: dict) -> dict:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise EnvironmentError("DEEPSEEK_API_KEY environment variable not set.")

    client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)
    current_date = plan["current_date"]

    print(f"\n[Generator] Running 3 analysis agents in parallel (date={current_date})...")

    # Run Fundamental, Technical, Narrative agents concurrently
    agents = {
        "fundamental": lambda: _agent_fundamental(client, plan),
        "technical":   lambda: _agent_technical(client, plan),
        "narrative":   lambda: _agent_narrative(client, plan),
    }

    results = {}
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(fn): name for name, fn in agents.items()}
        for future in as_completed(futures):
            name = futures[future]
            results[name] = future.result()

    # Synthesize
    synthesis = _agent_synthesis(
        client, plan,
        results["fundamental"],
        results["technical"],
        results["narrative"],
    )

    # Narration and slides (sequential, depend on synthesis)
    narration = generate_narration(client, plan, synthesis)
    slides = generate_slides(client, plan, synthesis)

    return {
        "fundamental_analysis": results["fundamental"],
        "technical_analysis": results["technical"],
        "narrative_analysis": results["narrative"],
        "analysis": synthesis,          # synthesis = master analysis for downstream
        "narration": narration,
        "slides": slides,
    }
