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
    md = plan["market_data"]
    current_date = plan["current_date"]
    inp = plan["input"]
    val = md["valuation"]
    fin = md["financials"]
    analyst = md["analyst"]

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
    md = plan["market_data"]
    current_date = plan["current_date"]
    inp = plan["input"]
    tech = md["technical"]

    # Build price table (last 14 rows for display)
    history = md.get("price_history", [])[-14:]
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
    md = plan["market_data"]
    current_date = plan["current_date"]
    inp = plan["input"]
    tech = md["technical"]
    analyst = md["analyst"]
    ns = plan["narrative_signals"]

    # Format weekly news for prompt (show relevance tag for the model)
    weekly_news = md.get("weekly_news", [])
    if weekly_news:
        news_lines = "\n".join(
            f"- [{n['published_at']}] [{n.get('relevance','?')}] {n['publisher']}: {n['title']}"
            for n in weekly_news
        )
    else:
        news_lines = "（本周暂无抓取到新闻，请基于技术和情绪数据进行分析）"

    from datetime import date, timedelta
    week_start = (date.fromisoformat(current_date) - timedelta(days=7)).isoformat()

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
        sentiment_score=ns.get("sentiment_score", "N/A"),
        sentiment_label=ns.get("sentiment_label", "N/A"),
        recommendation=analyst.get("recommendation", "N/A"),
        upside_pct=analyst.get("upside_pct", "N/A"),
        short_interest=md.get("short_interest", "N/A"),
        institutional_pct=md.get("institutional_ownership_pct", "N/A"),
        beta=md.get("beta", "N/A"),
        key_themes="; ".join(ns.get("key_themes", [])),
        risk_level=ns.get("risk_level", "N/A"),
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
    md = plan["market_data"]
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


def generate_narration(client: OpenAI, plan: dict, synthesis: str) -> str:
    print("[Generator] Generating narration script...")
    md = plan["market_data"]
    inp = plan["input"]
    outline = plan["slide_outline"]
    current_date = plan["current_date"]

    chars_per_minute = 350 if inp["language"].startswith("zh") else 140
    word_count = chars_per_minute * inp["duration_minutes"]
    word_count_max = int(word_count * 1.15)
    slide_types = ", ".join(s["type"] for s in outline)

    prompt_tpl = _load_prompt("narration")
    user_msg = prompt_tpl.format(
        current_date=current_date,
        ticker=md["ticker"],
        duration_minutes=inp["duration_minutes"],
        word_count=word_count,
        word_count_max=word_count_max,
        language=inp["language"],
        style=inp["style"],
        slide_types=slide_types,
        analysis=synthesis,
    )

    system = (
        f"你是中文YouTube顶级财经主播，今天是{current_date}。"
        "以强Hook、高retention、节奏紧凑著称。"
        "稿子适配ElevenLabs TTS朗读，句子短促有力，数字用中文读法。"
        "禁止引用训练数据中的具体历史日期，所有日期信息来自提供的分析内容。"
    )
    result = _stream_call(client, system, user_msg)
    print("[Generator] Narration complete.")
    return result


def generate_slides(client: OpenAI, plan: dict, synthesis: str) -> dict:
    print("[Generator] Generating slides JSON...")
    md = plan["market_data"]
    inp = plan["input"]
    outline = plan["slide_outline"]
    current_date = plan["current_date"]

    slide_outline_str = "\n".join(
        f"- {s['type']}（约{s['approx_seconds']}秒）" for s in outline
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
        "你是专业的演示文稿设计师，今天是{current_date}。"
        "请严格以合法的JSON格式返回结果，包含 title、ticker、slides 三个字段。"
        "slides 必须是一个数组（array），每个元素包含 type 和 headline 字段。"
        "示例：{{\"title\": \"...\", \"ticker\": \"...\", \"slides\": [{{\"type\": \"cover\", \"headline\": \"...\"}}]}}"
    ).format(current_date=current_date)

    raw = _chat_call(client, system, user_msg, json_mode=True)
    slides_data = json.loads(raw)
    slides_data = _normalize_slides(slides_data)
    print(f"[Generator] Slides complete: {len(slides_data['slides'])} slides.")
    return slides_data


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
