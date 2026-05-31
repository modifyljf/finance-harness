"""
GeneratorAgent — Multi-Agent Pipeline.
4 analysis agents (Fundamental | Technical | Narrative | Synthesis) → Narration + Slides.
"""
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta

from messaging.agents.base import BaseAgent
from messaging.dto.candidate import Candidate
from messaging.dto.hard_rules import (
    MODEL_REASONER, MODEL_CHAT,
    CHARS_PER_MINUTE_ZH, CHARS_PER_MINUTE_EN, MAX_NARRATION_MINUTES,
)


# ── Per-slide narration instructions ─────────────────────────────────────────

_SLIDE_INSTRUCTIONS: dict[str, str] = {
    "cover": (
        "写30秒的狠Hook（约150字）：\n"
        "- 以反常识陈述句或悬念问句开场，禁止以'大家好''欢迎收看'开场\n"
        "- 引发恐惧/好奇/认知冲突，体现本周时效性\n"
        "- 结尾固定接：'这周到底发生了什么？我们一条一条拆。'\n"
        "- Hook结束后，再用2-3句点出股价和核心矛盾"
    ),
    "market_overview": (
        "- 从宏观或行业背景切入，1-2句定性\n"
        "- 用具体数字说话（价格、涨跌幅、市值、PE），不要形容词堆砌\n"
        "- 引导观众注意最关键的1个数据异常点\n"
        "- 用类比让数字有感觉"
    ),
    "price_action": (
        "- 引导观众'看图'，描述近期走势特征和形态\n"
        "- 明确说出支撑位和阻力位的具体价格\n"
        "- 给出一个明确的技术判断：偏多/偏空/震荡\n"
        "- 说明触发反转需要满足的具体条件"
    ),
    "key_points": (
        "这是全片核心段，必须写最多字：\n"
        "- 每个要点单独成段，用'第一点''第二点'引出\n"
        "- 每个要点：给结论 → 给具体数据 → 用类比或场景 → 说投资含义\n"
        "- 每个要点结尾用一句金句式总结\n"
        "- 如有分析师评级变化，必须引用机构名称和具体观点"
    ),
    "news": (
        "- 逐条播报本周最重要的3-5条新闻，每条附上一句影响解读\n"
        "- 如有分析师评级变化，必须说出机构名称、评级方向和目标价（如有）\n"
        "- 多家机构观点分歧时，客观呈现，不强行统一结论\n"
        "- 结尾用一句话说明：本周新闻整体对股价是偏多、偏空还是中性信号"
    ),
    "financials": (
        "- 用比喻让财务数字有画面感\n"
        "- 触及盈利质量、增长可持续性、现金流三个维度\n"
        "- 用对比说明（如：同期行业平均是X，它是Y）\n"
        "- 结尾说明财务数据对估值的含义"
    ),
    "risk": (
        "- 用'但是'或'然而'硬转折引入风险主题\n"
        "- 每个风险说清楚：是什么 → 为什么 → 影响有多大\n"
        "- 不要只列标题，要说影响机制和传导路径\n"
        "- 结尾必须是：'知道了风险，才能管好仓位。'"
    ),
    "catalyst": (
        "- 说明具体时间节点（下周/本季度末/财报前后）\n"
        "- 每个催化剂说明潜在影响量级（小/中/大）及理由\n"
        "- 区分正向和负向催化剂，各自展开说\n"
        "- 如有机构升降评级，结合催化剂方向说明"
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

_POS_FIRST = (
    "这是视频的第一段（cover/hook）。\n"
    "必须以反常识陈述句或悬念问句开场，制造张力。\n"
    "结尾用：'这周到底发生了什么？我们一条一条拆。'"
)
_POS_MIDDLE = (
    "这是视频的中间段，前面已经介绍过股票和日期。\n"
    "绝对禁止：'大家好' '朋友们' '欢迎收看' 等问候语；重新介绍股票代码、公司名称、今天日期。\n"
    "直接承接上一段节奏展开本段内容，用一句承上启下的话作为段首。"
)
_POS_LAST = (
    "这是视频的最后一段（summary）。\n"
    "绝对禁止重复问候语和股票介绍。\n"
    "直接收束全片，给出核心结论，引导点赞订阅，结尾必须有免责声明。"
)


def _make_tts_narration(narration: str) -> str:
    text = re.sub(r'\[幻灯片: \w+\]\n?', '', narration)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _format_expert_quotes(expert_quotes: list[dict]) -> str:
    if not expert_quotes:
        return "（本周暂无分析师评级变化）"
    lines = []
    for q in expert_quotes:
        stance_zh = {"bullish": "看多", "bearish": "看空", "neutral": "中性"}.get(q.get("stance", ""), "中性")
        lines.append(
            f"- [{q.get('date', '')}] [{stance_zh}] {q.get('quote', '')} — {q.get('context', '')}"
        )
    return "\n".join(lines)


def _format_news(news_items: list[dict]) -> str:
    if not news_items:
        return "（本周暂无抓取到新闻，请基于技术和情绪数据进行分析）"
    lines = []
    for n in news_items:
        line = f"- [{n['published_at']}] [{n.get('category', 'other')}] {n['publisher']}: {n['title']}"
        if n.get("why_relevant"):
            line += f"\n  → {n['why_relevant']}"
        lines.append(line)
    return "\n".join(lines)


# ── Agent 1: Fundamental ──────────────────────────────────────────────────────

class GeneratorAgent(BaseAgent):

    @staticmethod
    def _format_research_pack(rp: dict) -> str:
        if not rp:
            return "（本次运行未执行深度研究）"
        lines = []
        segments = rp.get("product_segments", [])
        if segments:
            lines.append("【产品/业务线收入拆分】")
            for s in segments:
                lines.append(f"  {s.get('name','')}: {s.get('revenue','')}  增速={s.get('growth_pct','')}%  ({s.get('period','')})")
        guidance = rp.get("management_guidance", [])
        if guidance:
            lines.append("【管理层前瞻指引】")
            for g in guidance:
                lines.append(f"  {g.get('metric','')}: {g.get('value','')} [{g.get('direction','')}] — {g.get('context','')}")
        cp = rp.get("competitive_position", {})
        if cp:
            lines.append("【竞争格局】")
            lines.append(f"  市场份额: {cp.get('market_share','N/A')}")
            lines.append(f"  核心壁垒: {cp.get('moat','N/A')}")
            lines.append(f"  主要威胁: {cp.get('threats','N/A')}")
        thesis = rp.get("analyst_thesis", {})
        if thesis:
            lines.append("【分析师核心论点】")
            lines.append(f"  看多: {thesis.get('bull_case','')}")
            lines.append(f"  看空: {thesis.get('bear_case','')}")
        dq = rp.get("data_quality", {})
        if dq:
            lines.append(f"【数据质量】置信度={dq.get('confidence','N/A')} | {dq.get('notes','')}")
        return "\n".join(lines) if lines else "（深度研究未返回有效数据）"

    def _fundamental(self, plan: dict) -> str:
        print("[Agent:Fundamental] Analyzing...")
        md = plan["market_snapshot"]
        val = plan["valuation_snapshot"]
        fin = plan["financial_snapshot"]
        analyst = plan["analyst_snapshot"]
        inp = plan["input"]
        current_date = plan["current_date"]
        research_summary = self._format_research_pack(plan.get("research_pack", {}))

        prompt = self._load_prompt("fundamental").format(
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
            research_summary=research_summary,
        )
        system = f"你是专业的基本面分析师。今天是{current_date}，分析必须基于提供的实时数据，禁止引用训练数据中的具体历史日期。"
        result = self._stream(system, prompt)
        print("[Agent:Fundamental] Done.")
        return result

    def _technical(self, plan: dict) -> str:
        print("[Agent:Technical] Analyzing...")
        md = plan["market_snapshot"]
        tech = plan["technical_indicators"]
        inp = plan["input"]
        current_date = plan["current_date"]

        history = plan["price_history"]["items"][-14:]
        price_table = "\n".join(f"{p['date']}: ${p['close']}" for p in history)
        bb = tech.get("bollinger") or {}

        prompt = self._load_prompt("technical").format(
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
        result = self._stream(system, prompt, model=MODEL_CHAT)
        print("[Agent:Technical] Done.")
        return result

    def _narrative(self, plan: dict) -> str:
        print("[Agent:Narrative] Analyzing...")
        md = plan["market_snapshot"]
        tech = plan["technical_indicators"]
        analyst = plan["analyst_snapshot"]
        ns = plan["computed_signals"]
        inp = plan["input"]
        current_date = plan["current_date"]
        news_pack = plan["news_evidence_pack"]

        news_lines = _format_news(news_pack.get("items", []))
        expert_quotes_str = _format_expert_quotes(news_pack.get("expert_quotes", []))

        week_start = (date.fromisoformat(current_date) - timedelta(days=7)).isoformat()
        _momentum_zh = {"strong": "强势", "neutral": "中性", "weak": "弱势"}

        prompt = self._load_prompt("narrative").format(
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
            expert_quotes=expert_quotes_str,
        )
        system = (
            f"你是专业的市场叙事与情绪分析师，专注周度市场分析。今天是{current_date}，"
            "分析必须基于提供的数据（包括本周新闻标题和分析师观点），禁止引用训练数据中的具体历史日期或事件。"
        )
        result = self._stream(system, prompt)
        print("[Agent:Narrative] Done.")
        return result

    @staticmethod
    def _format_internal_valuation(iv: dict) -> str:
        if not iv:
            return "（本次运行未生成内部估值模型）"
        sc = iv.get("scenarios", {})
        bear = sc.get("bear", {})
        base = sc.get("base", {})
        bull = sc.get("bull", {})
        rm = iv.get("revenue_model", {})
        lines = [
            f"业务模式：{iv.get('business_model_type', 'N/A')}",
            f"估值方法：{iv.get('valuation_methodology', 'N/A')}",
            f"方法依据：{iv.get('methodology_rationale', '')}",
            "",
            "收入模型（3年）：",
        ]
        for key in ("y1", "y2", "y3"):
            yr = rm.get(key, {})
            if yr:
                lines.append(f"  {yr.get('period','')}: 营收 ${yr.get('revenue_est','N/A')}  增速 {yr.get('growth_pct','N/A')}%")
        lines += [
            "",
            "三情景目标价：",
            f"  悲观 Bear: ${bear.get('target_price','N/A')}  ({bear.get('implied_upside_pct','N/A')}%)  — {bear.get('key_assumption','')}",
            f"  基准 Base: ${base.get('target_price','N/A')}  ({base.get('implied_upside_pct','N/A')}%)  — {base.get('key_assumption','')}",
            f"  乐观 Bull: ${bull.get('target_price','N/A')}  ({bull.get('implied_upside_pct','N/A')}%)  — {bull.get('key_assumption','')}",
            "",
            f"推导逻辑：{iv.get('methodology_note', '')}",
            f"与共识对比：{iv.get('consensus_vs_internal', '')}",
        ]
        return "\n".join(lines)

    def _synthesis(self, plan: dict, fundamental: str, technical: str, narrative: str) -> str:
        print("[Agent:Synthesis] Synthesizing all dimensions...")
        md = plan["market_snapshot"]
        inp = plan["input"]
        current_date = plan["current_date"]
        iv_summary = self._format_internal_valuation(plan.get("internal_valuation", {}))

        prompt = self._load_prompt("synthesis").format(
            current_date=current_date,
            ticker=md["ticker"],
            company_name=md["company_name"],
            language=inp["language"],
            fundamental_analysis=fundamental,
            technical_analysis=technical,
            narrative_analysis=narrative,
            internal_valuation_summary=iv_summary,
        )
        system = f"你是首席投资分析师，今天是{current_date}。请将多维度分析整合为一份权威、简洁、可操作的综合报告。"
        result = self._chat(system, prompt, model=MODEL_CHAT)
        print("[Agent:Synthesis] Done.")
        return result

    # ── Narration ──────────────────────────────────────────────────────────────

    @staticmethod
    def _build_fact_anchor(plan: dict, slide_type: str) -> str:
        """
        For slides that make specific technical/price claims, inject the raw data
        as a hard constraint so the model cannot fabricate candle counts or prices.
        """
        if slide_type not in ("price_action", "key_points"):
            return ""

        md = plan["market_snapshot"]
        tech = plan["technical_indicators"]
        history = plan["price_history"]["items"]

        price_table = "\n".join(
            f"  {p['date']}: ${p['close']}" for p in history[-14:]
        )

        # Detect consecutive up/down days from actual data
        closes = [p["close"] for p in history]
        streak_desc = ""
        if len(closes) >= 2:
            direction = "up" if closes[-1] > closes[-2] else "down"
            streak = 1
            for i in range(len(closes) - 2, 0, -1):
                if direction == "up" and closes[i] > closes[i - 1]:
                    streak += 1
                elif direction == "down" and closes[i] < closes[i - 1]:
                    streak += 1
                else:
                    break
            streak_desc = f"最近连续{'上涨' if direction == 'up' else '下跌'} {streak} 天"

        bb = tech.get("bollinger") or {}
        lines = [
            "## 数据锚点（技术描述必须以此为准，禁止编造价格走势）",
            f"当前价格: ${md.get('current_price', 'N/A')}",
            f"MA20: ${tech.get('ma20', 'N/A')}  MA50: ${tech.get('ma50', 'N/A')}",
            f"RSI(14): {tech.get('rsi_14', 'N/A')} ({tech.get('rsi_signal', 'N/A')})",
            f"布林带: 上轨${bb.get('upper','N/A')} / 中轨${bb.get('middle','N/A')} / 下轨${bb.get('lower','N/A')}",
            f"成交量趋势: {tech.get('volume_trend', 'N/A')}",
            f"MA信号: {tech.get('ma_signal', 'N/A')}",
            f"52周高点: ${md.get('52w_high','N/A')}  52周低点: ${md.get('52w_low','N/A')}",
            f"区间位置: {md.get('52w_range_position_pct','N/A')}%",
            f"实际走势: {streak_desc}",
            "近14日收盘价（唯一可引用的价格事实）:",
            price_table,
            "约束：所有具体价格数字、均线数值、K线描述（如'连续X根阳线'）必须与上方数据一致，不得编造。\n",
        ]
        return "\n".join(lines) + "\n"

    def _slide_narration(
        self,
        plan: dict,
        synthesis: str,
        slide: dict,
        target_chars: int,
        slide_index: int,
        total_slides: int,
    ) -> str:
        md = plan["market_snapshot"]
        inp = plan["input"]
        current_date = plan["current_date"]
        slide_type = slide["type"]
        min_chars = int(target_chars * 0.90)
        max_chars = int(target_chars * 1.05)  # tight ceiling — model tends to overrun
        instructions = _SLIDE_INSTRUCTIONS.get(slide_type, "展开说明本幻灯片的内容，要说透而不是点到。")

        if slide_index == 0:
            pos_rule = _POS_FIRST
        elif slide_index == total_slides - 1:
            pos_rule = _POS_LAST
        else:
            pos_rule = _POS_MIDDLE

        fact_anchor = self._build_fact_anchor(plan, slide_type)

        user_msg = self._load_prompt("narration_slide").format(
            current_date=current_date,
            ticker=md["ticker"],
            slide_type=slide_type,
            slide_goal=slide.get("goal", f"展开讲解 {slide_type} 相关内容"),
            target_chars=target_chars,
            min_chars=min_chars,
            max_chars=max_chars,
            language=inp["language"],
            analysis=synthesis,
            slide_instructions=instructions,
            slide_index=slide_index + 1,
            total_slides=total_slides,
            position_rule=pos_rule,
            fact_anchor=fact_anchor,
        )
        system = (
            f"你是中文YouTube顶级财经主播，今天是{current_date}。"
            "以强Hook、高retention著称。稿子适配fish.audio TTS，句子短促有力，数字用中文读法。"
            "禁止引用训练数据中的具体历史日期。"
            f"【字数铁律】本段输出必须在{min_chars}到{max_chars}字之间，超出即视为失败，严禁超出上限。"
        )
        if slide_type in ("cover", "key_points", "risk"):
            return self._stream(system, user_msg, model=MODEL_REASONER)
        return self._chat(system, user_msg, model=MODEL_CHAT)

    def _smooth_narration(self, plan: dict, narration: str) -> str:
        md = plan["market_snapshot"]
        current_date = plan["current_date"]
        ticker = md["ticker"]
        company = md.get("company_name", "")
        segment_count = narration.count("[幻灯片:")

        system = (
            "你是专业的视频脚本编辑，负责将分段口播稿整合成一篇连贯的视频稿件。"
            "只修改影响连贯性的部分，不改变实质分析内容和数据。"
        )
        user_msg = (
            f"以下是{ticker}（{company}）股票点评视频的分段口播稿，今天是{current_date}，总共{segment_count}段。\n\n"
            "存在的问题需要修复：\n"
            "1. 非首段（非cover）出现了问候语（大家好、朋友们等）→ 直接删除\n"
            "2. 中间段重新介绍了日期、股票名称等已说过的内容 → 删除重复部分\n"
            "3. 部分段落之间缺乏衔接感 → 加一句承上启下的过渡\n\n"
            "修改要求：\n"
            "- 保留所有 [幻灯片: xxx] 标注，位置不变\n"
            "- 保持每段字数大致不变（允许±5%）\n"
            "- 不改变任何价格、数据、分析结论\n"
            "- 直接输出修改后的完整稿件，不加任何说明\n\n"
            f"原始稿件：\n{narration}"
        )
        print("[Generator] Editor pass: smoothing cross-segment continuity...")
        result = self._chat(system, user_msg, model=MODEL_CHAT)
        print("[Generator] Editor pass done.")
        return result

    def generate_narration(self, plan: dict, synthesis: str) -> str:
        print("[Generator] Generating narration per slide (parallel)...")
        inp = plan["input"]
        outline = plan["slide_outline"]
        current_date = plan["current_date"]

        chars_per_minute = CHARS_PER_MINUTE_ZH if inp["language"].startswith("zh") else CHARS_PER_MINUTE_EN
        # Cap at MAX_NARRATION_MINUTES regardless of requested duration
        max_total = chars_per_minute * MAX_NARRATION_MINUTES
        target_total = chars_per_minute * inp["duration_minutes"]
        total_chars = min(target_total, max_total)
        total_seconds = sum(s.get("approx_seconds", 60) for s in outline)

        def _target(slide: dict) -> int:
            secs = slide.get("approx_seconds", 60)
            return max(80, int(total_chars * secs / total_seconds))

        slide_results: dict[str, str] = {}
        total_slides = len(outline)

        def _gen(args: tuple) -> tuple[str, str]:
            idx, slide = args
            chars = _target(slide)
            text = self._slide_narration(plan, synthesis, slide, chars, idx, total_slides)
            print(f"[Narration:{slide['type']}] {len(text)}字 (目标{chars}字)")
            return slide["type"], text

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(_gen, (i, s)): s for i, s in enumerate(outline)}
            for future in as_completed(futures):
                stype, text = future.result()
                slide_results[stype] = text

        parts = [
            f"[幻灯片: {s['type']}]\n{slide_results.get(s['type'], '')}"
            for s in outline
        ]
        raw = "\n\n".join(parts)
        full = self._smooth_narration(plan, raw)
        print(f"[Generator] Narration complete. {len(full)}字 ≈ {len(full)/chars_per_minute:.1f}分钟")
        return full

    # ── Slide schema snippets (one per type) ────────────────────────────────

    _SLIDE_SCHEMAS: dict[str, str] = {
        "cover": (
            '{"type":"cover",'
            '"headline":"吸引眼球的问句或悬念标题（15字以内，来自口播Hook）",'
            '"subtitle":"本周深度分析·{current_date}",'
            '"hook":"口播开场最核心的一句冲突或悬念（25字以内）"}'
        ),
        "market_overview": (
            '{"type":"market_overview",'
            '"headline":"本期关键数字摘要，必须含价格/PE/涨跌幅等具体数字（30字以内）",'
            '"callout":"口播中最值得关注的数据异常或信号，必须有具体数字（35字以内）"}'
        ),
        "price_action": (
            '{"type":"price_action",'
            '"headline":"一句话技术判断（来自口播结论）",'
            '"signal":"bullish或bearish或neutral",'
            '"bullets":['
            '"支撑位：$X（来自口播中提到的支撑价格）",'
            '"阻力位：$X（来自口播中提到的阻力价格）",'
            '"趋势描述（来自口播）",'
            '"RSI数值+含义（来自口播）"'
            ']}'
        ),
        "key_points": (
            '{"type":"key_points",'
            '"headline":"本期核心主题（8字以内）",'
            '"points":['
            '{"title":"要点标题（12字以内，来自口播）","body":"直接引用口播原句，保留具体数字和逻辑链（60字以内）","tag":"bull或bear或neutral"}'
            ']}'
        ),
        "news": (
            '{"type":"news",'
            '"headline":"本周市场焦点一句话（25字以内，来自口播）",'
            '"news_items":['
            '{"title":"口播中提到的新闻标题（30字以内）","category":"earnings或analyst或product或regulation或macro或other","impact":"positive或negative或neutral","note":"口播对该条新闻的影响解读原句（35字以内）"}'
            '],'
            '"analyst_summary":"口播中提到的分析师观点（35字以内，含机构名）"}'
        ),
        "financials": (
            '{"type":"financials",'
            '"headline":"口播中的财务健康度结论（20字以内）",'
            '"metrics":['
            '{"label":"营收增速","value":"+X%（来自口播）","trend":"up或down或flat"},'
            '{"label":"毛利率","value":"X%（来自口播）","trend":"up或down或flat"},'
            '{"label":"净利率","value":"X%（来自口播）","trend":"up或down或flat"},'
            '{"label":"Forward PE","value":"Xx（来自口播）","trend":"up或down或flat"},'
            '{"label":"自由现金流","value":"口播中的描述","trend":"up或down或flat"},'
            '{"label":"ROE","value":"X%（来自口播）","trend":"up或down或flat"}'
            '],'
            '"callout":"口播中最关键的财务风险或亮点原句，必须有具体数字（45字以内）"}'
        ),
        "risk": (
            '{"type":"risk",'
            '"headline":"口播中的风险等级判断（高/中/低）",'
            '"risks":['
            '{"title":"风险标题（12字以内，来自口播）","body":"直接引用口播中对该风险机制的说明原句（60字以内）","severity":"high或medium或low"}'
            ']}'
        ),
        "catalyst": (
            '{"type":"catalyst",'
            '"headline":"催化剂概览（20字以内，来自口播）",'
            '"catalysts":['
            '{"event":"事件名称（20字以内，来自口播）","direction":"positive或negative","timing":"口播中提到的具体时间节点","impact":"large或medium或small"}'
            ']}'
        ),
        "outlook": (
            '{"type":"outlook",'
            '"headline":"口播中的方向判断（看多/中性/看空+时间维度）",'
            '"subtitle":"看多或中性或看空",'
            '"base_range":{"low":"$X（来自口播）","high":"$X（来自口播）","timeframe":"1-4周"},'
            '"scenario_bear":{"condition":"口播中空头破位条件原句（30字以内）","target":"风险价位$X（来自口播内部估值悲观情景）"},'
            '"scenario_base":{"condition":"口播中基准假设原句（30字以内）","target":"基准目标价$X（来自口播内部估值基准情景）"},'
            '"scenario_bull":{"condition":"口播中多头触发条件原句（30字以内）","target":"目标价$X（来自口播内部估值乐观情景）"}}'
        ),
        "summary": (
            '{"type":"summary",'
            '"headline":"口播中的核心结论金句（25字以内）",'
            '"verdict":"bullish或bearish或neutral",'
            '"action":"口播中的仓位和等待条件建议（30字以内）",'
            '"subtitle":"投资有风险，以上内容仅供参考，不构成任何投资建议。"}'
        ),
    }

    @staticmethod
    def _parse_narration_segments(narration: str) -> dict[str, str]:
        """Split narration into {slide_type: segment_text} by slide markers."""
        parts = re.split(r'\[幻灯片: (\w+)\]', narration)
        segments: dict[str, str] = {}
        for i in range(1, len(parts), 2):
            stype = parts[i]
            content = parts[i + 1].strip() if i + 1 < len(parts) else ""
            segments[stype] = content
        return segments

    @staticmethod
    def _safe_parse_json(raw: str) -> dict:
        """Parse JSON, falling back to extracting the first {...} block if needed."""
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # Try to extract JSON object from markdown fences or partial output
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(raw[start:end + 1])
                except json.JSONDecodeError:
                    pass
            raise

    def _extract_one_slide(
        self,
        slide_type: str,
        segment: str,
        plan: dict,
        _attempt: int = 0,
    ) -> dict:
        """Extract structured slide content from a single narration segment."""
        md = plan["market_snapshot"]
        inp = plan["input"]
        current_date = plan["current_date"]
        schema = self._SLIDE_SCHEMAS.get(slide_type, '{"type":"' + slide_type + '","headline":""}')
        schema = schema.replace("{current_date}", current_date)

        user_msg = self._load_prompt("slide_extract").format(
            current_date=current_date,
            ticker=md["ticker"],
            company_name=md["company_name"],
            language=inp["language"],
            slide_type=slide_type,
            narration_segment=segment,
            schema=schema,
        )
        system = (
            f"你是财经演示文稿设计师，今天是{current_date}。"
            "只从口播稿中提取信息，不得编造任何数字或结论。"
            "返回合法 JSON 对象，不要任何额外文字。"
        )
        raw = self._chat(system, user_msg, json_mode=True)
        try:
            return self._safe_parse_json(raw)
        except json.JSONDecodeError as exc:
            if _attempt < 2:
                return self._extract_one_slide(slide_type, segment, plan, _attempt + 1)
            print(f"[Generator] Slide '{slide_type}' parse failed: {exc}, using fallback.")
            return {"type": slide_type, "headline": segment[:40]}

    def generate_slides(self, plan: dict, narration: str) -> dict:
        """Generate slides by extracting content from each narration segment in parallel."""
        print("[Generator] Generating slides from narration (parallel extraction)...")
        md = plan["market_snapshot"]
        outline = plan["slide_outline"]
        current_date = plan["current_date"]

        segments = self._parse_narration_segments(narration)

        def _extract(slide: dict) -> dict:
            stype = slide["type"]
            segment = segments.get(stype, "")
            if not segment:
                return {"type": stype, "headline": ""}
            result = self._extract_one_slide(stype, segment, plan)
            print(f"[Slide:{stype}] extracted OK")
            return result

        slide_objects: list[dict] = [None] * len(outline)
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(_extract, s): i for i, s in enumerate(outline)}
            for future in as_completed(futures):
                idx = futures[future]
                slide_objects[idx] = future.result()

        title_slide = next((s for s in slide_objects if s.get("type") == "cover"), {})
        headline = title_slide.get("headline", md["ticker"])

        slides_data = {
            "title": headline,
            "ticker": md["ticker"],
            "slides": slide_objects,
        }
        print(f"[Generator] Slides complete: {len(slide_objects)} slides.")
        return slides_data

    @staticmethod
    def _normalize_slides(slides_data: dict) -> dict:
        slides = slides_data.get("slides")
        if isinstance(slides, list):
            return slides_data
        if isinstance(slides, dict):
            slides_data["slides"] = [
                {"type": k, **(v if isinstance(v, dict) else {"headline": str(v)})}
                for k, v in slides.items()
            ]
        return slides_data

    def generate_youtube_meta(self, plan: dict, synthesis: str) -> dict:
        print("[Generator] Generating YouTube metadata...")
        md = plan["market_snapshot"]
        inp = plan["input"]
        outline = plan["slide_outline"]
        current_date = plan["current_date"]
        news_pack = plan["news_evidence_pack"]

        slide_outline_str = " → ".join(s["type"] for s in outline)
        analysis_summary = synthesis[:600].rsplit("。", 1)[0] + "。"

        # Ground the title generator with actual this-week news to prevent hallucination
        news_lines = "\n".join(
            f"- [{n['published_at']}] {n['title']}"
            for n in news_pack.get("items", [])[:8]
        )

        user_msg = self._load_prompt("youtube").format(
            current_date=current_date,
            ticker=md["ticker"],
            company_name=md.get("company_name", md["ticker"]),
            duration_minutes=inp["duration_minutes"],
            analysis_summary=analysis_summary,
            slide_outline=slide_outline_str,
            weekly_news=news_lines or "（暂无新闻数据）",
        )
        system = (
            f"你是YouTube财经频道运营专家，今天是{current_date}。"
            "标题内容必须严格基于提供的【分析摘要】和【本周新闻】，禁止虚构事件结果。"
            "严格按照JSON格式返回，字段：title、description、tags（数组）。"
        )
        raw = self._chat(system, user_msg, json_mode=True, model=MODEL_CHAT)
        meta = json.loads(raw)
        print(f"[Generator] YouTube metadata done. Title: {meta.get('title', '')[:60]}")

        # Second dedicated call: thumbnail copy from the title
        title = meta.get("title", md["ticker"])
        thumb_sys = "你是YouTube封面图文案专家。严格以JSON格式返回，不要任何额外文字。"
        thumb_user = (
            f"将以下YouTube标题拆解为封面图文字：\n标题：{title}\n\n"
            "规则：\n"
            "- hook：主标题，12字以内，省略股票代码，保留最强冲突/悬念\n"
            "- hook_highlight：hook中最关键的2-4个字（会用高亮色标注），必须是hook的子字符串\n"
            "- subhook：副标题，18字以内，口语化补充\n"
            "- verdict：根据标题语气判断，只能填 bullish 或 bearish 或 neutral\n\n"
            '返回格式：{"hook":"...","hook_highlight":"...","subhook":"...","verdict":"..."}'
        )
        try:
            thumb_raw = self._chat(thumb_sys, thumb_user, json_mode=True, model=MODEL_CHAT)
            thumb = json.loads(thumb_raw)
            meta["thumbnail_hook"]      = thumb.get("hook", "")
            meta["thumbnail_highlight"] = thumb.get("hook_highlight", "")
            meta["thumbnail_subhook"]   = thumb.get("subhook", "")
            meta["thumbnail_verdict"]   = thumb.get("verdict", "neutral")
        except Exception as exc:
            print(f"[Generator] Thumbnail copy fallback: {exc}")
            meta.setdefault("thumbnail_hook", title.split("｜")[-1][:12] if "｜" in title else title[:12])
            meta.setdefault("thumbnail_highlight", "")
            meta.setdefault("thumbnail_subhook", "")
            meta.setdefault("thumbnail_verdict", "neutral")

        print(
            f"[Generator] Thumbnail hook: '{meta['thumbnail_hook']}' "
            f"| highlight: '{meta['thumbnail_highlight']}' "
            f"| verdict: {meta['thumbnail_verdict']}"
        )
        return meta

    # ── Entry points ───────────────────────────────────────────────────────────

    def run(self, plan: dict) -> Candidate:
        current_date = plan["current_date"]
        print(f"\n[Generator] Running 3 analysis agents in parallel (date={current_date})...")

        agents = {
            "fundamental": lambda: self._fundamental(plan),
            "technical":   lambda: self._technical(plan),
            "narrative":   lambda: self._narrative(plan),
        }
        results = {}
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {executor.submit(fn): name for name, fn in agents.items()}
            for future in as_completed(futures):
                results[futures[future]] = future.result()

        synthesis = self._synthesis(plan, results["fundamental"], results["technical"], results["narrative"])
        narration = self.generate_narration(plan, synthesis)
        slides = self.generate_slides(plan, narration)   # extracted from narration, not synthesis
        narration_tts = _make_tts_narration(narration)

        return Candidate(
            fundamental_analysis=results["fundamental"],
            technical_analysis=results["technical"],
            narrative_analysis=results["narrative"],
            synthesis=synthesis,
            narration=narration,
            narration_tts=narration_tts,
            slides=slides,
            attempt=0,
        )

    def regenerate(self, plan: dict, candidate: Candidate, targets: list[str]) -> Candidate:
        """Re-run only specified targets, reuse everything else."""
        fundamental = candidate.fundamental_analysis
        technical = candidate.technical_analysis
        narrative = candidate.narrative_analysis
        synthesis = candidate.synthesis
        narration = candidate.narration
        slides = candidate.slides

        if "synthesis" in targets:
            synthesis = self._synthesis(plan, fundamental, technical, narrative)
        if "narration" in targets:
            narration = self.generate_narration(plan, synthesis)
        if "slides" in targets:
            slides = self.generate_slides(plan, narration)

        narration_tts = _make_tts_narration(narration)
        return Candidate(
            fundamental_analysis=fundamental,
            technical_analysis=technical,
            narrative_analysis=narrative,
            synthesis=synthesis,
            narration=narration,
            narration_tts=narration_tts,
            slides=slides,
            attempt=candidate.attempt + 1,
        )
