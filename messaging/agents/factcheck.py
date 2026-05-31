"""
FactCheckAgent — builds authoritative ground-truth numbers and scans analysis
texts for factual violations before the Generator runs.

Two outputs injected into plan:
  plan["authoritative_numbers"] — canonical number dict, injected as hard constraints
  plan["fact_check_report"]     — list of detected issues for audit
"""
import json

from messaging.agents.base import BaseAgent
from messaging.dto.hard_rules import MODEL_CHAT


# RSI thresholds — used for signal validation
_RSI_OVERBOUGHT = 70
_RSI_OVERSOLD   = 30


def _rsi_signal_label(rsi: float | None) -> str:
    if rsi is None:
        return "unknown"
    if rsi >= _RSI_OVERBOUGHT:
        return "overbought"
    if rsi <= _RSI_OVERSOLD:
        return "oversold"
    return "neutral"


def _fmt_b(val: float | None) -> str:
    """Format a raw dollar value to $XB string."""
    if val is None:
        return "N/A"
    if val >= 1e12:
        return f"${val/1e12:.2f}T"
    if val >= 1e9:
        return f"${val/1e9:.2f}B"
    return f"${val/1e6:.0f}M"


class FactCheckAgent(BaseAgent):

    # ── Build authoritative numbers ────────────────────────────────────────────

    def _build_authoritative_numbers(self, plan: dict) -> dict:
        """
        Build a ground-truth dict from verified plan data sources.
        All downstream agents (Generator prompts) must use these numbers exactly.
        """
        md   = plan["market_snapshot"]
        val  = plan["valuation_snapshot"]
        fin  = plan["financial_snapshot"]
        tech = plan["technical_indicators"]
        iv   = plan.get("internal_valuation", {})
        earn = plan.get("earnings_snapshot", {})

        current_price = md.get("current_price")
        market_cap    = md.get("market_cap")
        shares_out    = market_cap / current_price if (market_cap and current_price) else None
        total_rev     = fin.get("revenue") or earn.get("revenue")

        rsi = tech.get("rsi_14")
        rsi_label = _rsi_signal_label(rsi)

        bb = tech.get("bollinger") or {}

        # Valuation derivation from internal model
        iv_sc   = iv.get("scenarios", {})
        iv_base = iv_sc.get("base", {})
        iv_bear = iv_sc.get("bear", {})
        iv_bull = iv_sc.get("bull", {})
        ma      = iv.get("multiple_analysis", {})

        # Build derivation string so narration can say "15x PE × $15.48 = $232"
        fp  = val.get("pe_forward")
        eps = fin.get("eps_forward")
        if fp and eps and iv_base.get("target_price"):
            base_derivation = (
                f"{ma.get('fair_multiple', fp)}x {ma.get('multiple_type','Forward PE')} × "
                f"${eps} EPS = ${iv_base.get('target_price')}"
            )
        else:
            base_derivation = iv.get("methodology_note", "基于内部估值模型推导")

        return {
            # ── Price & market cap ──
            "ticker":              md["ticker"],
            "current_price":       current_price,
            "market_cap_raw":      market_cap,
            "market_cap_str":      md.get("market_cap_str", _fmt_b(market_cap)),
            "shares_outstanding_m": round(shares_out / 1e6, 0) if shares_out else None,

            # ── Revenue ceiling (any segment/product ARR claim must be < this) ──
            "total_annual_revenue_raw": total_rev,
            "total_annual_revenue_str": _fmt_b(total_rev),
            "total_annual_revenue_b":   round(total_rev / 1e9, 1) if total_rev else None,

            # ── Valuation multiples ──
            "forward_pe":          val.get("pe_forward"),
            "trailing_pe":         val.get("pe_trailing"),
            "forward_eps":         fin.get("eps_forward"),
            "peg":                 val.get("peg"),
            "enterprise_value":    val.get("enterprise_value"),
            "ev_revenue":          val.get("ev_revenue"),
            "ev_ebitda":           val.get("ev_ebitda"),
            "total_debt":          fin.get("total_debt"),
            "total_cash":          fin.get("total_cash"),

            # ── Technical indicators (authoritative) ──
            "rsi_14":              rsi,
            "rsi_signal":          rsi_label,   # "neutral" / "overbought" / "oversold"
            "rsi_overbought_threshold": _RSI_OVERBOUGHT,
            "rsi_oversold_threshold":   _RSI_OVERSOLD,
            "ma20":                tech.get("ma20"),
            "ma50":                tech.get("ma50"),
            "bb_upper":            bb.get("upper"),
            "bb_middle":           bb.get("middle"),
            "bb_lower":            bb.get("lower"),
            "52w_high":            md.get("52w_high"),
            "52w_low":             md.get("52w_low"),

            # ── Target prices (ONLY from internal_valuation, never from analyst consensus alone) ──
            "valuation_base_price":      iv_base.get("target_price"),
            "valuation_bear_price":      iv_bear.get("target_price"),
            "valuation_bull_price":      iv_bull.get("target_price"),
            "valuation_method":          iv.get("valuation_methodology", "N/A"),
            "valuation_base_derivation": base_derivation,
            "valuation_methodology_statement": iv.get("methodology_statement", ""),

            # ── Growth & margins ──
            "revenue_growth_yoy_pct":  fin.get("revenue_growth_yoy"),
            "earnings_growth_yoy_pct": fin.get("earnings_growth_yoy"),
            "gross_margin_pct":        fin.get("gross_margin"),
            "peg":                     val.get("peg"),

            # ── EPS surprise (most recent quarter) ──
            "last_eps_actual":        earn.get("eps_actual"),
            "last_eps_estimate":      earn.get("eps_estimate"),
            "last_eps_surprise_pct":  earn.get("eps_surprise_pct"),
            "next_earnings_date":     earn.get("next_earnings_date"),

            # ── Company type & metric constraints (from ValuationAgent) ──
            **self._build_metric_constraints(iv),
        }

    # ── Metric constraints by company type ────────────────────────────────────

    # primary metric, secondary metrics, prohibited as PRIMARY anchor
    _TYPE_METRICS = {
        "GrowthSoftware": {
            "primary":    "Forward PE + PEG + EV/FCF",
            "secondary":  "EV/Revenue, Rule of 40",
            "prohibited": "TTM PE（禁止用Trailing PE作主要锚点）",
        },
        "HyperGrowth": {
            "primary":    "EV/Revenue + Revenue Growth Rate + FCF Margin",
            "secondary":  "Rule of 40, NRR",
            "prohibited": "PE（PE对亏损或早期盈利公司无意义，禁止用PE作主要估值锚点）",
        },
        "Semiconductor": {
            "primary":    "Forward PE + EV/EBITDA + Revenue Growth",
            "secondary":  "Gross Margin, Book-to-Bill",
            "prohibited": "PS Ratio（半导体不看销售额倍数）",
        },
        "PlatformTech": {
            "primary":    "Forward PE + EV/EBITDA + FCF Yield",
            "secondary":  "Segment Revenue Mix",
            "prohibited": "PEG（平台科技增速不线性，PEG失真）",
        },
        "MatureCashCow": {
            "primary":    "Forward PE + FCF Yield + Dividend Yield",
            "secondary":  "EV/EBITDA",
            "prohibited": "EV/Revenue（成熟公司不看Revenue倍数）",
        },
        "Financial": {
            "primary":    "P/B + ROTCE + NIM",
            "secondary":  "Dividend Yield, Efficiency Ratio",
            "prohibited": "PE/EV/Revenue（银行用P/B，PE和EV指标均不适用）",
        },
        "Insurance": {
            "primary":    "P/B + Combined Ratio + ROE",
            "secondary":  "Dividend Yield",
            "prohibited": "PE/EV/Revenue（保险公司不看PE）",
        },
        "REIT": {
            "primary":    "P/FFO + Dividend Yield + NOI Growth",
            "secondary":  "Cap Rate, Occupancy Rate",
            "prohibited": "PE/EV（REIT用FFO而非EPS，PE无意义）",
        },
        "Energy": {
            "primary":    "EV/EBITDA + FCF Yield + Reserve Life",
            "secondary":  "EV/DACF",
            "prohibited": "PE（能源公司盈利周期性强，PE失真）",
        },
        "Biotech": {
            "primary":    "EV/Revenue + Pipeline NPV + Cash Runway",
            "secondary":  "Peak Sales Estimate",
            "prohibited": "PE（生物医药通常亏损，PE无意义）",
        },
        "Consumer": {
            "primary":    "Forward PE + EV/EBITDA + SSS Growth",
            "secondary":  "Gross Margin Trend",
            "prohibited": "EV/Revenue（消费品不看Revenue倍数）",
        },
    }

    def _build_metric_constraints(self, iv: dict) -> dict:
        company_type = iv.get("company_type", "")
        rules = self._TYPE_METRICS.get(company_type, {})
        return {
            "company_type":              company_type,
            "primary_valuation_metric":  rules.get("primary", "Forward PE"),
            "secondary_metrics":         rules.get("secondary", ""),
            "prohibited_metric_note":    rules.get("prohibited", ""),
        }

    # ── LLM scan for violations ────────────────────────────────────────────────

    def _llm_scan(self, plan: dict, auth: dict, texts: dict[str, str]) -> list[dict]:
        """
        Ask LLM to scan analysis texts for numbers that contradict authoritative data.
        Returns list of {field, found_value, correct_value, severity, location} dicts.
        """
        combined = "\n\n".join(
            f"=== {label} ===\n{text[:1200]}"
            for label, text in texts.items()
            if text and text.strip()
        )

        total_rev_b = auth.get("total_annual_revenue_b")
        rsi         = auth.get("rsi_14")
        rsi_signal  = auth.get("rsi_signal")
        mc_str      = auth.get("market_cap_str")

        prompt = f"""以下是由多个AI Agent生成的股票分析文本，请检查其中是否存在与权威数据相矛盾的错误。

## 权威数字（来自API，不可质疑）

- 股票代码：{auth['ticker']}
- 当前价格：${auth.get('current_price')}
- 真实市值：{mc_str}（注意：是十亿级别，不是万亿）
- 全年总营收：{auth.get('total_annual_revenue_str')}（任何产品线/业务线ARR不得超过此数）
- Forward PE：{auth.get('forward_pe')}x
- RSI(14)：{rsi}，正确信号：{rsi_signal}（超买阈值=70，超卖阈值=30，{rsi}属于neutral区间）
- 内部估值基准目标价：${auth.get('valuation_base_price')}（{auth.get('valuation_base_derivation')}）

## 需要检查的分析文本

{combined}

## 检查规则

1. 市值数字是否与真实市值一致（允许±5%误差）
2. 任何产品线/ARR数字是否超过全年总营收 → 若超过则必定是混淆了ARR/Bookings/Pipeline
3. RSI信号描述是否准确（RSI={rsi}应为neutral，不应出现"超买""严重超买"）
4. 目标价是否有明确推导依据，还是凭空出现
5. 是否存在自相矛盾（如同一段先说neutral后说severely overbought）

## 输出格式（JSON数组，只列真实存在的错误，没有错误则返回[]）

[
  {{
    "location": "哪个分析段落（fundamental/technical/narrative/synthesis）",
    "violation_type": "market_cap_error | revenue_ceiling_exceeded | rsi_signal_wrong | target_price_unsourced | self_contradiction | other",
    "found_text": "原文中出现的错误描述（引用原文，30字以内）",
    "correct_value": "正确的数字或描述",
    "severity": "critical | warning"
  }}
]"""

        system = f"你是数据一致性审计员，今天是{plan['current_date']}。只返回JSON数组，不要任何额外文字。"
        try:
            raw = self._chat(system, prompt, json_mode=True, model=MODEL_CHAT)
            result = json.loads(raw)
            if isinstance(result, list):
                return result
            if isinstance(result, dict) and "issues" in result:
                return result["issues"]
            return []
        except Exception as exc:
            print(f"[FactCheck] LLM scan failed (non-fatal): {exc}")
            return []

    # ── Deterministic checks ───────────────────────────────────────────────────

    def _deterministic_checks(self, auth: dict, plan: dict) -> list[dict]:
        """
        Mathematical consistency checks — no LLM needed.
        Validates key financial identity equations.
        """
        issues = []
        price    = auth.get("current_price")
        mc       = auth.get("market_cap_raw")
        shares_m = auth.get("shares_outstanding_m")
        fp_pe    = auth.get("forward_pe")
        fp_eps   = auth.get("forward_eps")
        val      = plan.get("valuation_snapshot", {})
        ps_ratio = val.get("ps_ratio")
        total_rev = auth.get("total_annual_revenue_raw")

        def _flag(vtype, found, correct, severity="warning"):
            issues.append({
                "location": "authoritative_numbers",
                "violation_type": vtype,
                "found_text": found,
                "correct_value": correct,
                "severity": severity,
            })

        # 1. MarketCap = Price × Shares Outstanding
        if price and mc and shares_m:
            implied_mc = price * shares_m * 1e6
            if abs(implied_mc - mc) / mc > 0.15:
                _flag("market_cap_mismatch",
                      f"Price(${price}) × Shares({shares_m}M) = {_fmt_b(implied_mc)} ≠ API MarketCap={_fmt_b(mc)}",
                      _fmt_b(mc))

        # 2. Forward PE = Price / Forward EPS (allow ±10%)
        if price and fp_pe and fp_eps and fp_eps > 0:
            implied_pe = price / fp_eps
            if abs(implied_pe - fp_pe) / fp_pe > 0.10:
                _flag("pe_eps_mismatch",
                      f"Price(${price}) / EPS(${fp_eps}) = {implied_pe:.1f}x ≠ ForwardPE={fp_pe}x",
                      f"{implied_pe:.2f}x")

        # 3. EV = MarketCap + Debt - Cash (allow ±10%)
        ev          = auth.get("enterprise_value")
        total_debt  = auth.get("total_debt")
        total_cash  = auth.get("total_cash")
        if mc and ev and total_debt is not None and total_cash is not None:
            implied_ev = mc + total_debt - total_cash
            if abs(implied_ev - ev) / ev > 0.10:
                _flag("ev_mismatch",
                      f"MC({_fmt_b(mc)})+Debt({_fmt_b(total_debt)})-Cash({_fmt_b(total_cash)})={_fmt_b(implied_ev)} ≠ API EV={_fmt_b(ev)}",
                      _fmt_b(ev))

        # 4. EV/Revenue cross-check (allow ±10%)
        ev_rev  = auth.get("ev_revenue")
        if ev and total_rev and ev_rev and total_rev > 0:
            implied_ev_rev = ev / total_rev
            if abs(implied_ev_rev - ev_rev) / ev_rev > 0.10:
                _flag("ev_revenue_mismatch",
                      f"EV({_fmt_b(ev)}) / Rev({_fmt_b(total_rev)}) = {implied_ev_rev:.1f}x ≠ API EV/Rev={ev_rev}x",
                      f"{implied_ev_rev:.2f}x")

        # 5. PEG sanity: Forward PE / PEG should give a reasonable growth rate (5-50%)
        # Note: API PEG uses long-term (5yr) growth estimate, not current-year EPS growth.
        # We only flag if PEG implies an implausible growth rate (< 1% or > 200%).
        peg = auth.get("peg")
        if fp_pe and peg and peg > 0:
            implied_growth = fp_pe / peg
            if implied_growth < 1 or implied_growth > 200:
                _flag("peg_implausible",
                      f"ForwardPE({fp_pe}) / PEG({peg}) implies {implied_growth:.0f}% long-term growth (unusual)",
                      f"Expected 1-200% range")

        # 4. Any segment revenue ceiling breach (from research_pack)
        research = plan.get("research_pack", {})
        for seg in research.get("product_segments", []):
            rev_str = seg.get("revenue", "")
            # Try to extract a dollar figure and compare vs total_rev
            # Simple heuristic: if the segment revenue string contains "B" and total_rev
            import re
            m = re.search(r'\$?([\d.]+)\s*[Bb]', rev_str)
            if m and total_rev:
                seg_b = float(m.group(1))
                total_b = total_rev / 1e9
                if seg_b > total_b * 1.05:
                    _flag("segment_revenue_ceiling",
                          f"Segment '{seg.get('name','')}' revenue {rev_str} > total annual revenue {_fmt_b(total_rev)}",
                          f"Must be < {_fmt_b(total_rev)}",
                          severity="critical")

        if issues:
            print(f"[FactCheck] Deterministic checks: {len(issues)} issue(s) found")
            for iss in issues:
                print(f"  [{iss['severity'].upper()}] {iss['violation_type']}: {iss['found_text']}")
        return issues

    # ── Expectation Gap Engine ─────────────────────────────────────────────────

    def _compute_expectation_gap(self, plan: dict) -> dict:
        """
        Compare actual results vs analyst consensus to surface the expectation gap.
        This is the core of 'Why Now' — where market assumptions diverge from reality.
        """
        earn      = plan.get("earnings_snapshot", {})
        estimates = plan.get("analyst_estimates", {})
        analyst   = plan.get("analyst_snapshot", {})
        md        = plan["market_snapshot"]

        # EPS: actual vs estimate (most recent quarter)
        eps_actual   = earn.get("eps_actual")
        eps_estimate = earn.get("eps_estimate")
        eps_surprise = earn.get("eps_surprise_pct")

        # Revenue: actual quarter vs analyst consensus for that quarter
        rev_estimates = estimates.get("revenue_estimates", [])
        current_q_est = next(
            (e for e in rev_estimates if str(e.get("period", "")).lower() in ("0q", "+0q")), None
        )
        actual_rev  = earn.get("revenue")
        est_rev     = current_q_est.get("avg_revenue") if current_q_est else None
        rev_surprise_pct = None
        if actual_rev and est_rev and est_rev > 0:
            rev_surprise_pct = round((actual_rev - est_rev) / est_rev * 100, 1)

        # Stock price vs analyst consensus target
        current_price  = md.get("current_price")
        target_mean    = analyst.get("target_mean")
        consensus_gap  = analyst.get("upside_pct")  # already computed in planner

        # Direction classification
        direction = "unknown"
        if eps_surprise is not None:
            if eps_surprise >= 10:
                direction = "strong_beat"
            elif eps_surprise >= 3:
                direction = "beat"
            elif eps_surprise <= -10:
                direction = "strong_miss"
            elif eps_surprise <= -3:
                direction = "miss"
            else:
                direction = "inline"

        # Narrative framing for synthesis
        gap_narrative = self._build_gap_narrative(
            eps_surprise, rev_surprise_pct, consensus_gap, direction, plan
        )

        result = {
            "eps_actual":         eps_actual,
            "eps_estimate":       eps_estimate,
            "eps_surprise_pct":   eps_surprise,
            "revenue_actual":     actual_rev,
            "revenue_estimate":   est_rev,
            "revenue_surprise_pct": rev_surprise_pct,
            "consensus_target":   target_mean,
            "current_price":      current_price,
            "consensus_gap_pct":  consensus_gap,
            "last_quarter":       earn.get("last_quarter"),
            "next_earnings_date": earn.get("next_earnings_date"),
            "direction":          direction,
            "gap_narrative":      gap_narrative,
        }

        beat_str = f"EPS超预期{eps_surprise:+.1f}%" if eps_surprise else "EPS数据缺失"
        rev_str  = f"营收超预期{rev_surprise_pct:+.1f}%" if rev_surprise_pct else ""
        print(f"[FactCheck] Expectation Gap: {direction} | {beat_str} | {rev_str} | "
              f"股价vs共识目标价={consensus_gap:+.1f}%" if consensus_gap else "")

        return result

    def _build_gap_narrative(
        self, eps_surprise, rev_surprise, consensus_gap, direction, plan: dict
    ) -> str:
        """Build a human-readable expectation gap summary for injection into synthesis."""
        md    = plan["market_snapshot"]
        earn  = plan.get("earnings_snapshot", {})
        lines = []

        if direction in ("strong_beat", "beat"):
            lines.append(
                f"【市场错误假设】市场预期公司增速放缓，"
                f"但最近一季度（{earn.get('last_quarter','')}）"
                f"EPS实际${earn.get('eps_actual')} vs 预期${earn.get('eps_estimate')}，"
                f"超预期{eps_surprise:+.1f}%。"
            )
            if rev_surprise and rev_surprise > 0:
                lines.append(f"营收同样超预期{rev_surprise:+.1f}%，基本面改善信号明确。")
        elif direction in ("strong_miss", "miss"):
            lines.append(
                f"【市场担忧验证】最近一季度（{earn.get('last_quarter','')}）"
                f"EPS实际${earn.get('eps_actual')} vs 预期${earn.get('eps_estimate')}，"
                f"低于预期{eps_surprise:+.1f}%，市场担忧部分落地。"
            )
        else:
            lines.append(
                f"【业绩符合预期】最近一季度（{earn.get('last_quarter','')}）"
                f"EPS实际${earn.get('eps_actual')} vs 预期${earn.get('eps_estimate')}，"
                f"基本符合市场预期。"
            )

        if consensus_gap is not None:
            if consensus_gap > 15:
                lines.append(
                    f"【价格错配】当前股价${md.get('current_price')}，"
                    f"分析师平均目标价${plan.get('analyst_snapshot',{}).get('target_mean')}，"
                    f"隐含上行空间{consensus_gap:+.1f}%——市场定价明显落后于机构共识。"
                )
            elif consensus_gap < -10:
                lines.append(
                    f"【价格透支】当前股价已高于分析师平均目标价{abs(consensus_gap):.1f}%，"
                    f"短期存在向下均值回归压力。"
                )

        if plan.get("earnings_snapshot", {}).get("next_earnings_date"):
            lines.append(
                f"【下一个验证节点】下次财报{earn.get('next_earnings_date')}，"
                f"将是预期差能否进一步扩大的关键时刻。"
            )

        return " ".join(lines)

    # ── WhyNow & Narrative Gap ─────────────────────────────────────────────────

    def _build_why_now_structured(self, plan: dict, eg: dict) -> dict:
        """
        Build structured WhyNow output from expectation gap + research pack.
        Returns {old_thesis, new_thesis, what_changed, why_now, direction, eps_surprise_pct}
        """
        research   = plan.get("research_pack", {})
        thesis     = research.get("analyst_thesis", {})
        guidance   = research.get("management_guidance", [])
        md         = plan["market_snapshot"]
        iv         = plan.get("internal_valuation", {})
        company    = md.get("company_name", md["ticker"])

        # old_thesis: what bears / skeptics currently believe
        old_thesis = (thesis.get("bear_case") or
                      f"市场认为{company}增长已放缓，当前估值已反映大部分利好。")

        # new_thesis: what the data actually shows
        new_thesis = (thesis.get("bull_case") or
                      f"基本面数据显示{company}实际表现优于市场预期，认知差正在形成。")

        # what_changed: concrete evidence from recent quarter
        eps_surprise = eg.get("eps_surprise_pct") or 0
        rev_surprise = eg.get("revenue_surprise_pct") or 0
        parts = []
        if abs(eps_surprise) >= 3:
            label = "超预期" if eps_surprise > 0 else "低于预期"
            parts.append(f"最近一季度（{eg.get('last_quarter','N/A')}）EPS{label}{abs(eps_surprise):.1f}%")
        if abs(rev_surprise) >= 3:
            label = "超预期" if rev_surprise > 0 else "低于预期"
            parts.append(f"营收{label}{abs(rev_surprise):.1f}%")
        if guidance:
            parts.append(f"管理层给出{len(guidance)}条前瞻指引，显示业务走向")
        company_type = iv.get("company_type", "")
        if company_type:
            parts.append(f"公司类型已确认为{company_type}，估值方法已切换至{iv.get('valuation_methodology','N/A')}")
        what_changed = "；".join(parts) if parts else "近期基本面出现新变化，市场认知尚未更新。"

        # why_now: timing rationale
        next_date  = eg.get("next_earnings_date", "")
        consensus_gap = eg.get("consensus_gap_pct")
        if next_date and consensus_gap is not None:
            if consensus_gap > 15:
                why_now = (f"股价仍低于分析师共识目标价{consensus_gap:.1f}%，"
                           f"下次财报（{next_date}）是预期差验证的关键节点。")
            elif consensus_gap < -10:
                why_now = (f"股价已超分析师目标价{abs(consensus_gap):.1f}%，"
                           f"下次财报（{next_date}）是检验当前溢价是否成立的时刻。")
            else:
                why_now = f"预期差正在形成，下次财报（{next_date}）将是方向选择时刻。"
        elif next_date:
            why_now = f"下次财报（{next_date}）是最近一个验证节点。"
        else:
            why_now = "预期差已经出现，市场认知更新尚未发生，是关注时机。"

        return {
            "old_thesis":       old_thesis,
            "new_thesis":       new_thesis,
            "what_changed":     what_changed,
            "why_now":          why_now,
            "direction":        eg.get("direction", "unknown"),
            "eps_surprise_pct": eps_surprise,
            "rev_surprise_pct": rev_surprise,
        }

    def _compute_narrative_gap(self, plan: dict, why_now: dict) -> dict:
        """
        LLM derives structured narrative gap: {market_consensus, reality, gap, implication}.
        Uses research pack + financial data + expectation gap as inputs.
        Non-fatal: returns empty dict on failure.
        """
        md       = plan["market_snapshot"]
        fin      = plan["financial_snapshot"]
        val      = plan["valuation_snapshot"]
        research = plan.get("research_pack", {})
        thesis   = research.get("analyst_thesis", {})
        iv       = plan.get("internal_valuation", {})
        eg       = plan.get("expectation_gap", {})
        current_date = plan["current_date"]

        prompt = self._load_prompt("narrative_gap").format(
            current_date=current_date,
            ticker=md["ticker"],
            company_name=md.get("company_name", md["ticker"]),
            sector=md.get("sector", "N/A"),
            current_price=md.get("current_price", "N/A"),
            market_cap_str=md.get("market_cap_str", "N/A"),
            forward_pe=val.get("pe_forward", "N/A"),
            revenue_growth=fin.get("revenue_growth_yoy", "N/A"),
            earnings_growth=fin.get("earnings_growth_yoy", "N/A"),
            gross_margin=fin.get("gross_margin", "N/A"),
            eps_surprise=eg.get("eps_surprise_pct", "N/A"),
            rev_surprise=eg.get("revenue_surprise_pct", "N/A"),
            direction=eg.get("direction", "unknown"),
            old_thesis=why_now.get("old_thesis", ""),
            new_thesis=why_now.get("new_thesis", ""),
            what_changed=why_now.get("what_changed", ""),
            bull_case=thesis.get("bull_case", "N/A"),
            bear_case=thesis.get("bear_case", "N/A"),
            company_type=iv.get("company_type", "N/A"),
            valuation_method=iv.get("valuation_methodology", "N/A"),
            base_target=iv.get("scenarios", {}).get("base", {}).get("target_price", "N/A"),
            expected_value=iv.get("expected_value", "N/A"),
            language=plan["input"]["language"],
        )

        system = (
            f"你是顶级股票研究分析师，今天是{current_date}。"
            "请基于提供的数据推导叙事缺口，输出严格JSON。"
            "不得编造数字，只基于提供的事实进行逻辑推理。"
        )

        try:
            raw = self._chat(system, prompt, json_mode=True, model=MODEL_CHAT)
            result = json.loads(raw)
            print(f"[FactCheck] NarrativeGap: consensus='{result.get('market_consensus','')[:40]}...' gap='{result.get('gap','')[:40]}...'")
            return result
        except Exception as exc:
            print(f"[FactCheck] NarrativeGap failed (non-fatal): {exc}")
            return {}

    # ── Main entry ─────────────────────────────────────────────────────────────

    def run(self, plan: dict) -> dict:
        print("[FactCheck] Building authoritative numbers and scanning for violations...")

        auth = self._build_authoritative_numbers(plan)
        expectation_gap = self._compute_expectation_gap(plan)
        why_now      = self._build_why_now_structured(plan, expectation_gap)
        narrative_gap = self._compute_narrative_gap(plan, why_now)

        # Collect all generated analysis texts
        texts = {
            "fundamental": plan.get("_fundamental_text", ""),
            "technical":   plan.get("_technical_text", ""),
            "narrative":   plan.get("_narrative_text", ""),
            "synthesis":   plan.get("_synthesis_text", ""),
        }
        # Filter out empty
        texts = {k: v for k, v in texts.items() if v and v.strip()}

        issues = self._deterministic_checks(auth, plan)
        if texts:
            llm_issues = self._llm_scan(plan, auth, texts)
            issues.extend(llm_issues)

        critical = [i for i in issues if i.get("severity") == "critical"]
        warnings = [i for i in issues if i.get("severity") == "warning"]

        print(
            f"[FactCheck] Done. Critical={len(critical)} | Warnings={len(warnings)} | "
            f"MarketCap={auth.get('market_cap_str')} | RSI={auth.get('rsi_14')}({auth.get('rsi_signal')}) | "
            f"RevCeiling={auth.get('total_annual_revenue_str')} | "
            f"Target Base=${auth.get('valuation_base_price')}"
        )

        return {
            "authoritative_numbers": auth,
            "expectation_gap":       expectation_gap,
            "why_now":          why_now,
            "narrative_gap":    narrative_gap,
            "issues": issues,
            "critical_count": len(critical),
            "warning_count":  len(warnings),
            "summary": f"{len(critical)} critical, {len(warnings)} warnings detected",
        }
