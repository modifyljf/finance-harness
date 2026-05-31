"""ValuationAgent — builds an internal valuation model for any company and sector."""
import json

from messaging.agents.base import BaseAgent
from messaging.dto.hard_rules import MODEL_REASONER


class ValuationAgent(BaseAgent):

    def run(self, plan: dict) -> dict:
        """
        Build an internal valuation model from plan data.
        Returns a structured dict injected into plan["internal_valuation"].
        Non-fatal: returns empty dict on any failure.
        """
        print("[Valuation] Building internal valuation model...")
        md       = plan["market_snapshot"]
        fin      = plan["financial_snapshot"]
        val      = plan["valuation_snapshot"]
        analyst  = plan["analyst_snapshot"]
        earnings = plan.get("earnings_snapshot", {})
        estimates = plan.get("analyst_estimates", {})
        current_date = plan["current_date"]
        inp      = plan["input"]

        def _fmt_estimates(items: list[dict], key: str) -> str:
            if not items:
                return "暂无数据"
            lines = []
            for e in items:
                period = e.get("period", "")
                val_   = e.get(key)
                growth = e.get("implied_yoy_growth_pct") or e.get("analyst_count")
                lines.append(f"  {period}: {val_}" + (f"  (YoY {growth}%)" if isinstance(growth, float) else ""))
            return "\n".join(lines)

        eps_lines = []
        for e in estimates.get("earnings_estimates", []):
            eps_lines.append(
                f"  {e['period']}: avg={e.get('avg_eps')}  low={e.get('low_eps')}  high={e.get('high_eps')}  "
                f"yago={e.get('year_ago_eps')}  analysts={e.get('analyst_count')}"
            )

        rev_lines = []
        for e in estimates.get("revenue_estimates", []):
            rev_lines.append(
                f"  {e['period']}: avg={e.get('avg_revenue')}  yago={e.get('year_ago_revenue')}  "
                f"YoY={e.get('implied_yoy_growth_pct')}%  analysts={e.get('analyst_count')}"
            )

        prompt = self._load_prompt("valuation").format(
            current_date=current_date,
            ticker=md["ticker"],
            company_name=md.get("company_name", md["ticker"]),
            sector=md.get("sector", "N/A"),
            industry=md.get("industry", "N/A"),
            language=inp["language"],
            current_price=md.get("current_price", "N/A"),
            market_cap_str=md.get("market_cap_str", "N/A"),
            pe_forward=val.get("pe_forward", "N/A"),
            pe_trailing=val.get("pe_trailing", "N/A"),
            peg=val.get("peg", "N/A"),
            ps_ratio=val.get("ps_ratio", "N/A"),
            pb_ratio=val.get("pb_ratio", "N/A"),
            ev_ebitda=val.get("ev_ebitda", "N/A"),
            revenue_growth_yoy=fin.get("revenue_growth_yoy", "N/A"),
            earnings_growth_yoy=fin.get("earnings_growth_yoy", "N/A"),
            gross_margin=fin.get("gross_margin", "N/A"),
            operating_margin=fin.get("operating_margin", "N/A"),
            profit_margin=fin.get("profit_margin", "N/A"),
            roe=fin.get("roe", "N/A"),
            eps_trailing=fin.get("eps_trailing", "N/A"),
            eps_forward=fin.get("eps_forward", "N/A"),
            free_cashflow=fin.get("free_cashflow", "N/A"),
            last_quarter=earnings.get("last_quarter", "N/A"),
            q_revenue=earnings.get("revenue", "N/A"),
            q_revenue_qoq=earnings.get("revenue_qoq_pct", "N/A"),
            q_gross_margin=earnings.get("gross_margin_pct", "N/A"),
            q_net_income=earnings.get("net_income", "N/A"),
            q_eps_actual=earnings.get("eps_actual", "N/A"),
            q_eps_estimate=earnings.get("eps_estimate", "N/A"),
            q_eps_surprise=earnings.get("eps_surprise_pct", "N/A"),
            q_fcf=earnings.get("free_cashflow", "N/A"),
            next_earnings_date=earnings.get("next_earnings_date", "N/A"),
            eps_estimates="\n".join(eps_lines) if eps_lines else "暂无数据",
            rev_estimates="\n".join(rev_lines) if rev_lines else "暂无数据",
            five_year_growth=estimates.get("five_year_growth_pct", "N/A"),
            next_year_eps_growth=estimates.get("next_year_eps_growth_pct", "N/A"),
            next_year_rev_growth=estimates.get("next_year_rev_growth_pct", "N/A"),
            analyst_target_mean=analyst.get("target_mean", "N/A"),
            analyst_target_low=analyst.get("target_low", "N/A"),
            analyst_target_high=analyst.get("target_high", "N/A"),
            analyst_count=analyst.get("analyst_count", "N/A"),
            analyst_upside=analyst.get("upside_pct", "N/A"),
        )

        system = (
            f"你是顶级卖方研究分析师，专注于跨行业股票估值建模。今天是{current_date}。"
            "请基于提供的数据构建内部估值模型，输出结构化JSON。"
            "对任何行业和业务模式均适用：SaaS、硬件、金融、消费、能源、医药等。"
            "禁止引用训练数据中的具体历史日期。"
        )

        try:
            raw = self._chat(system, prompt, json_mode=True, model=MODEL_REASONER)
            result = json.loads(raw)
            print(
                f"[Valuation] Done. Method={result.get('valuation_methodology')} | "
                f"Base={result.get('scenarios', {}).get('base', {}).get('target_price')} | "
                f"Bear={result.get('scenarios', {}).get('bear', {}).get('target_price')} | "
                f"Bull={result.get('scenarios', {}).get('bull', {}).get('target_price')}"
            )
            return result
        except Exception as exc:
            print(f"[Valuation] Failed (non-fatal): {exc}")
            return {}
