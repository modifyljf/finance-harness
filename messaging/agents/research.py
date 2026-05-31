"""
ResearchAgent — deep research via Tavily web search + SEC EDGAR XBRL.
Runs before ValuationAgent to provide product-level and competitive intelligence.
"""
import os
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date

import requests

from messaging.agents.base import BaseAgent
from messaging.dto.hard_rules import MODEL_CHAT


_EDGAR_HEADERS = {"User-Agent": "FinanceHarness contact@financeharness.io"}
_EDGAR_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_EDGAR_FACTS_URL   = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
_TAVILY_URL        = "https://api.tavily.com/search"

# XBRL revenue tags to look for (in priority order)
_REVENUE_TAGS = [
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "Revenues",
    "SalesRevenueNet",
    "SalesRevenueGoodsNet",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
]


# ── EDGAR helpers ─────────────────────────────────────────────────────────────

def _get_cik(ticker: str) -> str | None:
    """Map ticker → zero-padded 10-digit CIK using EDGAR's company tickers JSON."""
    try:
        resp = requests.get(_EDGAR_TICKERS_URL, headers=_EDGAR_HEADERS, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        ticker_up = ticker.upper()
        for entry in data.values():
            if str(entry.get("ticker", "")).upper() == ticker_up:
                return str(entry["cik_str"]).zfill(10)
    except Exception as exc:
        print(f"[Research/EDGAR] CIK lookup failed (non-fatal): {exc}")
    return None


def _fetch_company_facts(cik: str) -> dict:
    """Fetch XBRL company facts from EDGAR (all tagged financial data)."""
    try:
        resp = requests.get(
            _EDGAR_FACTS_URL.format(cik=cik),
            headers=_EDGAR_HEADERS,
            timeout=25,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        print(f"[Research/EDGAR] Company facts fetch failed (non-fatal): {exc}")
        return {}


def _extract_quarterly_revenue(facts: dict) -> list[dict]:
    """
    Extract the last 6 quarters of revenue from XBRL facts.
    Returns [{period, value, unit, form}] sorted newest-first.
    """
    gaap = facts.get("facts", {}).get("us-gaap", {})
    for tag in _REVENUE_TAGS:
        node = gaap.get(tag)
        if not node:
            continue
        units = node.get("units", {})
        usd_entries = units.get("USD", [])
        # Keep only 10-Q and 10-K quarterly entries (form=10-Q, period=instant quarter)
        quarterly = [
            e for e in usd_entries
            if e.get("form") in ("10-Q", "10-K")
            and e.get("fp") in ("Q1", "Q2", "Q3", "Q4", "FY")
            and e.get("end")
        ]
        # Sort by end date descending, deduplicate by period end
        seen: set[str] = set()
        result = []
        for e in sorted(quarterly, key=lambda x: x["end"], reverse=True):
            key = e["end"]
            if key not in seen:
                seen.add(key)
                result.append({
                    "period_end": e["end"],
                    "fiscal_period": e.get("fp"),
                    "revenue": e["val"],
                    "form": e.get("form"),
                })
            if len(result) >= 6:
                break
        if result:
            return result
    return []


def _extract_recent_filings(facts: dict) -> list[dict]:
    """Extract a brief list of recent filing dates from company facts metadata."""
    # facts JSON has entityName and top-level metadata
    entity = facts.get("entityName", "")
    # We can't get filing list from facts directly, but we have the data
    return []


def _format_edgar_summary(ticker: str, quarterly_revenue: list[dict]) -> str:
    if not quarterly_revenue:
        return f"EDGAR: {ticker} 季度收入数据未找到（可能未按XBRL标准标注）"
    lines = [f"EDGAR XBRL季度收入数据（{ticker}，最近{len(quarterly_revenue)}个报告期）："]
    for q in quarterly_revenue:
        rev_m = round(q["revenue"] / 1e6, 1)
        lines.append(f"  {q['period_end']} ({q['fiscal_period']}): ${rev_m}M  [{q['form']}]")
    return "\n".join(lines)


# ── Tavily helpers ────────────────────────────────────────────────────────────

def _tavily_search(api_key: str, query: str, max_results: int = 5) -> dict:
    """Execute a single Tavily search. Returns raw API response."""
    try:
        resp = requests.post(
            _TAVILY_URL,
            json={
                "api_key": api_key,
                "query": query,
                "search_depth": "advanced",
                "max_results": max_results,
                "include_answer": True,
                "include_raw_content": False,
            },
            timeout=25,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        print(f"[Research/Tavily] Search failed for '{query[:50]}' (non-fatal): {exc}")
        return {}


def _format_tavily_result(label: str, result: dict) -> str:
    """Format a single Tavily result into readable text for the LLM prompt."""
    if not result:
        return f"[{label}] 搜索无结果"
    lines = [f"[{label}]"]
    answer = result.get("answer")
    if answer:
        lines.append(f"摘要回答: {answer}")
    for r in result.get("results", [])[:4]:
        title   = r.get("title", "")
        content = r.get("content", "")[:300]
        url     = r.get("url", "")
        lines.append(f"  • {title}")
        lines.append(f"    {content}")
        lines.append(f"    来源: {url}")
    return "\n".join(lines)


# ── ResearchAgent ─────────────────────────────────────────────────────────────

class ResearchAgent(BaseAgent):

    def __init__(self):
        super().__init__()
        self._tavily_key = os.environ.get("TAVILY_API_KEY", "")

    def _build_queries(self, plan: dict) -> list[tuple[str, str]]:
        """Build targeted search queries based on plan context."""
        md      = plan["market_snapshot"]
        ticker  = md["ticker"]
        company = md.get("company_name", ticker)
        sector  = md.get("sector", "")
        current_year = plan["current_date"][:4]
        prev_year    = str(int(current_year) - 1)

        return [
            (
                "segment_revenue",
                f"{company} {ticker} revenue by segment product breakdown {current_year} {prev_year} annual report",
            ),
            (
                "earnings_guidance",
                f"{company} {ticker} earnings call transcript management guidance outlook {current_year}",
            ),
            (
                "competitive_position",
                f"{company} {ticker} market share competitive analysis vs competitors {sector} {current_year}",
            ),
            (
                "analyst_thesis",
                f"{company} {ticker} bull bear investment thesis analyst report {current_year}",
            ),
        ]

    def _run_tavily(self, plan: dict) -> dict[str, dict]:
        """Run all Tavily queries in parallel. Returns {label: raw_result}."""
        if not self._tavily_key:
            print("[Research/Tavily] TAVILY_API_KEY not set — skipping web search.")
            return {}

        queries = self._build_queries(plan)
        results: dict[str, dict] = {}

        def _search(label_query: tuple[str, str]) -> tuple[str, dict]:
            label, query = label_query
            return label, _tavily_search(self._tavily_key, query)

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(_search, lq): lq[0] for lq in queries}
            for future in as_completed(futures):
                label, result = future.result()
                results[label] = result
                hit_count = len(result.get("results", []))
                print(f"[Research/Tavily] '{label}': {hit_count} results")

        return results

    def _run_edgar(self, plan: dict) -> dict:
        """Fetch EDGAR XBRL facts and extract quarterly revenue."""
        ticker = plan["market_snapshot"]["ticker"]
        print(f"[Research/EDGAR] Looking up CIK for {ticker}...")
        cik = _get_cik(ticker)
        if not cik:
            print(f"[Research/EDGAR] CIK not found for {ticker}")
            return {"cik": None, "quarterly_revenue": [], "summary": f"EDGAR: {ticker} CIK未找到"}

        print(f"[Research/EDGAR] CIK={cik}. Fetching company facts...")
        facts = _fetch_company_facts(cik)
        quarterly_revenue = _extract_quarterly_revenue(facts)
        print(f"[Research/EDGAR] {len(quarterly_revenue)} quarterly revenue periods extracted.")

        return {
            "cik": cik,
            "entity_name": facts.get("entityName", ""),
            "quarterly_revenue": quarterly_revenue,
            "summary": _format_edgar_summary(ticker, quarterly_revenue),
        }

    def _synthesize(self, plan: dict, tavily_results: dict[str, dict], edgar_data: dict) -> dict:
        """Use LLM to synthesize research into structured research_pack."""
        md      = plan["market_snapshot"]
        inp     = plan["input"]
        current_date = plan["current_date"]

        # Format all research inputs for the prompt
        tavily_sections = "\n\n".join(
            _format_tavily_result(label, result)
            for label, result in tavily_results.items()
        )

        prompt = self._load_prompt("research").format(
            current_date=current_date,
            ticker=md["ticker"],
            company_name=md.get("company_name", md["ticker"]),
            sector=md.get("sector", "N/A"),
            industry=md.get("industry", "N/A"),
            language=inp["language"],
            current_price=md.get("current_price", "N/A"),
            tavily_research=tavily_sections if tavily_sections else "（Tavily搜索未执行，API Key未配置）",
            edgar_summary=edgar_data.get("summary", "（EDGAR数据未获取）"),
            edgar_quarterly_revenue=json.dumps(
                edgar_data.get("quarterly_revenue", []), ensure_ascii=False, indent=2
            ),
        )

        system = (
            f"你是顶级股票研究分析师，今天是{current_date}。"
            "请从以下研究材料中提取结构化情报，输出标准JSON。"
            "只提取来自研究材料的事实，不要编造数字。"
            "禁止引用训练数据中的具体历史日期。"
        )

        try:
            raw = self._chat(system, prompt, json_mode=True, model=MODEL_CHAT)
            result = json.loads(raw)
            return result
        except Exception as exc:
            print(f"[Research] Synthesis failed (non-fatal): {exc}")
            return {}

    def run(self, plan: dict) -> dict:
        """
        Run full research pipeline: Tavily + EDGAR → LLM synthesis.
        Returns research_pack dict, injected into plan["research_pack"].
        Non-fatal: returns partial results on any failure.
        """
        print("[Research] Starting deep research (Tavily + EDGAR)...")

        # Run Tavily and EDGAR concurrently
        tavily_results: dict[str, dict] = {}
        edgar_data: dict = {}

        with ThreadPoolExecutor(max_workers=2) as executor:
            f_tavily = executor.submit(self._run_tavily, plan)
            f_edgar  = executor.submit(self._run_edgar,  plan)
            tavily_results = f_tavily.result()
            edgar_data     = f_edgar.result()

        # LLM synthesis
        print("[Research] Synthesizing research findings...")
        research_pack = self._synthesize(plan, tavily_results, edgar_data)

        # Always attach raw EDGAR data regardless of synthesis outcome
        research_pack["_edgar_quarterly_revenue"] = edgar_data.get("quarterly_revenue", [])
        research_pack["_edgar_cik"]               = edgar_data.get("cik")
        research_pack["_tavily_query_count"]       = len(tavily_results)

        print(
            f"[Research] Done. "
            f"Tavily={len(tavily_results)} queries | "
            f"EDGAR CIK={edgar_data.get('cik')} | "
            f"Segments={len(research_pack.get('product_segments', []))} | "
            f"Guidance items={len(research_pack.get('management_guidance', []))}"
        )
        return research_pack
