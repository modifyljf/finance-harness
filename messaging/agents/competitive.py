"""
CompetitiveAgent — peer comparison + market positioning analysis.
Fetches yfinance data for 3-5 identified competitors, compares vs target company,
adds Tavily market share research, synthesizes into structured competitive_pack.
"""
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import yfinance as yf

from messaging.agents.base import BaseAgent
from messaging.dto.hard_rules import MODEL_CHAT

_TAVILY_URL = "https://api.tavily.com/search"


def _tavily_search(api_key: str, query: str, max_results: int = 4) -> dict:
    try:
        resp = requests.post(
            _TAVILY_URL,
            json={"api_key": api_key, "query": query, "search_depth": "basic",
                  "max_results": max_results, "include_answer": True},
            timeout=20,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        print(f"[Competitive/Tavily] '{query[:40]}' failed (non-fatal): {exc}")
        return {}


def _fetch_peer_snapshot(ticker: str) -> dict:
    """Fetch key financial metrics for a peer via yfinance."""
    try:
        t = yf.Ticker(ticker)
        info = t.info
        hist = t.history(period="1y").dropna()
        ytd_pct = None
        if len(hist) >= 2:
            ytd_pct = round((hist["Close"].iloc[-1] - hist["Close"].iloc[0]) / hist["Close"].iloc[0] * 100, 1)
        cp = info.get("currentPrice") or info.get("regularMarketPrice")
        mc = info.get("marketCap")
        mc_str = f"${mc/1e9:.1f}B" if mc and mc >= 1e9 else (f"${mc/1e6:.0f}M" if mc else "N/A")
        return {
            "ticker":             ticker.upper(),
            "company_name":       info.get("longName", ticker.upper()),
            "current_price":      cp,
            "market_cap_str":     mc_str,
            "pe_forward":         round(info.get("forwardPE"), 1) if info.get("forwardPE") else None,
            "pe_trailing":        round(info.get("trailingPE"), 1) if info.get("trailingPE") else None,
            "peg":                round(info.get("pegRatio"), 2) if info.get("pegRatio") else None,
            "revenue_growth_yoy": round(info.get("revenueGrowth", 0) * 100, 1) if info.get("revenueGrowth") else None,
            "gross_margin":       round(info.get("grossMargins", 0) * 100, 1) if info.get("grossMargins") else None,
            "operating_margin":   round(info.get("operatingMargins", 0) * 100, 1) if info.get("operatingMargins") else None,
            "ps_ratio":           round(info.get("priceToSalesTrailing12Months"), 2) if info.get("priceToSalesTrailing12Months") else None,
            "ytd_performance_pct": ytd_pct,
            "recommendation":     info.get("recommendationKey", "").replace("_", " ").title(),
            "target_mean":        info.get("targetMeanPrice"),
        }
    except Exception as exc:
        print(f"[Competitive] Peer {ticker} fetch failed (non-fatal): {exc}")
        return {"ticker": ticker.upper(), "error": str(exc)}


class CompetitiveAgent(BaseAgent):

    def __init__(self):
        super().__init__()
        import os
        self._tavily_key = os.environ.get("TAVILY_API_KEY", "")

    def _identify_peers(self, plan: dict) -> list[str]:
        """Ask LLM for 3-5 competitor tickers. Returns list of ticker strings."""
        md = plan["market_snapshot"]
        system = "你是股票行业分析师。只返回JSON数组，不要任何额外文字。"
        user = (
            f"公司：{md.get('company_name')} ({md['ticker']})\n"
            f"行业：{md.get('sector')} / {md.get('industry')}\n\n"
            f"请列出该公司最相关的4个竞争对手的美股股票代码，以JSON数组返回。\n"
            f"示例格式：[\"NOW\", \"MSFT\", \"ORCL\", \"SAP\"]\n"
            f"只返回JSON数组，不要任何其他文字。"
        )
        try:
            raw = self._chat(system, user, json_mode=True, model=MODEL_CHAT)
            peers = json.loads(raw)
            if isinstance(peers, list):
                return [str(p).upper().strip() for p in peers if p][:5]
        except Exception as exc:
            print(f"[Competitive] Peer identification failed (non-fatal): {exc}")
        return []

    def _run_tavily(self, plan: dict) -> str:
        """Search for market share and competitive positioning."""
        if not self._tavily_key:
            return ""
        md = plan["market_snapshot"]
        query = (
            f"{md.get('company_name')} {md['ticker']} market share "
            f"competitive landscape vs competitors {plan['current_date'][:4]}"
        )
        result = _tavily_search(self._tavily_key, query)
        answer = result.get("answer", "")
        snippets = "\n".join(
            f"  • {r.get('title','')}: {r.get('content','')[:200]}"
            for r in result.get("results", [])[:3]
        )
        return f"{answer}\n{snippets}".strip()

    def _synthesize(self, plan: dict, peer_data: list[dict], tavily_text: str) -> dict:
        """LLM synthesis of competitive landscape into structured JSON."""
        md = plan["market_snapshot"]
        inp = plan["input"]
        current_date = plan["current_date"]

        target_snap = {
            "ticker":             md["ticker"],
            "company_name":       md.get("company_name"),
            "current_price":      md.get("current_price"),
            "market_cap_str":     md.get("market_cap_str"),
            "pe_forward":         plan["valuation_snapshot"].get("pe_forward"),
            "revenue_growth_yoy": plan["financial_snapshot"].get("revenue_growth_yoy"),
            "gross_margin":       plan["financial_snapshot"].get("gross_margin"),
        }

        prompt = self._load_prompt("competitive").format(
            current_date=current_date,
            ticker=md["ticker"],
            company_name=md.get("company_name", md["ticker"]),
            sector=md.get("sector", "N/A"),
            industry=md.get("industry", "N/A"),
            language=inp["language"],
            target_snapshot=json.dumps(target_snap, ensure_ascii=False, indent=2),
            peer_snapshots=json.dumps(peer_data, ensure_ascii=False, indent=2),
            tavily_competitive=tavily_text or "（未获取到竞争格局搜索结果）",
        )
        system = (
            f"你是顶级行业分析师，今天是{current_date}。"
            "请基于提供的数据进行竞争分析，输出结构化JSON。"
            "禁止编造任何财务数字，只使用提供的数据。"
        )
        try:
            raw = self._chat(system, prompt, json_mode=True, model=MODEL_CHAT)
            return json.loads(raw)
        except Exception as exc:
            print(f"[Competitive] Synthesis failed (non-fatal): {exc}")
            return {"peers": peer_data, "error": str(exc)}

    def run(self, plan: dict) -> dict:
        print("[Competitive] Identifying peers and fetching comparison data...")

        peer_tickers = self._identify_peers(plan)
        if not peer_tickers:
            print("[Competitive] No peers identified — skipping.")
            return {}

        print(f"[Competitive] Peers: {peer_tickers}")

        # Fetch peer data + Tavily in parallel
        peer_data: list[dict] = [None] * len(peer_tickers)
        tavily_text = ""

        with ThreadPoolExecutor(max_workers=6) as executor:
            f_tavily = executor.submit(self._run_tavily, plan)
            futures = {executor.submit(_fetch_peer_snapshot, t): i for i, t in enumerate(peer_tickers)}
            for future in as_completed(futures):
                peer_data[futures[future]] = future.result()
            tavily_text = f_tavily.result()

        peer_data = [p for p in peer_data if p]
        print(f"[Competitive] Fetched {len(peer_data)} peers. Synthesizing...")

        result = self._synthesize(plan, peer_data, tavily_text)
        print(f"[Competitive] Done. Moat={result.get('moat_rating')} | "
              f"Valuation vs peers={result.get('valuation_position')}")
        return result
