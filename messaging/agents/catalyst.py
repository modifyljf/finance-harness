"""
CatalystAgent — builds a verified event calendar with real dates via Tavily search.
Prevents LLM from guessing event dates (e.g., Dreamforce timing).
"""
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from messaging.agents.base import BaseAgent
from messaging.dto.hard_rules import MODEL_CHAT

_TAVILY_URL = "https://api.tavily.com/search"


def _tavily_search(api_key: str, query: str, max_results: int = 3) -> dict:
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
        print(f"[Catalyst/Tavily] '{query[:40]}' failed (non-fatal): {exc}")
        return {}


# Common recurring events by sector keyword — used to seed the search
_SECTOR_EVENTS = {
    "technology": ["annual developer conference", "investor day", "earnings call"],
    "software":   ["user conference", "investor day", "earnings call"],
    "financial":  ["investor day", "earnings call", "fed meeting"],
    "healthcare": ["FDA panel", "investor day", "earnings call"],
    "default":    ["investor day", "earnings call", "annual meeting"],
}


class CatalystAgent(BaseAgent):

    def __init__(self):
        super().__init__()
        self._tavily_key = os.environ.get("TAVILY_API_KEY", "")

    def _extract_event_candidates(self, plan: dict) -> list[str]:
        """Extract event mentions from research_pack. Returns list of event name strings."""
        research = plan.get("research_pack", {})
        guidance = research.get("management_guidance", [])
        thesis   = research.get("analyst_thesis", {})

        candidates = set()
        # Pull from guidance contexts
        for g in guidance:
            ctx = g.get("context", "")
            if any(kw in ctx.lower() for kw in ["conference", "day", "summit", "meeting", "earnings", "call"]):
                candidates.add(ctx[:60])

        # Always include next earnings date as a catalyst
        earnings_snap = plan.get("earnings_snapshot", {})
        next_date = earnings_snap.get("next_earnings_date")
        if next_date:
            candidates.add(f"Next earnings report ({next_date})")

        # Add sector-standard events
        sector = plan["market_snapshot"].get("sector", "").lower()
        key = next((k for k in _SECTOR_EVENTS if k in sector), "default")
        company = plan["market_snapshot"].get("company_name", plan["market_snapshot"]["ticker"])
        for evt in _SECTOR_EVENTS[key]:
            candidates.add(f"{company} {evt} 2025 2026")

        return list(candidates)[:8]

    def _search_event_date(self, query: str, company: str, year: str) -> dict:
        """Search Tavily for the actual date of an event."""
        search_query = f"{company} {query} date {year} schedule announcement"
        result = _tavily_search(self._tavily_key, search_query)
        return {
            "query":  query,
            "answer": result.get("answer", ""),
            "snippets": [
                {"title": r.get("title", ""), "content": r.get("content", "")[:200], "url": r.get("url", "")}
                for r in result.get("results", [])[:3]
            ],
        }

    def _synthesize(self, plan: dict, search_results: list[dict]) -> list[dict]:
        """LLM synthesizes search results into structured event calendar."""
        md = plan["market_snapshot"]
        inp = plan["input"]
        current_date = plan["current_date"]

        results_text = json.dumps(search_results, ensure_ascii=False, indent=2)

        prompt = self._load_prompt("catalyst").format(
            current_date=current_date,
            ticker=md["ticker"],
            company_name=md.get("company_name", md["ticker"]),
            sector=md.get("sector", "N/A"),
            language=inp["language"],
            search_results=results_text,
            next_earnings_date=plan.get("earnings_snapshot", {}).get("next_earnings_date", "未知"),
        )
        system = (
            f"你是专业的投资事件日历分析师，今天是{current_date}。"
            "请从搜索结果中提取真实的事件日期，输出结构化JSON数组。"
            "如果日期不确定，必须标注date_confidence为low，不得猜测具体日期。"
        )
        try:
            raw = self._chat(system, prompt, json_mode=True, model=MODEL_CHAT)
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, dict) and "events" in parsed:
                return parsed["events"]
            return []
        except Exception as exc:
            print(f"[Catalyst] Synthesis failed (non-fatal): {exc}")
            return []

    def run(self, plan: dict) -> list[dict]:
        print("[Catalyst] Building verified event calendar...")

        if not self._tavily_key:
            print("[Catalyst] TAVILY_API_KEY not set — returning empty calendar.")
            return []

        candidates = self._extract_event_candidates(plan)
        company  = plan["market_snapshot"].get("company_name", plan["market_snapshot"]["ticker"])
        year     = plan["current_date"][:4]

        print(f"[Catalyst] Searching dates for {len(candidates)} event candidates...")

        search_results: list[dict] = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(self._search_event_date, c, company, year) for c in candidates]
            for future in as_completed(futures):
                search_results.append(future.result())

        events = self._synthesize(plan, search_results)
        high_conf = sum(1 for e in events if e.get("date_confidence") == "high")
        print(f"[Catalyst] Done. {len(events)} events | {high_conf} high-confidence dates.")
        return events
