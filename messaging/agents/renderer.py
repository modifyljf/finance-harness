"""RendererAgent — deterministic HTML rendering (no LLM calls)."""
import json
import re
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from messaging.agents.base import BaseAgent
from messaging.dto.candidate import Candidate
from messaging.dto.hard_rules import TTS_CHARS_PER_SEC_ZH
from messaging.dto.rendered_message import RenderedMessage

TEMPLATES_DIR = Path(__file__).parent.parent.parent / "templates"


def _compute_autoslide(narration: str, slide_types: list[str]) -> dict[str, int]:
    """Return {slide_type: autoslide_ms} based on narration character counts."""
    segments = re.split(r'\[幻灯片: (\w+)\]', narration)
    char_counts: dict[str, int] = {}
    for i in range(1, len(segments), 2):
        stype = segments[i]
        content = segments[i + 1].strip() if i + 1 < len(segments) else ""
        char_counts[stype] = len(content)

    result: dict[str, int] = {}
    for stype in slide_types:
        chars = char_counts.get(stype, 0)
        secs = chars / TTS_CHARS_PER_SEC_ZH if chars else 30
        result[stype] = int(secs * 1000)  # milliseconds for Reveal.js
    return result


class RendererAgent(BaseAgent):

    def run(self, plan: dict, candidate: Candidate) -> RenderedMessage:
        print("[Renderer] Rendering HTML deck...")

        env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)), autoescape=False)
        template = env.get_template("deck.html.j2")

        md = plan["market_snapshot"]
        price_history = plan["price_history"]["items"]
        price_labels = json.dumps([p["date"] for p in price_history])
        price_values = json.dumps([p["close"] for p in price_history])

        price_change = md.get("price_change_pct", 0) or 0
        price_color = "#22c55e" if price_change >= 0 else "#ef4444"
        price_sign = "+" if price_change >= 0 else ""

        tech = plan["technical_indicators"]
        val = plan["valuation_snapshot"]

        # Overwrite market_overview headline with real numbers (renderer has authority)
        range_pos = md.get("52w_range_position_pct")
        pe_val = val.get("pe_trailing") or val.get("pe_forward")
        change_str = f"{'+' if price_change >= 0 else ''}{price_change}%"
        parts = [f"${md.get('current_price', 'N/A')}  {change_str}"]
        if pe_val:
            parts.append(f"PE {pe_val}x")
        if range_pos is not None:
            parts.append(f"52W区间 {range_pos}%")
        real_overview_headline = "  ·  ".join(parts)

        slides = candidate.slides
        slide_list = slides.get("slides", [])
        for s in slide_list:
            if s.get("type") == "market_overview":
                s["headline"] = real_overview_headline
                break

        # Compute per-slide autoslide timing from narration character counts
        slide_types = [s["type"] for s in slide_list]
        autoslide_map = _compute_autoslide(candidate.narration, slide_types)
        for s in slide_list:
            s["autoslide_ms"] = autoslide_map.get(s["type"], 30000)

        html = template.render(
            slides=slide_list,
            title=slides.get("title", md["ticker"]),
            ticker=md["ticker"],
            company_name=md.get("company_name", md["ticker"]),
            current_price=md.get("current_price", "N/A"),
            price_change_pct=price_change,
            price_sign=price_sign,
            price_color=price_color,
            market_cap_str=md.get("market_cap_str", "N/A"),
            pe_ratio=val.get("pe_trailing", "N/A"),
            high_52w=md.get("52w_high", "N/A"),
            low_52w=md.get("52w_low", "N/A"),
            volume=md.get("volume", "N/A"),
            price_labels=price_labels,
            price_values=price_values,
            ma20=tech.get("ma20"),
            ma50=tech.get("ma50"),
        )

        print(f"[Renderer] HTML deck rendered.")
        return html  # caller writes to disk
