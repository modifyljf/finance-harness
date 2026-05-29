"""
Stage 4: Renderer
Renders slides.json + plan.json into a Reveal.js HTML presentation via Jinja2.
"""
import json
from pathlib import Path

from jinja2 import Environment, FileSystemLoader


TEMPLATES_DIR = Path(__file__).parent.parent / "templates"


def run(plan: dict, slides_data: dict, output_path: Path) -> None:
    print("[Renderer] Rendering HTML deck...")

    env = Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=False,
    )
    template = env.get_template("deck.html.j2")

    md = plan["market_snapshot"]
    price_history = plan["price_history"]["items"]
    price_labels = json.dumps([p["date"] for p in price_history])
    price_values = json.dumps([p["close"] for p in price_history])

    price_change = md.get("price_change_pct", 0) or 0
    price_color = "#22c55e" if price_change >= 0 else "#ef4444"
    price_sign = "+" if price_change >= 0 else ""

    tech = plan["technical_indicators"]
    val  = plan["valuation_snapshot"]

    # Overwrite market_overview headline with real numbers so AI can't hallucinate it
    range_pos = md.get("52w_range_position_pct")
    pe_val    = val.get("pe_trailing") or val.get("pe_forward")
    change    = md.get("price_change_pct", 0) or 0
    change_str = f"{'+' if change >= 0 else ''}{change}%"
    parts = [f"${md.get('current_price', 'N/A')}  {change_str}"]
    if pe_val:
        parts.append(f"PE {pe_val}x")
    if range_pos is not None:
        parts.append(f"52W区间 {range_pos}%")
    real_overview_headline = "  ·  ".join(parts)

    slides = slides_data["slides"]
    for s in slides:
        if s.get("type") == "market_overview":
            s["headline"] = real_overview_headline
            break

    html = template.render(
        slides=slides_data["slides"],
        title=slides_data.get("title", md["ticker"]),
        ticker=md["ticker"],
        company_name=md.get("company_name", md["ticker"]),
        current_price=md.get("current_price", "N/A"),
        price_change_pct=md.get("price_change_pct", 0),
        price_sign=price_sign,
        price_color=price_color,
        market_cap_str=md.get("market_cap_str", "N/A"),
        pe_ratio=plan["valuation_snapshot"].get("pe_trailing", "N/A"),
        high_52w=md.get("52w_high", "N/A"),
        low_52w=md.get("52w_low", "N/A"),
        volume=md.get("volume", "N/A"),
        price_labels=price_labels,
        price_values=price_values,
        ma20=tech.get("ma20"),
        ma50=tech.get("ma50"),
    )

    output_path.write_text(html, encoding="utf-8")
    print(f"[Renderer] Saved: {output_path}")
