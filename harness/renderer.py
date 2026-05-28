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

    md = plan["market_data"]
    price_history = md.get("price_history", [])
    price_labels = json.dumps([p["date"] for p in price_history])
    price_values = json.dumps([p["close"] for p in price_history])

    price_change = md.get("price_change_pct", 0) or 0
    price_color = "#22c55e" if price_change >= 0 else "#ef4444"
    price_sign = "+" if price_change >= 0 else ""

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
        pe_ratio=md.get("pe_ratio", "N/A"),
        high_52w=md.get("52w_high", "N/A"),
        low_52w=md.get("52w_low", "N/A"),
        volume=md.get("volume", "N/A"),
        price_labels=price_labels,
        price_values=price_values,
    )

    output_path.write_text(html, encoding="utf-8")
    print(f"[Renderer] Saved: {output_path}")
