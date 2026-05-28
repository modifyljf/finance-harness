"""
Stage 1: Planner
Fetches market data via yfinance and builds a comprehensive content plan (plan.json).
Includes: price history, technical indicators, analyst consensus, narrative signals.
"""
import time
from datetime import date, datetime, timezone

import yfinance as yf


# ── Technical Indicators ────────────────────────────────────────────────────

def _calculate_rsi(closes: list, period: int = 14) -> float | None:
    if len(closes) < period + 1:
        return None
    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains = [max(d, 0) for d in deltas]
    losses = [abs(min(d, 0)) for d in deltas]
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss == 0:
        return 100.0
    return round(100 - 100 / (1 + avg_gain / avg_loss), 1)


def _calculate_ma(closes: list, period: int) -> float | None:
    if len(closes) < period:
        return None
    return round(sum(closes[-period:]) / period, 2)


def _calculate_bollinger(closes: list, period: int = 20) -> dict | None:
    if len(closes) < period:
        return None
    recent = closes[-period:]
    ma = sum(recent) / period
    std = (sum((x - ma) ** 2 for x in recent) / period) ** 0.5
    return {
        "upper": round(ma + 2 * std, 2),
        "middle": round(ma, 2),
        "lower": round(ma - 2 * std, 2),
    }


def _volume_trend(volumes: list, period: int = 10) -> str:
    if len(volumes) < period * 2:
        return "unknown"
    recent_avg = sum(volumes[-period:]) / period
    prior_avg = sum(volumes[-period * 2:-period]) / period
    ratio = recent_avg / prior_avg if prior_avg else 1
    if ratio > 1.2:
        return "expanding"
    elif ratio < 0.8:
        return "shrinking"
    return "stable"


def _range_position(current: float, low52: float, high52: float) -> float | None:
    if not all([current, low52, high52]) or high52 == low52:
        return None
    return round((current - low52) / (high52 - low52) * 100, 1)


# ── News Fetching ────────────────────────────────────────────────────────────

def _build_keywords(ticker: str, company_name: str) -> list[str]:
    """Build a list of keyword variants to match against news text."""
    keywords = [ticker.upper()]
    # Add each significant word from company name (skip short filler words)
    skip = {"corp", "corporation", "inc", "ltd", "llc", "co", "the", "group", "holdings"}
    for word in company_name.split():
        w = word.strip(".,").lower()
        if len(w) > 3 and w not in skip:
            keywords.append(word.lower())
    return list(dict.fromkeys(keywords))  # deduplicate, preserve order


def fetch_weekly_news(t: yf.Ticker, ticker: str, company_name: str = "") -> list[dict]:
    """
    Fetch news from the past 7 days relevant to the specific ticker.
    Relevance scoring (checked against title + description):
      +3  ticker symbol found in title
      +2  company name keyword found in title
      +1  ticker or company keyword found in description/summary only
       0  no match → excluded unless nothing better available
    """
    cutoff = time.time() - 7 * 24 * 3600
    keywords = _build_keywords(ticker, company_name)
    raw_news = []

    try:
        raw_news = t.news or []
    except Exception:
        pass

    scored = []
    for item in raw_news:
        content = item.get("content") or item
        title = content.get("title") or item.get("title") or ""
        description = (
            content.get("summary")
            or content.get("description")
            or item.get("summary", "")
        )
        publisher = (
            (content.get("provider") or {}).get("displayName")
            or item.get("publisher", "")
        )
        pub_ts = content.get("pubDate") or item.get("providerPublishTime")

        # Normalize timestamp
        if isinstance(pub_ts, str):
            try:
                pub_dt = datetime.fromisoformat(pub_ts.replace("Z", "+00:00"))
                pub_unix = pub_dt.timestamp()
                pub_str = pub_dt.strftime("%Y-%m-%d %H:%M")
            except ValueError:
                pub_unix = 0
                pub_str = pub_ts[:16]
        elif isinstance(pub_ts, (int, float)):
            pub_unix = pub_ts
            pub_str = datetime.fromtimestamp(pub_ts).strftime("%Y-%m-%d %H:%M")
        else:
            continue

        if pub_unix < cutoff or not title:
            continue

        title_lower = title.lower()
        desc_lower = description.lower()
        ticker_kw = keywords[0].lower()  # always the symbol

        # Score by relevance
        score = 0
        if ticker_kw in title_lower:
            score += 3
        else:
            for kw in keywords[1:]:  # company name words
                if kw in title_lower:
                    score += 2
                    break
        if score == 0:
            if ticker_kw in desc_lower:
                score += 1
            else:
                for kw in keywords[1:]:
                    if kw in desc_lower:
                        score += 1
                        break

        scored.append({
            "title": title,
            "publisher": publisher,
            "published_at": pub_str,
            "_score": score,
            "_pub_unix": pub_unix,
        })

    # Sort: relevance desc, then newest first; exclude score=0 if any scored items exist
    scored.sort(key=lambda x: (-x["_score"], -x["_pub_unix"]))
    min_score = 1 if any(s["_score"] > 0 for s in scored) else 0

    seen = set()
    unique = []
    for r in scored:
        if r["_score"] < min_score:
            break
        if r["title"] not in seen:
            seen.add(r["title"])
            unique.append({
                "title": r["title"],
                "publisher": r["publisher"],
                "published_at": r["published_at"],
                "relevance": "high" if r["_score"] >= 3 else "medium" if r["_score"] == 2 else "low",
            })
        if len(unique) >= 15:
            break

    high = sum(1 for u in unique if u["relevance"] == "high")
    med = sum(1 for u in unique if u["relevance"] == "medium")
    print(f"[Planner] Found {len(unique)} relevant news (past 7 days): {high} high / {med} medium relevance for {ticker.upper()}.")
    return unique


# ── Data Fetching ────────────────────────────────────────────────────────────

def fetch_market_data(ticker: str, t: yf.Ticker = None) -> dict:
    if t is None:
        t = yf.Ticker(ticker)
    info = t.info

    # Fetch 90 days for MA50, keep 30 for chart display
    hist_90 = t.history(period="100d").dropna()
    hist_30 = hist_90.tail(30)

    closes_90 = [round(float(c), 2) for c in hist_90["Close"].tolist()]
    volumes_90 = [int(v) for v in hist_90["Volume"].tolist()]

    price_history = [
        {"date": str(d.date()), "close": round(float(c), 2)}
        for d, c in zip(hist_30.index, hist_30["Close"])
    ]

    current_price = (
        info.get("currentPrice")
        or info.get("regularMarketPrice")
        or (closes_90[-1] if closes_90 else None)
    )
    prev_close = info.get("previousClose") or info.get("regularMarketPreviousClose")
    price_change = round(current_price - prev_close, 2) if current_price and prev_close else None
    price_change_pct = round(price_change / prev_close * 100, 2) if price_change and prev_close else None

    market_cap = info.get("marketCap")
    if market_cap:
        if market_cap >= 1e12:
            market_cap_str = f"${market_cap/1e12:.2f}T"
        elif market_cap >= 1e9:
            market_cap_str = f"${market_cap/1e9:.2f}B"
        else:
            market_cap_str = f"${market_cap/1e6:.0f}M"
    else:
        market_cap_str = "N/A"

    # Technical indicators
    rsi = _calculate_rsi(closes_90)
    ma20 = _calculate_ma(closes_90, 20)
    ma50 = _calculate_ma(closes_90, 50)
    bollinger = _calculate_bollinger(closes_90, 20)
    vol_trend = _volume_trend(volumes_90)

    high52 = info.get("fiftyTwoWeekHigh")
    low52 = info.get("fiftyTwoWeekLow")
    range_pos = _range_position(current_price, low52, high52)

    # RSI interpretation
    if rsi is not None:
        if rsi >= 70:
            rsi_signal = "overbought"
        elif rsi <= 30:
            rsi_signal = "oversold"
        else:
            rsi_signal = "neutral"
    else:
        rsi_signal = "unknown"

    # Price vs MA signal
    ma_signal = "unknown"
    if current_price and ma20 and ma50:
        if current_price > ma20 > ma50:
            ma_signal = "bullish"
        elif current_price < ma20 < ma50:
            ma_signal = "bearish"
        else:
            ma_signal = "mixed"

    # Analyst consensus
    target_mean = info.get("targetMeanPrice")
    target_high = info.get("targetHighPrice")
    target_low = info.get("targetLowPrice")
    analyst_count = info.get("numberOfAnalystOpinions")
    recommendation = info.get("recommendationKey", "").replace("_", " ").title()
    upside_pct = (
        round((target_mean - current_price) / current_price * 100, 1)
        if target_mean and current_price
        else None
    )

    # Short interest & institutional
    short_float = info.get("shortPercentOfFloat")
    institutional_pct = info.get("heldPercentInstitutions")

    # Valuation
    pe_trailing = info.get("trailingPE")
    pe_forward = info.get("forwardPE")
    peg = info.get("pegRatio")
    ps_ratio = info.get("priceToSalesTrailing12Months")
    pb_ratio = info.get("priceToBook")
    ev_ebitda = info.get("enterpriseToEbitda")

    # Profitability
    revenue = info.get("totalRevenue")
    revenue_growth = info.get("revenueGrowth")
    earnings_growth = info.get("earningsGrowth")
    gross_margin = info.get("grossMargins")
    operating_margin = info.get("operatingMargins")
    profit_margin = info.get("profitMargins")
    free_cashflow = info.get("freeCashflow")
    return_on_equity = info.get("returnOnEquity")

    return {
        # Identity
        "ticker": ticker.upper(),
        "company_name": info.get("longName", ticker.upper()),
        "sector": info.get("sector", ""),
        "industry": info.get("industry", ""),
        "short_description": (info.get("longBusinessSummary") or "")[:400],

        # Price
        "current_price": current_price,
        "prev_close": prev_close,
        "price_change": price_change,
        "price_change_pct": price_change_pct,
        "market_cap": market_cap,
        "market_cap_str": market_cap_str,
        "volume": info.get("volume") or info.get("regularMarketVolume"),
        "avg_volume": info.get("averageVolume"),
        "beta": info.get("beta"),
        "52w_high": high52,
        "52w_low": low52,
        "52w_range_position_pct": range_pos,

        # Technical
        "technical": {
            "rsi_14": rsi,
            "rsi_signal": rsi_signal,
            "ma20": ma20,
            "ma50": ma50,
            "bollinger": bollinger,
            "volume_trend": vol_trend,
            "ma_signal": ma_signal,
            "price_vs_ma20_pct": round((current_price - ma20) / ma20 * 100, 1) if current_price and ma20 else None,
            "price_vs_ma50_pct": round((current_price - ma50) / ma50 * 100, 1) if current_price and ma50 else None,
        },

        # Analyst / Sentiment
        "analyst": {
            "recommendation": recommendation,
            "target_mean": target_mean,
            "target_high": target_high,
            "target_low": target_low,
            "analyst_count": analyst_count,
            "upside_pct": upside_pct,
        },
        "short_interest": round(short_float * 100, 1) if short_float else None,
        "institutional_ownership_pct": round(institutional_pct * 100, 1) if institutional_pct else None,

        # Valuation
        "valuation": {
            "pe_trailing": round(pe_trailing, 2) if pe_trailing else None,
            "pe_forward": round(pe_forward, 2) if pe_forward else None,
            "peg": round(peg, 2) if peg else None,
            "ps_ratio": round(ps_ratio, 2) if ps_ratio else None,
            "pb_ratio": round(pb_ratio, 2) if pb_ratio else None,
            "ev_ebitda": round(ev_ebitda, 2) if ev_ebitda else None,
        },

        # Financials
        "financials": {
            "revenue": revenue,
            "revenue_growth_yoy": round(revenue_growth * 100, 1) if revenue_growth else None,
            "earnings_growth_yoy": round(earnings_growth * 100, 1) if earnings_growth else None,
            "gross_margin": round(gross_margin * 100, 1) if gross_margin else None,
            "operating_margin": round(operating_margin * 100, 1) if operating_margin else None,
            "profit_margin": round(profit_margin * 100, 1) if profit_margin else None,
            "free_cashflow": free_cashflow,
            "roe": round(return_on_equity * 100, 1) if return_on_equity else None,
            "eps_trailing": info.get("trailingEps"),
            "eps_forward": info.get("forwardEps"),
        },

        # Chart data (30-day)
        "price_history": price_history,

        # Weekly news (fetched separately in run())
        "weekly_news": [],
    }


def build_narrative_signals(md: dict, current_date: str) -> dict:
    """Derive narrative and sentiment signals from market data."""
    tech = md["technical"]
    analyst = md["analyst"]
    rsi = tech.get("rsi_14")
    range_pos = md.get("52w_range_position_pct")

    # Sentiment score 0-100 (composite)
    score_parts = []
    if rsi is not None:
        score_parts.append(100 - abs(rsi - 50) * 2)  # 50 RSI = 100, extremes = lower
    if range_pos is not None:
        score_parts.append(range_pos)
    if tech.get("ma_signal") == "bullish":
        score_parts.append(75)
    elif tech.get("ma_signal") == "bearish":
        score_parts.append(25)
    else:
        score_parts.append(50)

    sentiment_score = round(sum(score_parts) / len(score_parts), 1) if score_parts else 50

    if sentiment_score >= 65:
        sentiment_label = "偏多"
    elif sentiment_score <= 35:
        sentiment_label = "偏空"
    else:
        sentiment_label = "中性"

    # Upside/downside from analyst target
    upside = analyst.get("upside_pct")
    analyst_sentiment = "看多" if upside and upside > 10 else "看空" if upside and upside < -5 else "中性"

    return {
        "analysis_date": current_date,
        "sentiment_score": sentiment_score,
        "sentiment_label": sentiment_label,
        "analyst_sentiment": analyst_sentiment,
        "key_themes": _derive_themes(md),
        "risk_level": _derive_risk_level(md),
    }


def _derive_themes(md: dict) -> list[str]:
    themes = []
    sector = md.get("sector", "")
    tech = md["technical"]
    analyst = md["analyst"]

    if "Technology" in sector or "Semiconductor" in md.get("industry", ""):
        themes.append("AI与算力需求驱动")
    if "Health" in sector:
        themes.append("医疗创新周期")
    if "Energy" in sector:
        themes.append("能源转型主题")
    if "Financial" in sector:
        themes.append("利率环境敏感")

    rsi = tech.get("rsi_14") or 50
    if rsi > 65:
        themes.append("动量强势，追高风险上升")
    elif rsi < 40:
        themes.append("超跌修复机会")

    upside = analyst.get("upside_pct")
    if upside and upside > 20:
        themes.append(f"分析师平均目标价存在{upside}%上行空间")
    elif upside and upside < 0:
        themes.append(f"分析师目标价低于现价{abs(upside)}%")

    return themes[:4]


def _derive_risk_level(md: dict) -> str:
    beta = md.get("beta") or 1.0
    rsi = (md["technical"].get("rsi_14") or 50)
    range_pos = (md.get("52w_range_position_pct") or 50)

    risk_score = 0
    if beta > 1.5:
        risk_score += 2
    elif beta > 1.0:
        risk_score += 1
    if rsi > 70:
        risk_score += 2
    if range_pos > 85:
        risk_score += 1

    if risk_score >= 4:
        return "高风险"
    elif risk_score >= 2:
        return "中等风险"
    return "较低风险"


def build_slide_outline(duration_minutes: int) -> list:
    outline = [
        {"type": "cover", "approx_seconds": 15},
        {"type": "market_overview", "approx_seconds": 45},
        {"type": "price_action", "approx_seconds": 60},
        {"type": "key_points", "approx_seconds": 60},
    ]
    if duration_minutes >= 6:
        outline.append({"type": "financials", "approx_seconds": 60})
        outline.append({"type": "risk", "approx_seconds": 45})
    if duration_minutes >= 8:
        outline.append({"type": "catalyst", "approx_seconds": 45})
    outline.append({"type": "outlook", "approx_seconds": 45})
    outline.append({"type": "summary", "approx_seconds": 30})
    return outline


def run(ticker: str, market: str, language: str, duration_minutes: int, style: str) -> dict:
    current_date = date.today().isoformat()
    print(f"[Planner] Fetching market data for {ticker} (as of {current_date})...")

    t = yf.Ticker(ticker)
    market_data = fetch_market_data(ticker, t)
    market_data["weekly_news"] = fetch_weekly_news(t, ticker, market_data.get("company_name", ""))

    narrative_signals = build_narrative_signals(market_data, current_date)
    slide_outline = build_slide_outline(duration_minutes)

    plan = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "current_date": current_date,
        "input": {
            "ticker": ticker,
            "market": market,
            "language": language,
            "duration_minutes": duration_minutes,
            "style": style,
        },
        "market_data": market_data,
        "narrative_signals": narrative_signals,
        "slide_outline": slide_outline,
    }

    tech = market_data["technical"]
    print(
        f"[Planner] Done. RSI={tech.get('rsi_14')} | MA20=${tech.get('ma20')} | "
        f"MA50=${tech.get('ma50')} | Sentiment={narrative_signals['sentiment_label']} | "
        f"{len(slide_outline)} slides"
    )
    return plan
