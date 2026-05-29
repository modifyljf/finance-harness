"""
Stage 1: Planner — Evidence Pack Builder
Fetches market data via yfinance and builds a structured evidence package (plan.json).

Design principle: Planner is the data engineering layer.
It provides structured signals and facts, not investment conclusions.
All analysis judgment belongs to the Generator agents.
"""
import os
import time
from datetime import date, datetime, timedelta, timezone

import requests
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


# ── News Classification ──────────────────────────────────────────────────────

_CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "earnings": ["earnings", "revenue", "eps", "profit", "loss", "quarterly", "q1", "q2",
                 "q3", "q4", "fiscal", "guidance", "beat", "miss", "results", "income"],
    "product":  ["product", "launch", "chip", "gpu", "cpu", "model", "hardware", "software",
                 "release", "unveil", "announce", "platform"],
    "regulation": ["regulation", "sec", "ftc", "antitrust", "ban", "sanction", "compliance",
                   "law", "government", "export", "restrict", "probe", "investigation"],
    "analyst":  ["analyst", "upgrade", "downgrade", "price target", "rating", "buy", "sell",
                 "hold", "coverage", "initiat", "reiterat", "overweight", "underweight"],
    "macro":    ["fed", "interest rate", "inflation", "gdp", "economy", "recession",
                 "market", "nasdaq", "s&p", "tariff", "trade", "treasury"],
    "insider":  ["insider", "ceo", "cfo", "executive", "board", "buyback",
                 "repurchase", "acquisition", "merger", "stake"],
    "partnership": ["partnership", "deal", "contract", "agreement", "collaborat",
                    "joint venture", "supply", "customer", "client"],
}

_CATEGORY_WHY: dict[str, str] = {
    "earnings":    "May affect earnings expectations, guidance, or valuation multiples.",
    "product":     "May affect product demand, competitive positioning, or revenue outlook.",
    "regulation":  "May introduce regulatory risk, export restrictions, or compliance costs.",
    "analyst":     "May shift analyst consensus, price targets, or investor sentiment.",
    "macro":       "May affect sector risk appetite, valuation multiples, or macro conditions.",
    "insider":     "May signal management confidence, corporate strategy, or capital allocation.",
    "partnership": "May affect revenue growth opportunities or supply chain positioning.",
    "other":       "Flagged as relevant to the stock by keyword match.",
}


def _classify_news_item(title: str, description: str = "") -> tuple[str, str]:
    text = (title + " " + description).lower()
    for category, keywords in _CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                return category, _CATEGORY_WHY[category]
    return "other", _CATEGORY_WHY["other"]


# ── News Fetching ────────────────────────────────────────────────────────────

def _build_keywords(ticker: str, company_name: str) -> list[str]:
    keywords = [ticker.upper()]
    skip = {"corp", "corporation", "inc", "ltd", "llc", "co", "the", "group", "holdings"}
    for word in company_name.split():
        w = word.strip(".,").lower()
        if len(w) > 3 and w not in skip:
            keywords.append(word.lower())
    return list(dict.fromkeys(keywords))


def _fetch_newsapi(ticker: str, company_name: str, api_key: str) -> list[dict]:
    """
    Fetch past-7-day news from NewsAPI and enrich with the same schema
    used by fetch_weekly_news (category, why_relevant, matched_keywords, relevance_score).
    """
    keywords = _build_keywords(ticker, company_name)
    ticker_kw = keywords[0]  # symbol

    # Build query: e.g. NVDA OR "NVIDIA" OR "Jensen Huang"
    company_words = [k for k in keywords[1:] if len(k) >= 5][:2]
    if company_words:
        q = f'{ticker_kw} OR "{" ".join(company_words)}"'
    else:
        q = ticker_kw

    from_date = (date.today() - timedelta(days=7)).isoformat()
    to_date   = date.today().isoformat()

    try:
        resp = requests.get(
            "https://newsapi.org/v2/everything",
            params={
                "q": q,
                "from": from_date,
                "to": to_date,
                "language": "en",
                "sortBy": "relevance",
                "pageSize": 30,
                "apiKey": api_key,
            },
            timeout=12,
        )
        if resp.status_code == 429:
            print("[Planner/NewsAPI] Rate limit hit (429) — skipping, will use yfinance news.")
            return []
        if resp.status_code == 401:
            print("[Planner/NewsAPI] Invalid API key (401) — check NEWS_API_KEY.")
            return []
        resp.raise_for_status()
        articles = resp.json().get("articles", [])
    except requests.exceptions.Timeout:
        print("[Planner/NewsAPI] Request timed out — skipping.")
        return []
    except requests.exceptions.RequestException as exc:
        print(f"[Planner/NewsAPI] Request error: {exc}")
        return []
    except Exception as exc:
        print(f"[Planner/NewsAPI] Unexpected error: {exc}")
        return []

    result: list[dict] = []
    seen_titles: set[str] = set()

    for article in articles:
        title       = (article.get("title") or "").strip()
        description = article.get("description") or ""
        publisher   = (article.get("source") or {}).get("name") or ""
        url         = article.get("url") or ""
        pub_raw     = article.get("publishedAt") or ""

        if not title or title == "[Removed]":
            continue

        try:
            pub_dt  = datetime.fromisoformat(pub_raw.replace("Z", "+00:00"))
            pub_str = pub_dt.strftime("%Y-%m-%d %H:%M")
        except Exception:
            pub_str = pub_raw[:16]

        # Relevance scoring (same scale as yfinance scorer: 0-1)
        title_lower = title.lower()
        desc_lower  = description.lower()
        matched: list[str] = []
        if ticker_kw.lower() in title_lower:
            score = 1.0
            matched.append(ticker_kw.upper())
        else:
            hit = next((k for k in keywords[1:] if k.lower() in title_lower), None)
            if hit:
                score = 0.67
                matched.append(hit)
            elif ticker_kw.lower() in desc_lower:
                score = 0.33
                matched.append(ticker_kw.upper())
            else:
                score = 0.1  # keep but mark low

        category, why_relevant = _classify_news_item(title, description)

        if title not in seen_titles:
            seen_titles.add(title)
            result.append({
                "title":            title,
                "publisher":        publisher,
                "published_at":     pub_str,
                "url":              url,
                "category":         category,
                "why_relevant":     why_relevant,
                "matched_keywords": matched,
                "relevance_score":  round(score, 2),
                "_score":           score,
            })

    result.sort(key=lambda x: -x["_score"])
    # Drop very-low relevance if better items exist
    min_score = 0.33 if any(r["_score"] >= 0.33 for r in result) else 0
    filtered  = [r for r in result if r["_score"] >= min_score][:15]
    output    = [{k: v for k, v in r.items() if k != "_score"} for r in filtered]

    high = sum(1 for r in output if r["relevance_score"] >= 1.0)
    med  = sum(1 for r in output if 0.5 <= r["relevance_score"] < 1.0)
    print(f"[Planner/NewsAPI] {len(output)} articles (past 7d): {high} high / {med} medium for {ticker.upper()}.")
    return output


def _merge_news(yf_items: list[dict], api_items: list[dict], max_items: int = 20) -> list[dict]:
    """
    Merge yfinance + NewsAPI results, deduplicate by title similarity,
    sort by relevance_score desc. Returns up to max_items.
    """
    seen: set[str] = set()
    merged: list[dict] = []

    for item in yf_items + api_items:
        # Normalise title for dedup (lowercase, strip punctuation)
        key = "".join(c for c in item["title"].lower() if c.isalnum() or c == " ").strip()
        key = " ".join(key.split())  # collapse whitespace
        if key and key not in seen:
            seen.add(key)
            merged.append(item)

    merged.sort(key=lambda x: -x.get("relevance_score", 0))
    return merged[:max_items]


def fetch_weekly_news(t: yf.Ticker, ticker: str, company_name: str = "") -> list[dict]:
    """
    Fetch past-7-day news and enrich each item with:
    - category (earnings / product / regulation / analyst / macro / insider / partnership / other)
    - why_relevant: one-sentence explanation of potential impact
    - matched_keywords: which keywords triggered the relevance match
    - relevance_score: 0.0–1.0 normalized from raw match score
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
        ticker_kw = keywords[0].lower()

        score = 0
        matched: list[str] = []
        if ticker_kw in title_lower:
            score += 3
            matched.append(ticker_kw.upper())
        else:
            for kw in keywords[1:]:
                if kw in title_lower:
                    score += 2
                    matched.append(kw)
                    break
        if score == 0:
            if ticker_kw in desc_lower:
                score += 1
                matched.append(ticker_kw.upper())
            else:
                for kw in keywords[1:]:
                    if kw in desc_lower:
                        score += 1
                        matched.append(kw)
                        break

        category, why_relevant = _classify_news_item(title, description)

        scored.append({
            "title": title,
            "publisher": publisher,
            "published_at": pub_str,
            "category": category,
            "why_relevant": why_relevant,
            "matched_keywords": matched,
            "relevance_score": round(score / 3, 2),
            "_score": score,
            "_pub_unix": pub_unix,
        })

    scored.sort(key=lambda x: (-x["_score"], -x["_pub_unix"]))
    min_score = 1 if any(s["_score"] > 0 for s in scored) else 0

    seen: set[str] = set()
    unique: list[dict] = []
    for r in scored:
        if r["_score"] < min_score:
            break
        if r["title"] not in seen:
            seen.add(r["title"])
            unique.append({k: v for k, v in r.items() if not k.startswith("_")})
        if len(unique) >= 15:
            break

    high = sum(1 for u in unique if u["relevance_score"] >= 1.0)
    med  = sum(1 for u in unique if 0.5 <= u["relevance_score"] < 1.0)
    print(f"[Planner] Found {len(unique)} relevant news (past 7 days): {high} high / {med} medium for {ticker.upper()}.")
    return unique


# ── Market Data Fetching ─────────────────────────────────────────────────────

def fetch_market_data(ticker: str, t: yf.Ticker = None) -> dict:
    """
    Returns a dict with five separate snapshots plus raw price history items.
    Keys: market_snapshot, price_history_items, technical_indicators,
          valuation_snapshot, financial_snapshot, analyst_snapshot
    """
    if t is None:
        t = yf.Ticker(ticker)
    info = t.info

    hist_90 = t.history(period="100d").dropna()
    hist_30 = hist_90.tail(30)

    closes_90 = [round(float(c), 2) for c in hist_90["Close"].tolist()]
    volumes_90 = [int(v) for v in hist_90["Volume"].tolist()]

    price_history_items = [
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

    rsi    = _calculate_rsi(closes_90)
    ma20   = _calculate_ma(closes_90, 20)
    ma50   = _calculate_ma(closes_90, 50)
    bollinger = _calculate_bollinger(closes_90, 20)
    vol_trend = _volume_trend(volumes_90)

    high52   = info.get("fiftyTwoWeekHigh")
    low52    = info.get("fiftyTwoWeekLow")
    range_pos = _range_position(current_price, low52, high52)

    rsi_signal = (
        "overbought" if rsi is not None and rsi >= 70
        else "oversold" if rsi is not None and rsi <= 30
        else "neutral" if rsi is not None
        else "unknown"
    )

    ma_signal = "unknown"
    if current_price and ma20 and ma50:
        if current_price > ma20 > ma50:
            ma_signal = "bullish"
        elif current_price < ma20 < ma50:
            ma_signal = "bearish"
        else:
            ma_signal = "mixed"

    target_mean  = info.get("targetMeanPrice")
    target_high  = info.get("targetHighPrice")
    target_low   = info.get("targetLowPrice")
    analyst_count = info.get("numberOfAnalystOpinions")
    recommendation = info.get("recommendationKey", "").replace("_", " ").title()
    upside_pct = (
        round((target_mean - current_price) / current_price * 100, 1)
        if target_mean and current_price else None
    )

    short_float      = info.get("shortPercentOfFloat")
    institutional_pct = info.get("heldPercentInstitutions")
    pe_trailing      = info.get("trailingPE")
    pe_forward       = info.get("forwardPE")
    peg              = info.get("pegRatio")
    ps_ratio         = info.get("priceToSalesTrailing12Months")
    pb_ratio         = info.get("priceToBook")
    ev_ebitda        = info.get("enterpriseToEbitda")
    revenue          = info.get("totalRevenue")
    revenue_growth   = info.get("revenueGrowth")
    earnings_growth  = info.get("earningsGrowth")
    gross_margin     = info.get("grossMargins")
    operating_margin = info.get("operatingMargins")
    profit_margin    = info.get("profitMargins")
    free_cashflow    = info.get("freeCashflow")
    roe              = info.get("returnOnEquity")

    return {
        "price_history_items": price_history_items,

        "market_snapshot": {
            "ticker":          ticker.upper(),
            "company_name":    info.get("longName", ticker.upper()),
            "sector":          info.get("sector", ""),
            "industry":        info.get("industry", ""),
            "short_description": (info.get("longBusinessSummary") or "")[:400],
            "current_price":   current_price,
            "prev_close":      prev_close,
            "price_change":    price_change,
            "price_change_pct": price_change_pct,
            "market_cap":      market_cap,
            "market_cap_str":  market_cap_str,
            "volume":          info.get("volume") or info.get("regularMarketVolume"),
            "avg_volume":      info.get("averageVolume"),
            "beta":            info.get("beta"),
            "52w_high":        high52,
            "52w_low":         low52,
            "52w_range_position_pct": range_pos,
            "short_interest":  round(short_float * 100, 1) if short_float else None,
            "institutional_ownership_pct": round(institutional_pct * 100, 1) if institutional_pct else None,
        },

        "technical_indicators": {
            "rsi_14":          rsi,
            "rsi_signal":      rsi_signal,
            "ma20":            ma20,
            "ma50":            ma50,
            "bollinger":       bollinger,
            "volume_trend":    vol_trend,
            "ma_signal":       ma_signal,
            "price_vs_ma20_pct": round((current_price - ma20) / ma20 * 100, 1) if current_price and ma20 else None,
            "price_vs_ma50_pct": round((current_price - ma50) / ma50 * 100, 1) if current_price and ma50 else None,
        },

        "valuation_snapshot": {
            "pe_trailing": round(pe_trailing, 2) if pe_trailing else None,
            "pe_forward":  round(pe_forward,  2) if pe_forward  else None,
            "peg":         round(peg,         2) if peg         else None,
            "ps_ratio":    round(ps_ratio,    2) if ps_ratio    else None,
            "pb_ratio":    round(pb_ratio,    2) if pb_ratio    else None,
            "ev_ebitda":   round(ev_ebitda,   2) if ev_ebitda   else None,
        },

        "financial_snapshot": {
            "revenue":            revenue,
            "revenue_growth_yoy": round(revenue_growth   * 100, 1) if revenue_growth   else None,
            "earnings_growth_yoy": round(earnings_growth * 100, 1) if earnings_growth  else None,
            "gross_margin":       round(gross_margin     * 100, 1) if gross_margin     else None,
            "operating_margin":   round(operating_margin * 100, 1) if operating_margin else None,
            "profit_margin":      round(profit_margin    * 100, 1) if profit_margin    else None,
            "free_cashflow":      free_cashflow,
            "roe":                round(roe * 100, 1) if roe else None,
            "eps_trailing":       info.get("trailingEps"),
            "eps_forward":        info.get("forwardEps"),
        },

        "analyst_snapshot": {
            "recommendation": recommendation,
            "target_mean":    target_mean,
            "target_high":    target_high,
            "target_low":     target_low,
            "analyst_count":  analyst_count,
            "upside_pct":     upside_pct,
        },
    }


# ── Computed Signals ─────────────────────────────────────────────────────────

def build_computed_signals(
    market_snapshot: dict,
    technical: dict,
    valuation: dict,
    analyst: dict,
    news_items: list[dict],
    current_date: str,
) -> dict:
    """
    Derive objective computed signals from market data.
    Outputs data-layer signals only — no buy/sell/hold conclusions.
    """
    rsi       = technical.get("rsi_14")
    range_pos = market_snapshot.get("52w_range_position_pct")
    ma_signal = technical.get("ma_signal", "unknown")
    beta      = market_snapshot.get("beta") or 1.0

    # Technical sentiment score (0-100): RSI centering + 52W range + MA signal
    _ma_score = {"bullish": 75, "bearish": 25, "mixed": 50, "unknown": 50}
    score_parts = []
    if rsi is not None:
        score_parts.append(100 - abs(rsi - 50) * 2)
    if range_pos is not None:
        score_parts.append(range_pos)
    score_parts.append(_ma_score.get(ma_signal, 50))
    technical_sentiment_score = round(sum(score_parts) / len(score_parts), 1) if score_parts else 50

    # Momentum state
    if rsi is not None and rsi > 60 and ma_signal == "bullish":
        momentum_state = "strong"
    elif rsi is not None and (rsi < 40 or ma_signal == "bearish"):
        momentum_state = "weak"
    else:
        momentum_state = "neutral"

    # Valuation pressure
    pe_t = valuation.get("pe_trailing")
    pe_f = valuation.get("pe_forward")
    peg  = valuation.get("peg")
    if (pe_t and pe_t > 60) or (pe_f and pe_f > 50) or (peg and peg > 3):
        valuation_pressure = "high"
    elif (pe_t and pe_t > 30) or (pe_f and pe_f > 25) or (peg and peg > 2):
        valuation_pressure = "medium"
    else:
        valuation_pressure = "low"

    # News intensity (count of high-relevance items)
    high_relevance = sum(1 for n in news_items if n.get("relevance_score", 0) >= 1.0)
    if high_relevance >= 5:
        news_intensity = "high"
    elif high_relevance >= 2:
        news_intensity = "medium"
    else:
        news_intensity = "low"

    # Risk flags — objective conditions, not conclusions
    risk_flags: list[str] = []
    if beta > 1.5:
        risk_flags.append(f"High volatility (Beta {beta:.1f})")
    if rsi is not None and rsi > 70:
        risk_flags.append(f"RSI overbought ({rsi}) — momentum reversal risk")
    if range_pos is not None and range_pos > 85:
        risk_flags.append(f"Near 52-week high ({range_pos}%) — limited upside buffer")
    if valuation_pressure == "high":
        risk_flags.append("Elevated valuation multiples — multiple compression risk")
    upside_pct = analyst.get("upside_pct")
    if upside_pct is not None and upside_pct < -5:
        risk_flags.append(f"Analyst target below current price ({upside_pct}% downside)")

    # Signal basis — enumerate the data points that drove the signals above
    signal_basis: list[str] = []
    if rsi is not None:
        signal_basis.append(f"RSI {rsi} ({technical.get('rsi_signal', '?')})")
    if ma_signal != "unknown":
        signal_basis.append(f"MA signal: {ma_signal}")
    vol = technical.get("volume_trend")
    if vol:
        signal_basis.append(f"Volume trend: {vol}")
    if range_pos is not None:
        signal_basis.append(f"52W range position: {range_pos}%")
    if upside_pct is not None:
        signal_basis.append(f"Analyst upside: {upside_pct}%")
    if news_intensity != "low":
        signal_basis.append(f"News intensity: {news_intensity} ({high_relevance} high-relevance items this week)")

    return {
        "as_of_date":               current_date,
        "technical_sentiment_score": technical_sentiment_score,
        "momentum_state":           momentum_state,
        "valuation_pressure":       valuation_pressure,
        "news_intensity":           news_intensity,
        "risk_flags":               risk_flags,
        "signal_basis":             signal_basis,
    }


# ── Slide Outline ─────────────────────────────────────────────────────────────

def build_slide_outline(duration_minutes: int) -> list:
    outline = [
        {
            "type": "cover",
            "approx_seconds": 20,
            "goal": "Introduce the ticker and the central question of the video.",
            "required_inputs": ["market_snapshot.ticker", "market_snapshot.company_name", "market_snapshot.current_price"],
        },
        {
            "type": "market_overview",
            "approx_seconds": 60,
            "goal": "Summarize current price, market cap, valuation and recent price movement.",
            "required_inputs": ["market_snapshot", "valuation_snapshot"],
        },
        {
            "type": "price_action",
            "approx_seconds": 70,
            "goal": "Explain recent price behavior using the 30-day chart and technical indicators.",
            "required_inputs": ["price_history", "technical_indicators"],
        },
        {
            "type": "key_points",
            "approx_seconds": 90,
            "goal": "Present the strongest bullish and bearish evidence this week.",
            "required_inputs": ["computed_signals", "news_evidence_pack"],
        },
    ]
    if duration_minutes >= 6:
        outline.append({
            "type": "financials",
            "approx_seconds": 80,
            "goal": "Explain whether fundamentals support the current valuation.",
            "required_inputs": ["financial_snapshot", "valuation_snapshot"],
        })
        outline.append({
            "type": "risk",
            "approx_seconds": 70,
            "goal": "Identify the main downside risks and uncertainty factors.",
            "required_inputs": ["computed_signals.risk_flags", "valuation_snapshot"],
        })
    if duration_minutes >= 8:
        outline.append({
            "type": "catalyst",
            "approx_seconds": 70,
            "goal": "Identify near-term events or narratives that may move the stock.",
            "required_inputs": ["news_evidence_pack", "analyst_snapshot"],
        })
    outline.append({
        "type": "outlook",
        "approx_seconds": 80,
        "goal": "Frame possible scenarios without forcing a buy/sell conclusion.",
        "required_inputs": ["technical_indicators", "financial_snapshot", "computed_signals"],
    })
    outline.append({
        "type": "summary",
        "approx_seconds": 40,
        "goal": "Summarize the key watchpoints and include the required disclaimer.",
        "required_inputs": ["computed_signals.risk_flags", "computed_signals.signal_basis"],
    })
    return outline


# ── Main Entry ────────────────────────────────────────────────────────────────

def run(ticker: str, market: str, language: str, duration_minutes: int, style: str) -> dict:
    current_date = date.today().isoformat()
    print(f"[Planner] Fetching market data for {ticker} (as of {current_date})...")

    t = yf.Ticker(ticker)
    raw          = fetch_market_data(ticker, t)
    company_name = raw["market_snapshot"].get("company_name", "")

    # Fetch news from both sources; each failure is non-fatal
    try:
        yf_news = fetch_weekly_news(t, ticker, company_name)
    except Exception as exc:
        print(f"[Planner] yfinance news failed (non-fatal): {exc}")
        yf_news = []

    api_key = os.environ.get("NEWS_API_KEY", "")
    if api_key:
        try:
            api_news = _fetch_newsapi(ticker, company_name, api_key)
        except Exception as exc:
            print(f"[Planner] NewsAPI failed (non-fatal): {exc}")
            api_news = []
    else:
        print("[Planner] NEWS_API_KEY not set — using yfinance news only.")
        api_news = []

    news_items = _merge_news(yf_news, api_news, max_items=20)

    ms   = raw["market_snapshot"]
    tech = raw["technical_indicators"]

    computed_signals = build_computed_signals(
        market_snapshot=ms,
        technical=tech,
        valuation=raw["valuation_snapshot"],
        analyst=raw["analyst_snapshot"],
        news_items=news_items,
        current_date=current_date,
    )
    slide_outline = build_slide_outline(duration_minutes)

    plan = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "current_date": current_date,
        "input": {
            "ticker":           ticker,
            "market":           market,
            "language":         language,
            "duration_minutes": duration_minutes,
            "style":            style,
        },
        "market_snapshot":   ms,
        "price_history":     {"period": "30d", "items": raw["price_history_items"]},
        "technical_indicators": tech,
        "valuation_snapshot":   raw["valuation_snapshot"],
        "financial_snapshot":   raw["financial_snapshot"],
        "analyst_snapshot":     raw["analyst_snapshot"],
        "news_evidence_pack": {
            "lookback_days": 7,
            "max_items":     15,
            "items":         news_items,
        },
        "computed_signals": computed_signals,
        "slide_outline":    slide_outline,
    }

    print(
        f"[Planner] Done. RSI={tech.get('rsi_14')} | MA20=${tech.get('ma20')} | "
        f"MA50=${tech.get('ma50')} | Momentum={computed_signals['momentum_state']} | "
        f"ValuationPressure={computed_signals['valuation_pressure']} | "
        f"{len(slide_outline)} slides"
    )
    return plan
