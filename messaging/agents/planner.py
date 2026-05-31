"""
PlannerAgent — Evidence Pack Builder.
Data engineering layer only: fetch, compute, classify. No investment conclusions.
"""
import os
import time
from datetime import date, datetime, timedelta, timezone

import requests
import yfinance as yf

from messaging.agents.base import BaseAgent
from messaging.dto.context import RunContext
from messaging.dto.hard_rules import NEWS_LOOKBACK_DAYS, NEWS_MAX_ITEMS, EXPERT_QUOTES_MAX


# ── Technical Indicators ──────────────────────────────────────────────────────

def _rsi(closes: list, period: int = 14) -> float | None:
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


def _ma(closes: list, period: int) -> float | None:
    if len(closes) < period:
        return None
    return round(sum(closes[-period:]) / period, 2)


def _bollinger(closes: list, period: int = 20) -> dict | None:
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


# ── News Classification ───────────────────────────────────────────────────────

_CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "earnings":    ["earnings", "revenue", "eps", "profit", "loss", "quarterly",
                    "q1", "q2", "q3", "q4", "fiscal", "guidance", "beat", "miss", "results", "income"],
    "product":     ["product", "launch", "chip", "gpu", "cpu", "model", "hardware",
                    "software", "release", "unveil", "announce", "platform"],
    "regulation":  ["regulation", "sec", "ftc", "antitrust", "ban", "sanction",
                    "compliance", "law", "government", "export", "restrict", "probe", "investigation"],
    "analyst":     ["analyst", "upgrade", "downgrade", "price target", "rating", "buy",
                    "sell", "hold", "coverage", "initiat", "reiterat", "overweight", "underweight"],
    "macro":       ["fed", "interest rate", "inflation", "gdp", "economy", "recession",
                    "market", "nasdaq", "s&p", "tariff", "trade", "treasury"],
    "insider":     ["insider", "ceo", "cfo", "executive", "board", "buyback",
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


def _classify_news(title: str, description: str = "") -> tuple[str, str]:
    text = (title + " " + description).lower()
    for cat, keywords in _CATEGORY_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            return cat, _CATEGORY_WHY[cat]
    return "other", _CATEGORY_WHY["other"]


# ── News Fetching ─────────────────────────────────────────────────────────────

def _build_keywords(ticker: str, company_name: str) -> list[str]:
    skip = {"corp", "corporation", "inc", "ltd", "llc", "co", "the", "group", "holdings"}
    words = [w.strip(".,").lower() for w in company_name.split()]
    extras = [w for w in words if len(w) > 3 and w not in skip]
    return list(dict.fromkeys([ticker.upper()] + extras))


def _fetch_yf_news(t: yf.Ticker, ticker: str, company_name: str) -> list[dict]:
    cutoff = time.time() - NEWS_LOOKBACK_DAYS * 24 * 3600
    keywords = _build_keywords(ticker, company_name)
    ticker_kw = keywords[0].lower()

    raw_news = []
    try:
        raw_news = t.news or []
    except Exception:
        pass

    scored = []
    for item in raw_news:
        content = item.get("content") or item
        title = content.get("title") or item.get("title") or ""
        description = content.get("summary") or content.get("description") or item.get("summary", "")
        publisher = (content.get("provider") or {}).get("displayName") or item.get("publisher", "")
        pub_ts = content.get("pubDate") or item.get("providerPublishTime")

        if isinstance(pub_ts, str):
            try:
                pub_dt = datetime.fromisoformat(pub_ts.replace("Z", "+00:00"))
                pub_unix, pub_str = pub_dt.timestamp(), pub_dt.strftime("%Y-%m-%d %H:%M")
            except ValueError:
                pub_unix, pub_str = 0, pub_ts[:16]
        elif isinstance(pub_ts, (int, float)):
            pub_unix = pub_ts
            pub_str = datetime.fromtimestamp(pub_ts).strftime("%Y-%m-%d %H:%M")
        else:
            continue

        if pub_unix < cutoff or not title:
            continue

        title_l, desc_l = title.lower(), description.lower()
        score, matched = 0, []
        if ticker_kw in title_l:
            score, matched = 3, [ticker_kw.upper()]
        else:
            for kw in keywords[1:]:
                if kw in title_l:
                    score, matched = 2, [kw]
                    break
        if score == 0:
            if ticker_kw in desc_l:
                score, matched = 1, [ticker_kw.upper()]
            else:
                for kw in keywords[1:]:
                    if kw in desc_l:
                        score, matched = 1, [kw]
                        break

        cat, why = _classify_news(title, description)
        scored.append({
            "title": title, "publisher": publisher, "published_at": pub_str,
            "url": "", "category": cat, "why_relevant": why,
            "matched_keywords": matched,
            "relevance_score": round(score / 3, 2),
            "expert_quotes": [],
            "_score": score, "_pub": pub_unix,
        })

    scored.sort(key=lambda x: (-x["_score"], -x["_pub"]))
    min_score = 1 if any(s["_score"] > 0 for s in scored) else 0
    seen: set[str] = set()
    out = []
    for r in scored:
        if r["_score"] < min_score or r["title"] in seen:
            continue
        seen.add(r["title"])
        out.append({k: v for k, v in r.items() if not k.startswith("_")})
        if len(out) >= 15:
            break

    high = sum(1 for u in out if u["relevance_score"] >= 1.0)
    print(f"[Planner/yfinance] {len(out)} news items ({high} high-relevance) for {ticker.upper()}")
    return out


def _fetch_newsapi(ticker: str, company_name: str, api_key: str) -> list[dict]:
    keywords = _build_keywords(ticker, company_name)
    ticker_kw = keywords[0]
    company_words = [k for k in keywords[1:] if len(k) >= 5][:2]
    q = f'{ticker_kw} OR "{" ".join(company_words)}"' if company_words else ticker_kw

    from_date = (date.today() - timedelta(days=NEWS_LOOKBACK_DAYS)).isoformat()

    try:
        resp = requests.get(
            "https://newsapi.org/v2/everything",
            params={"q": q, "from": from_date, "to": date.today().isoformat(),
                    "language": "en", "sortBy": "relevance", "pageSize": 30, "apiKey": api_key},
            timeout=12,
        )
        if resp.status_code == 429:
            print("[Planner/NewsAPI] Rate limit (429) — skipping.")
            return []
        if resp.status_code == 401:
            print("[Planner/NewsAPI] Invalid API key (401).")
            return []
        resp.raise_for_status()
        articles = resp.json().get("articles", [])
    except requests.exceptions.Timeout:
        print("[Planner/NewsAPI] Timeout — skipping.")
        return []
    except Exception as exc:
        print(f"[Planner/NewsAPI] Error: {exc}")
        return []

    seen: set[str] = set()
    out = []
    for article in articles:
        title = (article.get("title") or "").strip()
        description = article.get("description") or ""
        publisher = (article.get("source") or {}).get("name") or ""
        url = article.get("url") or ""
        pub_raw = article.get("publishedAt") or ""

        if not title or title == "[Removed]":
            continue

        try:
            pub_dt = datetime.fromisoformat(pub_raw.replace("Z", "+00:00"))
            pub_str = pub_dt.strftime("%Y-%m-%d %H:%M")
        except Exception:
            pub_str = pub_raw[:16]

        title_l, desc_l = title.lower(), description.lower()
        matched = []
        if ticker_kw.lower() in title_l:
            score = 1.0
            matched.append(ticker_kw.upper())
        else:
            hit = next((k for k in keywords[1:] if k.lower() in title_l), None)
            if hit:
                score, matched = 0.67, [hit]
            elif ticker_kw.lower() in desc_l:
                score, matched = 0.33, [ticker_kw.upper()]
            else:
                score = 0.1

        cat, why = _classify_news(title, description)
        if title not in seen:
            seen.add(title)
            out.append({
                "title": title, "publisher": publisher, "published_at": pub_str,
                "url": url, "category": cat, "why_relevant": why,
                "matched_keywords": matched, "relevance_score": round(score, 2),
                "expert_quotes": [],
                "_score": score,
            })

    out.sort(key=lambda x: -x["_score"])
    min_score = 0.33 if any(r["_score"] >= 0.33 for r in out) else 0
    filtered = [{k: v for k, v in r.items() if k != "_score"}
                for r in out if r["_score"] >= min_score][:15]

    high = sum(1 for r in filtered if r["relevance_score"] >= 1.0)
    print(f"[Planner/NewsAPI] {len(filtered)} articles ({high} high-relevance) for {ticker.upper()}")
    return filtered


def _merge_news(yf_items: list[dict], api_items: list[dict], max_items: int = NEWS_MAX_ITEMS) -> list[dict]:
    seen: set[str] = set()
    merged = []
    for item in yf_items + api_items:
        key = " ".join("".join(c for c in item["title"].lower() if c.isalnum() or c == " ").split())
        if key and key not in seen:
            seen.add(key)
            merged.append(item)
    merged.sort(key=lambda x: -x.get("relevance_score", 0))
    return merged[:max_items]


# ── Expert Quotes (Analyst Rating Changes) ────────────────────────────────────

_BULLISH_GRADES = {"buy", "overweight", "outperform", "strong buy", "positive", "accumulate"}
_BEARISH_GRADES = {"sell", "underweight", "underperform", "negative", "reduce"}


def _fetch_expert_quotes(t: yf.Ticker, lookback_days: int = NEWS_LOOKBACK_DAYS) -> list[dict]:
    """Extract recent analyst rating changes from yfinance upgrades_downgrades."""
    quotes = []
    try:
        df = t.upgrades_downgrades
        if df is None or df.empty:
            return []

        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
        recent = df[df.index >= cutoff]

        for dt, row in recent.iterrows():
            firm = str(row.get("Firm", "Unknown")).strip()
            to_grade = str(row.get("ToGrade", "")).strip()
            from_grade = str(row.get("FromGrade", "")).strip()
            action = str(row.get("Action", "")).strip()

            stance = "neutral"
            if to_grade.lower() in _BULLISH_GRADES:
                stance = "bullish"
            elif to_grade.lower() in _BEARISH_GRADES:
                stance = "bearish"

            if from_grade and from_grade.lower() != "nan":
                quote_text = f"{firm} changed rating from {from_grade} to {to_grade}"
                context = f"{action} on {dt.strftime('%Y-%m-%d')}"
            else:
                quote_text = f"{firm} initiated coverage with {to_grade} rating"
                context = f"Initiated on {dt.strftime('%Y-%m-%d')}"

            quotes.append({
                "source": firm,
                "quote": quote_text,
                "stance": stance,
                "context": context,
                "date": dt.strftime("%Y-%m-%d"),
            })
    except Exception as exc:
        print(f"[Planner] expert_quotes fetch failed (non-fatal): {exc}")

    return quotes[:EXPERT_QUOTES_MAX]


# ── Market Data ───────────────────────────────────────────────────────────────

def _fetch_market_data(ticker: str, t: yf.Ticker) -> dict:
    info = t.info
    hist_90 = t.history(period="100d").dropna()
    hist_30 = hist_90.tail(30)

    closes_90 = [round(float(c), 2) for c in hist_90["Close"].tolist()]
    volumes_90 = [int(v) for v in hist_90["Volume"].tolist()]
    price_history_items = [
        {"date": str(d.date()), "close": round(float(c), 2)}
        for d, c in zip(hist_30.index, hist_30["Close"])
    ]

    current_price = (info.get("currentPrice") or info.get("regularMarketPrice")
                     or (closes_90[-1] if closes_90 else None))
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

    rsi_val    = _rsi(closes_90)
    ma20_val   = _ma(closes_90, 20)
    ma50_val   = _ma(closes_90, 50)
    boll       = _bollinger(closes_90, 20)
    vol_trend  = _volume_trend(volumes_90)
    high52     = info.get("fiftyTwoWeekHigh")
    low52      = info.get("fiftyTwoWeekLow")
    range_pos  = _range_position(current_price, low52, high52)

    rsi_signal = ("overbought" if rsi_val is not None and rsi_val >= 70
                  else "oversold" if rsi_val is not None and rsi_val <= 30
                  else "neutral" if rsi_val is not None else "unknown")

    ma_signal = "unknown"
    if current_price and ma20_val and ma50_val:
        if current_price > ma20_val > ma50_val:
            ma_signal = "bullish"
        elif current_price < ma20_val < ma50_val:
            ma_signal = "bearish"
        else:
            ma_signal = "mixed"

    target_mean   = info.get("targetMeanPrice")
    target_high   = info.get("targetHighPrice")
    target_low    = info.get("targetLowPrice")
    analyst_count = info.get("numberOfAnalystOpinions")
    recommendation = info.get("recommendationKey", "").replace("_", " ").title()
    upside_pct = (round((target_mean - current_price) / current_price * 100, 1)
                  if target_mean and current_price else None)

    short_float       = info.get("shortPercentOfFloat")
    institutional_pct = info.get("heldPercentInstitutions")
    pe_trailing       = info.get("trailingPE")
    pe_forward        = info.get("forwardPE")
    peg               = info.get("pegRatio")
    ps_ratio          = info.get("priceToSalesTrailing12Months")
    pb_ratio          = info.get("priceToBook")
    ev_ebitda         = info.get("enterpriseToEbitda")
    revenue           = info.get("totalRevenue")
    revenue_growth    = info.get("revenueGrowth")
    earnings_growth   = info.get("earningsGrowth")
    gross_margin      = info.get("grossMargins")
    operating_margin  = info.get("operatingMargins")
    profit_margin     = info.get("profitMargins")
    free_cashflow     = info.get("freeCashflow")
    roe               = info.get("returnOnEquity")

    return {
        "price_history_items": price_history_items,
        "market_snapshot": {
            "ticker":           ticker.upper(),
            "company_name":     info.get("longName", ticker.upper()),
            "sector":           info.get("sector", ""),
            "industry":         info.get("industry", ""),
            "short_description": (info.get("longBusinessSummary") or "")[:400],
            "current_price":    current_price,
            "prev_close":       prev_close,
            "price_change":     price_change,
            "price_change_pct": price_change_pct,
            "market_cap":       market_cap,
            "market_cap_str":   market_cap_str,
            "volume":           info.get("volume") or info.get("regularMarketVolume"),
            "avg_volume":       info.get("averageVolume"),
            "beta":             info.get("beta"),
            "52w_high":         high52,
            "52w_low":          low52,
            "52w_range_position_pct": range_pos,
            "short_interest":   round(short_float * 100, 1) if short_float else None,
            "institutional_ownership_pct": round(institutional_pct * 100, 1) if institutional_pct else None,
        },
        "technical_indicators": {
            "rsi_14":            rsi_val,
            "rsi_signal":        rsi_signal,
            "ma20":              ma20_val,
            "ma50":              ma50_val,
            "bollinger":         boll,
            "volume_trend":      vol_trend,
            "ma_signal":         ma_signal,
            "price_vs_ma20_pct": round((current_price - ma20_val) / ma20_val * 100, 1) if current_price and ma20_val else None,
            "price_vs_ma50_pct": round((current_price - ma50_val) / ma50_val * 100, 1) if current_price and ma50_val else None,
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
            "revenue":             revenue,
            "revenue_growth_yoy":  round(revenue_growth   * 100, 1) if revenue_growth   else None,
            "earnings_growth_yoy": round(earnings_growth  * 100, 1) if earnings_growth  else None,
            "gross_margin":        round(gross_margin      * 100, 1) if gross_margin     else None,
            "operating_margin":    round(operating_margin  * 100, 1) if operating_margin else None,
            "profit_margin":       round(profit_margin     * 100, 1) if profit_margin    else None,
            "free_cashflow":       free_cashflow,
            "roe":                 round(roe * 100, 1) if roe else None,
            "eps_trailing":        info.get("trailingEps"),
            "eps_forward":         info.get("forwardEps"),
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


# ── Earnings Snapshot ─────────────────────────────────────────────────────────

def _fetch_earnings_snapshot(t: yf.Ticker) -> dict:
    """Fetch most recent quarterly earnings data from yfinance. Non-fatal on any error."""
    snap: dict = {
        "last_quarter":      None,
        "revenue":           None,
        "revenue_qoq_pct":   None,
        "gross_profit":      None,
        "gross_margin_pct":  None,
        "net_income":        None,
        "operating_cashflow": None,
        "free_cashflow":     None,
        "eps_actual":        None,
        "eps_estimate":      None,
        "eps_surprise_pct":  None,
        "next_earnings_date": None,
    }

    def _safe_int(val):
        try:
            f = float(val)
            return None if f != f else int(f)
        except Exception:
            return None

    def _safe_float(val, decimals: int = 2):
        try:
            f = float(val)
            return None if f != f else round(f, decimals)
        except Exception:
            return None

    def _find_row(df, keywords: list[str]) -> str | None:
        for name in df.index:
            name_l = name.lower()
            if all(kw in name_l for kw in keywords):
                return name
        return None

    # Quarterly income statement
    try:
        qi = t.quarterly_income_stmt
        if qi is not None and not qi.empty:
            col0 = qi.columns[0]
            snap["last_quarter"] = str(col0.date()) if hasattr(col0, "date") else str(col0)[:10]

            row = _find_row(qi, ["total", "revenue"])
            if row:
                snap["revenue"] = _safe_int(qi.loc[row, col0])
                if len(qi.columns) >= 2:
                    prev = _safe_int(qi.loc[row, qi.columns[1]])
                    if snap["revenue"] and prev:
                        snap["revenue_qoq_pct"] = round((snap["revenue"] - prev) / abs(prev) * 100, 1)

            row = _find_row(qi, ["gross", "profit"])
            if row:
                snap["gross_profit"] = _safe_int(qi.loc[row, col0])
                if snap["revenue"] and snap["gross_profit"]:
                    snap["gross_margin_pct"] = round(snap["gross_profit"] / snap["revenue"] * 100, 1)

            row = _find_row(qi, ["net", "income"])
            if row:
                snap["net_income"] = _safe_int(qi.loc[row, col0])
    except Exception as exc:
        print(f"[Planner] quarterly_income_stmt failed (non-fatal): {exc}")

    # Quarterly cash flow
    try:
        qc = t.quarterly_cashflow
        if qc is not None and not qc.empty:
            col0 = qc.columns[0]

            row = _find_row(qc, ["operating"])
            if row:
                snap["operating_cashflow"] = _safe_int(qc.loc[row, col0])

            row = _find_row(qc, ["free"])
            if row:
                snap["free_cashflow"] = _safe_int(qc.loc[row, col0])
            elif snap["operating_cashflow"] is not None:
                capex_row = _find_row(qc, ["capital", "expenditure"])
                if capex_row:
                    capex = _safe_int(qc.loc[capex_row, col0])
                    if capex is not None:
                        snap["free_cashflow"] = snap["operating_cashflow"] + capex
    except Exception as exc:
        print(f"[Planner] quarterly_cashflow failed (non-fatal): {exc}")

    # EPS actual vs estimate + next earnings date
    try:
        ed = t.earnings_dates
        if ed is not None and not ed.empty:
            today = date.today()
            ed = ed.copy()
            ed.index = [i.date() if hasattr(i, "date") else i for i in ed.index]

            future = ed[ed.index > today]
            past   = ed[ed.index <= today]

            if not future.empty:
                snap["next_earnings_date"] = str(future.index.min())

            if not past.empty:
                row = past.iloc[0]
                snap["eps_actual"]   = _safe_float(row.get("Reported EPS"))
                snap["eps_estimate"] = _safe_float(row.get("EPS Estimate"))
                if snap["eps_actual"] is not None and snap["eps_estimate"]:
                    snap["eps_surprise_pct"] = round(
                        (snap["eps_actual"] - snap["eps_estimate"]) / abs(snap["eps_estimate"]) * 100, 1
                    )
    except Exception as exc:
        print(f"[Planner] earnings_dates failed (non-fatal): {exc}")

    return snap


# ── Analyst Forward Estimates ─────────────────────────────────────────────────

def _fetch_analyst_estimates(t: yf.Ticker) -> dict:
    """Fetch forward analyst EPS/Revenue estimates and growth rates. Non-fatal."""
    result: dict = {
        "earnings_estimates": [],
        "revenue_estimates":  [],
        "five_year_growth_pct":    None,
        "next_year_eps_growth_pct": None,
        "next_year_rev_growth_pct": None,
    }

    def _safe_val(val):
        try:
            f = float(val)
            return None if f != f else f
        except Exception:
            return None

    try:
        ee = t.earnings_estimates
        if ee is not None and not ee.empty:
            for period in ee.index:
                row = ee.loc[period]
                result["earnings_estimates"].append({
                    "period":         str(period),
                    "avg_eps":        _safe_val(row.get("Avg")),
                    "low_eps":        _safe_val(row.get("Low")),
                    "high_eps":       _safe_val(row.get("High")),
                    "analyst_count":  int(row.get("No. of Analysts", 0) or 0),
                    "year_ago_eps":   _safe_val(row.get("Year Ago EPS")),
                })
    except Exception as exc:
        print(f"[Planner] earnings_estimates failed (non-fatal): {exc}")

    try:
        re_ = t.revenue_estimates
        if re_ is not None and not re_.empty:
            for period in re_.index:
                row = re_.loc[period]
                avg_rev = _safe_val(row.get("Avg"))
                yago    = _safe_val(row.get("Year Ago Revenue"))
                growth  = round((avg_rev - yago) / abs(yago) * 100, 1) if avg_rev and yago else None
                result["revenue_estimates"].append({
                    "period":          str(period),
                    "avg_revenue":     int(avg_rev) if avg_rev else None,
                    "low_revenue":     int(_safe_val(row.get("Low")) or 0) or None,
                    "high_revenue":    int(_safe_val(row.get("High")) or 0) or None,
                    "analyst_count":   int(row.get("No. of Analysts", 0) or 0),
                    "year_ago_revenue": int(yago) if yago else None,
                    "implied_yoy_growth_pct": growth,
                })
    except Exception as exc:
        print(f"[Planner] revenue_estimates failed (non-fatal): {exc}")

    try:
        ge = t.growth_estimates
        if ge is not None and not ge.empty:
            ticker_col = ge.columns[0]
            def _g(label):
                for idx in ge.index:
                    if label.lower() in str(idx).lower():
                        return _safe_val(ge.loc[idx, ticker_col])
                return None
            five_yr = _g("next 5 years")
            if five_yr is not None:
                result["five_year_growth_pct"] = round(five_yr * 100, 1)
            next_yr_eps = _g("next year")
            if next_yr_eps is not None:
                result["next_year_eps_growth_pct"] = round(next_yr_eps * 100, 1)
    except Exception as exc:
        print(f"[Planner] growth_estimates failed (non-fatal): {exc}")

    # Derive next-year revenue growth from estimates
    annual = [e for e in result["revenue_estimates"] if e["period"] in ("+1Y", "1Y", "Next Year")]
    if annual:
        result["next_year_rev_growth_pct"] = annual[0].get("implied_yoy_growth_pct")

    return result


# ── Computed Signals ──────────────────────────────────────────────────────────

def _build_computed_signals(
    market_snapshot: dict,
    technical: dict,
    valuation: dict,
    analyst: dict,
    news_items: list[dict],
    current_date: str,
) -> dict:
    rsi       = technical.get("rsi_14")
    range_pos = market_snapshot.get("52w_range_position_pct")
    ma_signal = technical.get("ma_signal", "unknown")
    beta      = market_snapshot.get("beta") or 1.0

    _ma_score = {"bullish": 75, "bearish": 25, "mixed": 50, "unknown": 50}
    parts = []
    if rsi is not None:
        parts.append(100 - abs(rsi - 50) * 2)
    if range_pos is not None:
        parts.append(range_pos)
    parts.append(_ma_score.get(ma_signal, 50))
    technical_sentiment_score = round(sum(parts) / len(parts), 1) if parts else 50

    if rsi is not None and rsi > 60 and ma_signal == "bullish":
        momentum_state = "strong"
    elif rsi is not None and (rsi < 40 or ma_signal == "bearish"):
        momentum_state = "weak"
    else:
        momentum_state = "neutral"

    pe_t = valuation.get("pe_trailing")
    pe_f = valuation.get("pe_forward")
    peg  = valuation.get("peg")
    if (pe_t and pe_t > 60) or (pe_f and pe_f > 50) or (peg and peg > 3):
        valuation_pressure = "high"
    elif (pe_t and pe_t > 30) or (pe_f and pe_f > 25) or (peg and peg > 2):
        valuation_pressure = "medium"
    else:
        valuation_pressure = "low"

    high_rel = sum(1 for n in news_items if n.get("relevance_score", 0) >= 1.0)
    news_intensity = "high" if high_rel >= 5 else "medium" if high_rel >= 2 else "low"

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
        signal_basis.append(f"News intensity: {news_intensity} ({high_rel} high-relevance items)")

    return {
        "as_of_date":                current_date,
        "technical_sentiment_score": technical_sentiment_score,
        "momentum_state":            momentum_state,
        "valuation_pressure":        valuation_pressure,
        "news_intensity":            news_intensity,
        "risk_flags":                risk_flags,
        "signal_basis":              signal_basis,
    }


# ── Slide Outline ─────────────────────────────────────────────────────────────

def _build_slide_outline(duration_minutes: int) -> list[dict]:
    outline = [
        {"type": "cover",          "approx_seconds": 20, "goal": "Introduce the ticker and the central question of the video.",               "required_inputs": ["market_snapshot.ticker", "market_snapshot.company_name", "market_snapshot.current_price"]},
        {"type": "market_overview","approx_seconds": 60, "goal": "Summarize current price, market cap, valuation and recent price movement.", "required_inputs": ["market_snapshot", "valuation_snapshot"]},
        {"type": "price_action",   "approx_seconds": 70, "goal": "Explain recent price behavior using the 30-day chart and technical indicators.", "required_inputs": ["price_history", "technical_indicators"]},
        {"type": "key_points",     "approx_seconds": 90, "goal": "Present the strongest bullish and bearish evidence this week.",              "required_inputs": ["computed_signals", "news_evidence_pack"]},
        {"type": "news",           "approx_seconds": 60, "goal": "Present this week's key news headlines and analyst rating changes with attribution.", "required_inputs": ["news_evidence_pack.items", "news_evidence_pack.expert_quotes"]},
    ]
    if duration_minutes >= 6:
        outline += [
            {"type": "financials", "approx_seconds": 80, "goal": "Explain whether fundamentals support the current valuation.", "required_inputs": ["financial_snapshot", "valuation_snapshot"]},
            {"type": "risk",       "approx_seconds": 70, "goal": "Identify the main downside risks and uncertainty factors.",   "required_inputs": ["computed_signals.risk_flags", "valuation_snapshot"]},
        ]
    if duration_minutes >= 8:
        outline.append(
            {"type": "catalyst",   "approx_seconds": 70, "goal": "Identify near-term events or narratives that may move the stock.", "required_inputs": ["news_evidence_pack", "analyst_snapshot"]},
        )
    outline += [
        {"type": "outlook", "approx_seconds": 80, "goal": "Frame possible scenarios without forcing a buy/sell conclusion.",        "required_inputs": ["technical_indicators", "financial_snapshot", "computed_signals"]},
        {"type": "summary", "approx_seconds": 40, "goal": "Summarize the key watchpoints and include the required disclaimer.",     "required_inputs": ["computed_signals.risk_flags", "computed_signals.signal_basis"]},
    ]
    return outline


# ── PlannerAgent ──────────────────────────────────────────────────────────────

class PlannerAgent(BaseAgent):

    def run(self, ctx: RunContext) -> dict:
        current_date = date.today().isoformat()
        ticker = ctx.ticker
        print(f"[Planner] Fetching market data for {ticker} (as of {current_date})...")

        t = yf.Ticker(ticker)
        raw = _fetch_market_data(ticker, t)
        company_name = raw["market_snapshot"].get("company_name", "")

        # News: yfinance + NewsAPI (both non-fatal)
        yf_news = []
        try:
            yf_news = _fetch_yf_news(t, ticker, company_name)
        except Exception as exc:
            print(f"[Planner] yfinance news failed (non-fatal): {exc}")

        api_news = []
        api_key = os.environ.get("NEWS_API_KEY", "")
        if api_key:
            try:
                api_news = _fetch_newsapi(ticker, company_name, api_key)
            except Exception as exc:
                print(f"[Planner] NewsAPI failed (non-fatal): {exc}")
        else:
            print("[Planner] NEWS_API_KEY not set — using yfinance news only.")

        news_items = _merge_news(yf_news, api_news)

        # Expert quotes: recent analyst rating changes
        expert_quotes = []
        try:
            expert_quotes = _fetch_expert_quotes(t)
            if expert_quotes:
                print(f"[Planner] {len(expert_quotes)} analyst rating changes fetched.")
        except Exception as exc:
            print(f"[Planner] expert_quotes skipped (non-fatal): {exc}")

        # Quarterly earnings snapshot
        earnings_snapshot = {}
        try:
            earnings_snapshot = _fetch_earnings_snapshot(t)
            print(
                f"[Planner] Earnings: Q={earnings_snapshot.get('last_quarter')} | "
                f"Rev={earnings_snapshot.get('revenue')} | "
                f"EPS actual={earnings_snapshot.get('eps_actual')} estimate={earnings_snapshot.get('eps_estimate')} "
                f"surprise={earnings_snapshot.get('eps_surprise_pct')}% | "
                f"Next={earnings_snapshot.get('next_earnings_date')}"
            )
        except Exception as exc:
            print(f"[Planner] earnings_snapshot skipped (non-fatal): {exc}")

        # Analyst forward estimates
        analyst_estimates = {}
        try:
            analyst_estimates = _fetch_analyst_estimates(t)
            print(
                f"[Planner] Analyst estimates: "
                f"{len(analyst_estimates.get('earnings_estimates', []))} EPS periods | "
                f"{len(analyst_estimates.get('revenue_estimates', []))} Rev periods | "
                f"5Y growth={analyst_estimates.get('five_year_growth_pct')}%"
            )
        except Exception as exc:
            print(f"[Planner] analyst_estimates skipped (non-fatal): {exc}")

        ms   = raw["market_snapshot"]
        tech = raw["technical_indicators"]

        computed_signals = _build_computed_signals(
            market_snapshot=ms,
            technical=tech,
            valuation=raw["valuation_snapshot"],
            analyst=raw["analyst_snapshot"],
            news_items=news_items,
            current_date=current_date,
        )
        slide_outline = _build_slide_outline(ctx.duration_minutes)

        plan = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "current_date": current_date,
            "input": {
                "ticker":           ticker,
                "market":           ctx.market,
                "language":         ctx.language,
                "duration_minutes": ctx.duration_minutes,
                "style":            ctx.style,
            },
            "market_snapshot":    ms,
            "price_history":      {"period": "30d", "items": raw["price_history_items"]},
            "technical_indicators": tech,
            "valuation_snapshot":   raw["valuation_snapshot"],
            "financial_snapshot":   raw["financial_snapshot"],
            "earnings_snapshot":    earnings_snapshot,
            "analyst_estimates":    analyst_estimates,
            "analyst_snapshot":     raw["analyst_snapshot"],
            "news_evidence_pack": {
                "lookback_days":  NEWS_LOOKBACK_DAYS,
                "max_items":      NEWS_MAX_ITEMS,
                "items":          news_items,
                "expert_quotes":  expert_quotes,
            },
            "computed_signals":  computed_signals,
            "slide_outline":     slide_outline,
        }

        print(
            f"[Planner] Done. RSI={tech.get('rsi_14')} | MA20=${tech.get('ma20')} | "
            f"MA50=${tech.get('ma50')} | Momentum={computed_signals['momentum_state']} | "
            f"Valuation={computed_signals['valuation_pressure']} | "
            f"{len(slide_outline)} slides | {len(expert_quotes)} analyst quotes"
        )
        return plan
