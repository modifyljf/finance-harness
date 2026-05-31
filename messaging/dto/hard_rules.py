"""Hard constraints that must never be violated across the pipeline."""

CHARS_PER_MINUTE_ZH = 331        # calibrated: AVGO 767s actual playback
CHARS_PER_MINUTE_EN = 140
TTS_CHARS_PER_SEC_ZH = 5.52     # 331 / 60, calibrated from 12:47 actual playback
TTS_PAUSE_PER_BREAK_SEC = 1.0   # fish.audio 1s pause per paragraph break (\n\n)
TTS_SLIDE_BUFFER_MS = 900        # Reveal.js transition overhead per slide

MAX_NARRATION_MINUTES = 15   # hard ceiling regardless of target duration

NEWS_LOOKBACK_DAYS = 7
NEWS_MAX_ITEMS = 20
EXPERT_QUOTES_MAX = 10

EVAL_PASS_THRESHOLD = 80
MAX_EVAL_RETRIES = 2

REQUIRED_SLIDE_TYPES = frozenset({
    "cover", "market_overview", "ai_winner_loser", "bull_case", "bear_case",
    "narrative_score", "outlook", "summary"
})

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
MODEL_REASONER = "deepseek-v4-pro"    # deep analysis: fundamental, narrative, key slide narrations
MODEL_CHAT = "deepseek-v4-flash"      # fast calls: slides JSON, synthesis, YouTube meta, smooth pass
