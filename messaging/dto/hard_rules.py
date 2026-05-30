"""Hard constraints that must never be violated across the pipeline."""

CHARS_PER_MINUTE_ZH = 342   # measured from fish.audio TTS output
CHARS_PER_MINUTE_EN = 140
TTS_CHARS_PER_SEC_ZH = 5.70  # 342 / 60, used for per-slide autoslide timing

MAX_NARRATION_MINUTES = 15   # hard ceiling regardless of target duration

NEWS_LOOKBACK_DAYS = 7
NEWS_MAX_ITEMS = 20
EXPERT_QUOTES_MAX = 10

EVAL_PASS_THRESHOLD = 70
MAX_EVAL_RETRIES = 2

REQUIRED_SLIDE_TYPES = frozenset({
    "cover", "market_overview", "price_action", "key_points", "news", "outlook", "summary"
})

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
MODEL_REASONER = "deepseek-v4-pro"    # deep analysis: fundamental, narrative, key slide narrations
MODEL_CHAT = "deepseek-v4-flash"      # fast calls: slides JSON, synthesis, YouTube meta, smooth pass
