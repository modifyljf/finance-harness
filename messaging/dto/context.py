from dataclasses import dataclass, field


@dataclass
class RunContext:
    ticker: str
    market: str = "US"
    language: str = "zh-CN"
    duration_minutes: int = 9
    style: str = "professional"
    skip_eval: bool = False
    max_eval_retries: int = 2

    def __post_init__(self):
        self.ticker = self.ticker.upper()
