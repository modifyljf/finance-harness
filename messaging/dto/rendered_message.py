from dataclasses import dataclass, field


@dataclass
class RenderedMessage:
    deck_html: str
    slides: dict
    narration: str
    narration_tts: str
    youtube_meta: dict
    metadata: dict
