from dataclasses import dataclass, field


@dataclass
class Candidate:
    fundamental_analysis: str
    technical_analysis: str
    narrative_analysis: str
    synthesis: str
    narration: str
    narration_tts: str
    slides: dict
    attempt: int = 0
