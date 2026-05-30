import os
from pathlib import Path

from openai import OpenAI

from messaging.dto.hard_rules import DEEPSEEK_BASE_URL, MODEL_REASONER, MODEL_CHAT

PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts"


class BaseAgent:
    def __init__(self, client: OpenAI | None = None):
        self.client = client or self._make_client()

    @staticmethod
    def _make_client() -> OpenAI:
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise EnvironmentError("DEEPSEEK_API_KEY environment variable not set.")
        return OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)

    def _load_prompt(self, name: str) -> str:
        return (PROMPTS_DIR / f"{name}.txt").read_text(encoding="utf-8")

    def _stream(self, system: str, user: str, model: str = MODEL_REASONER) -> str:
        stream = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=8000,
            stream=True,
        )
        parts = []
        for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                parts.append(delta.content)
        return "".join(parts)

    def _chat(self, system: str, user: str, json_mode: bool = False, model: str = MODEL_CHAT) -> str:
        kwargs = dict(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=8000,
        )
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content
