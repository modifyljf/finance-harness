import os
import time
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

    def _stream(self, system: str, user: str, model: str = MODEL_REASONER, _retry: int = 3) -> str:
        for attempt in range(_retry):
            try:
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
                result = "".join(parts)
                if result.strip():
                    return result
                print(f"[Base] _stream empty response (attempt {attempt+1}/{_retry})")
            except Exception as exc:
                print(f"[Base] _stream error (attempt {attempt+1}/{_retry}): {type(exc).__name__}: {exc}")
                if attempt < _retry - 1:
                    time.sleep(2 ** attempt)  # 1s, 2s backoff
        return ""

    def _chat(self, system: str, user: str, json_mode: bool = False, model: str = MODEL_CHAT, _retry: int = 3) -> str:
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
        for attempt in range(_retry):
            try:
                response = self.client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content
                if content and content.strip():
                    return content
                print(f"[Base] _chat empty response (attempt {attempt+1}/{_retry})")
            except Exception as exc:
                print(f"[Base] _chat error (attempt {attempt+1}/{_retry}): {type(exc).__name__}: {exc}")
                if attempt < _retry - 1:
                    time.sleep(2 ** attempt)
        return ""
