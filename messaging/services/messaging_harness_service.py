"""
MessagingHarnessService — the only place that knows the pipeline order.
Planner → Generator → Evaluator (with retry) → Renderer
"""
import json
from datetime import datetime, timezone
from pathlib import Path

from messaging.agents.evaluator import EvaluatorAgent
from messaging.agents.generator import GeneratorAgent
from messaging.agents.planner import PlannerAgent
from messaging.agents.renderer import RendererAgent
from messaging.dto.candidate import Candidate
from messaging.dto.context import RunContext
from messaging.dto.evaluation import Evaluation
from messaging.dto.rendered_message import RenderedMessage

_BASE_OUTPUT = Path(__file__).parent.parent.parent / "output"


class MessagingHarnessService:

    def __init__(self):
        self._planner = PlannerAgent()
        self._generator = GeneratorAgent()
        self._evaluator = EvaluatorAgent()
        self._renderer = RendererAgent()

    def execute(self, ctx: RunContext) -> RenderedMessage:
        # Each run gets its own folder: output/{TICKER}/
        from datetime import datetime, timezone
        out_dir = _BASE_OUTPUT / ctx.ticker
        out_dir.mkdir(parents=True, exist_ok=True)
        self._out_dir = out_dir

        # Stage 1: Plan
        plan = self._planner.run(ctx)
        self._save_json(plan, "plan.json")

        # Stage 2: Generate
        candidate = self._generator.run(plan)
        self._save_candidate(candidate)

        # Stage 3: Evaluate + targeted retry
        eval_result = Evaluation(score=-1, passed=True, skipped=True)

        if not ctx.skip_eval:
            for attempt in range(ctx.max_eval_retries + 1):
                eval_result = self._evaluator.run(plan, candidate)
                self._save_json(eval_result.to_dict(), "eval_report.json")

                if eval_result.passed:
                    break

                print(f"\n[Eval] Score {eval_result.score}/100 — FAILED")
                for issue in eval_result.issues:
                    print(f"  - {issue}")

                if attempt == ctx.max_eval_retries:
                    print("[Eval] Max retries reached. Rendering with current output.")
                    break

                if not eval_result.retry_targets:
                    print("[Eval] No retry targets — skipping retry.")
                    break

                print(f"[Retry {attempt + 1}/{ctx.max_eval_retries}] Targets: {eval_result.retry_targets}")
                candidate = self._generator.regenerate(plan, candidate, eval_result.retry_targets)
                self._save_candidate(candidate)

        # Stage 4: Render
        deck_html = self._renderer.run(plan, candidate)
        deck_path = self._out_dir / "deck.html"
        deck_path.write_text(deck_html, encoding="utf-8")
        print(f"[Renderer] Saved: {deck_path}")

        # Save slides AFTER renderer patches market_overview headline
        self._save_json(candidate.slides, "slides.json")

        # Stage 5: YouTube metadata + thumbnail
        youtube_meta = self._generator.generate_youtube_meta(plan, candidate.synthesis)
        self._save_json(youtube_meta, "youtube_meta.json")

        thumbnail_html = self._renderer.generate_thumbnail(plan, youtube_meta)
        (self._out_dir / "thumbnail.html").write_text(thumbnail_html, encoding="utf-8")

        metadata = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "ticker": ctx.ticker,
            "duration_minutes": ctx.duration_minutes,
            "language": ctx.language,
            "eval_score": eval_result.score,
            "eval_passed": eval_result.passed,
            "slide_count": len(candidate.slides.get("slides", [])),
        }
        self._save_json(metadata, "metadata.json")
        print(f"[Service] All output saved to: {self._out_dir}")

        return RenderedMessage(
            deck_html=deck_html,
            slides=candidate.slides,
            narration=candidate.narration,
            narration_tts=candidate.narration_tts,
            youtube_meta=youtube_meta,
            thumbnail_html=thumbnail_html,
            metadata=metadata,
        )

    def _save_json(self, data: dict, filename: str) -> None:
        (self._out_dir / filename).write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _save_candidate(self, candidate: Candidate) -> None:
        texts = {
            "analysis_fundamental.txt": candidate.fundamental_analysis,
            "analysis_technical.txt":   candidate.technical_analysis,
            "analysis_narrative.txt":   candidate.narrative_analysis,
            "analysis_synthesis.txt":   candidate.synthesis,
            "narration.txt":            candidate.narration,
            "narration_tts.txt":        candidate.narration_tts,
        }
        for filename, text in texts.items():
            (self._out_dir / filename).write_text(text, encoding="utf-8")
