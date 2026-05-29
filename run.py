#!/usr/bin/env python3
"""
AI Stock Review Video Harness
Usage: python run.py --ticker NVDA --duration 6 --language zh-CN
"""
import argparse
import json
import sys
import webbrowser
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()  # reads .env in project root, does nothing if file absent

from harness import planner, generator, evaluator, renderer

OUTPUT_DIR = Path(__file__).parent / "output"


def save_json(data: dict, path: Path) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def save_text(text: str, path: Path) -> None:
    path.write_text(text, encoding="utf-8")


def make_tts_narration(narration: str) -> str:
    """Strip slide markers and clean up for ElevenLabs TTS input."""
    import re
    # Remove [幻灯片: xxx] markers
    text = re.sub(r'\[幻灯片: \w+\]\n?', '', narration)
    # Collapse 3+ blank lines into 2 (natural paragraph pause)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="AI Stock Review Video Harness")
    parser.add_argument("--ticker", required=True, help="Stock ticker symbol (e.g. NVDA)")
    parser.add_argument("--duration", type=int, default=9, help="Video duration in minutes (default: 9)")
    parser.add_argument("--language", default="zh-CN", help="Output language (default: zh-CN)")
    parser.add_argument("--market", default="US", help="Market (default: US)")
    parser.add_argument("--style", default="professional", help="Style (default: professional)")
    parser.add_argument("--open", action="store_true", help="Open deck.html in browser after generation")
    parser.add_argument("--skip-eval", action="store_true", help="Skip LLM evaluation step")
    args = parser.parse_args()

    ticker = args.ticker.upper()
    OUTPUT_DIR.mkdir(exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  AI Stock Review — {ticker}")
    print(f"  Duration: {args.duration}min | Language: {args.language}")
    print(f"{'='*60}\n")

    # Stage 1: Plan
    plan = planner.run(
        ticker=ticker,
        market=args.market,
        language=args.language,
        duration_minutes=args.duration,
        style=args.style,
    )
    save_json(plan, OUTPUT_DIR / "plan.json")

    # Stage 2: Generate (multi-agent)
    gen_result = generator.run(plan)
    save_text(gen_result.get("fundamental_analysis", ""), OUTPUT_DIR / "analysis_fundamental.txt")
    save_text(gen_result.get("technical_analysis", ""), OUTPUT_DIR / "analysis_technical.txt")
    save_text(gen_result.get("narrative_analysis", ""), OUTPUT_DIR / "analysis_narrative.txt")
    save_text(gen_result["analysis"], OUTPUT_DIR / "analysis_synthesis.txt")
    save_text(gen_result["narration"], OUTPUT_DIR / "narration.txt")
    save_text(make_tts_narration(gen_result["narration"]), OUTPUT_DIR / "narration_tts.txt")

    # Stage 3: Evaluate — with targeted retry (max 2 attempts)
    MAX_RETRIES = 2
    eval_report = {"score": -1, "passed": True, "skipped": True}

    if not args.skip_eval:
        for attempt in range(MAX_RETRIES + 1):
            eval_report = evaluator.run(plan, gen_result)
            save_json(eval_report, OUTPUT_DIR / "eval_report.json")

            if eval_report["passed"]:
                break

            targets = eval_report.get("retry_targets", [])
            score   = eval_report["score"]

            print(f"\n[Eval] Score {score}/100 — FAILED")
            for issue in eval_report.get("issues", []):
                print(f"  - {issue}")

            if attempt == MAX_RETRIES:
                print(f"[Eval] Max retries reached. Rendering with current output.\n")
                break

            if not targets:
                print("[Eval] No retry targets identified — skipping retry.\n")
                break

            print(f"[Retry {attempt + 1}/{MAX_RETRIES}] Targets: {targets}\n")

            if "synthesis" in targets:
                gen_result["analysis"] = generator.regenerate_synthesis(plan, gen_result)
                save_text(gen_result["analysis"], OUTPUT_DIR / "analysis_synthesis.txt")

            if "narration" in targets:
                gen_result["narration"] = generator.regenerate_narration(plan, gen_result)
                save_text(gen_result["narration"], OUTPUT_DIR / "narration.txt")
                save_text(make_tts_narration(gen_result["narration"]), OUTPUT_DIR / "narration_tts.txt")

            if "slides" in targets:
                gen_result["slides"] = generator.regenerate_slides(plan, gen_result)
                # slides.json saved after renderer patches it

    # Stage 4: Render — renderer patches slides in-place (e.g. market_overview headline)
    deck_path = OUTPUT_DIR / "deck.html"
    renderer.run(plan, gen_result["slides"], deck_path)

    # Save slides.json AFTER renderer so the file reflects patched values
    save_json(gen_result["slides"], OUTPUT_DIR / "slides.json")

    # Metadata
    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ticker": ticker,
        "duration_minutes": args.duration,
        "language": args.language,
        "eval_score": eval_report.get("score"),
        "eval_passed": eval_report.get("passed"),
        "slide_count": len(gen_result["slides"].get("slides", [])),
    }
    save_json(metadata, OUTPUT_DIR / "metadata.json")

    print(f"\n{'='*60}")
    print("  Generated:")
    print(f"  - output/deck.html")
    print(f"  - output/slides.json")
    print(f"  - output/narration.txt        (含幻灯片标注，供对照)")
    print(f"  - output/narration_tts.txt    (纯净版，直接送 ElevenLabs)")
    print(f"  - output/analysis_fundamental.txt")
    print(f"  - output/analysis_technical.txt")
    print(f"  - output/analysis_narrative.txt")
    print(f"  - output/analysis_synthesis.txt")
    print(f"  - output/eval_report.json")
    print(f"  - output/metadata.json")
    if not args.skip_eval:
        print(f"\n  Quality Score: {eval_report.get('score', 'N/A')}/100")
    print(f"{'='*60}\n")

    if args.open:
        webbrowser.open(deck_path.resolve().as_uri())


if __name__ == "__main__":
    main()
