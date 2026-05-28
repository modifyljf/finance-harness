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

from harness import planner, generator, evaluator, renderer

OUTPUT_DIR = Path(__file__).parent / "output"


def save_json(data: dict, path: Path) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def save_text(text: str, path: Path) -> None:
    path.write_text(text, encoding="utf-8")


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
    save_json(gen_result["slides"], OUTPUT_DIR / "slides.json")

    # Stage 3: Evaluate
    if not args.skip_eval:
        eval_report = evaluator.run(plan, gen_result)
        save_json(eval_report, OUTPUT_DIR / "eval_report.json")

        if not eval_report["passed"]:
            print(f"\n[WARN] Quality check failed (score: {eval_report['score']}/100)")
            for issue in eval_report.get("issues", []):
                print(f"  - {issue}")
            print("Continuing to render anyway...\n")
    else:
        eval_report = {"score": -1, "passed": True, "skipped": True}

    # Stage 4: Render
    deck_path = OUTPUT_DIR / "deck.html"
    renderer.run(plan, gen_result["slides"], deck_path)

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
    print(f"  - output/narration.txt")
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
