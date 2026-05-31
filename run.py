#!/usr/bin/env python3
"""
AI Stock Review Video Harness
Usage: python3 run.py --ticker NVDA --duration 12 --language zh-CN
Usage: source .venv/bin/activate && python run.py --ticker NVDA --duration 12 --language zh-CN
"""
import argparse
import webbrowser
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from messaging.dto.context import RunContext
from messaging.services.messaging_harness_service import MessagingHarnessService

OUTPUT_DIR = Path(__file__).parent / "output"


def main() -> None:
    parser = argparse.ArgumentParser(description="AI Stock Review Video Harness")
    parser.add_argument("--ticker",   required=True,  help="Stock ticker (e.g. NVDA)")
    parser.add_argument("--duration", type=int, default=12, help="Video duration in minutes (default: 12)")
    parser.add_argument("--language", default="zh-CN", help="Output language (default: zh-CN)")
    parser.add_argument("--market",   default="US",    help="Market (default: US)")
    parser.add_argument("--style",    default="professional", help="Style (default: professional)")
    parser.add_argument("--open",     action="store_true", help="Open deck.html after generation")
    parser.add_argument("--skip-eval", action="store_true", help="Skip LLM evaluation step")
    args = parser.parse_args()

    ctx = RunContext(
        ticker=args.ticker,
        market=args.market,
        language=args.language,
        duration_minutes=args.duration,
        style=args.style,
        skip_eval=args.skip_eval,
    )

    print(f"\n{'='*60}")
    print(f"  AI Stock Review — {ctx.ticker}")
    print(f"  Duration: {ctx.duration_minutes}min | Language: {ctx.language}")
    print(f"{'='*60}\n")

    service = MessagingHarnessService()
    result = service.execute(ctx)

    folder = f"output/{ctx.ticker}/"
    print(f"\n{'='*60}")
    print(f"  Generated in {folder}:")
    print(f"  - {folder}deck.html")
    print(f"  - {folder}slides.json")
    print(f"  - {folder}narration.txt        (含幻灯片标注，供对照)")
    print(f"  - {folder}narration_tts.txt    (纯净版，用于 autoslide 时序计算)")
    print(f"  - {folder}narration_tts_emotion.txt (情绪标注版，发送给 fish.audio)")
    print(f"  - {folder}analysis_fundamental.txt")
    print(f"  - {folder}analysis_technical.txt")
    print(f"  - {folder}analysis_narrative.txt")
    print(f"  - {folder}analysis_synthesis.txt")
    print(f"  - {folder}eval_report.json")
    print(f"  - {folder}youtube_meta.json")
    print(f"  - {folder}thumbnail.html      (1280×720 封面图，浏览器截图上传)")
    print(f"  - {folder}metadata.json")
    if not args.skip_eval:
        print(f"\n  Quality Score: {result.metadata.get('eval_score', 'N/A')}/100")
    print(f"{'='*60}\n")

    if args.open:
        deck = (OUTPUT_DIR / ctx.ticker / "deck.html").resolve()
        webbrowser.open(deck.as_uri())


if __name__ == "__main__":
    main()
