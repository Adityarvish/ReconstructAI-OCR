
import argparse
import logging
import sys
from typing import Optional
from dotenv import load_dotenv
load_dotenv()
from src import config
from src.pipeline import run_pipeline, PipelineError


def configure_logging() -> None:
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Intelligent Image OCR Extraction and Quality Enhancement System"
    )
    parser.add_argument(
        "image_path",
        nargs="?",
        default=config.DEFAULT_INPUT_IMAGE,
        help=f"Path to the input image (default: {config.DEFAULT_INPUT_IMAGE})",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Skip showing the original/processed crop preview windows at the end.",
    )
    parser.add_argument(
        "--roi",
        type=str,
        default=None,
        metavar="x,y,w,h",
        help=(
            "Region to crop, as 'x,y,width,height' in pixels. Skips the "
            "interactive cv2.selectROI() GUI window entirely - required "
            "on headless machines/remote sessions/containers where no "
            "display is available (selectROI() there either errors out "
            "or hangs, which otherwise looks like the whole system "
            "silently failing). Use e.g. --roi 5,133,205,10."
        ),
    )
    return parser.parse_args()


def _parse_roi(roi_str: Optional[str]):
    if roi_str is None:
        return None
    parts = roi_str.split(",")
    if len(parts) != 4:
        print(f"\n[ERROR] --roi must be 'x,y,w,h' (got '{roi_str}')\n", file=sys.stderr)
        sys.exit(1)
    try:
        x, y, w, h = (int(p.strip()) for p in parts)
    except ValueError:
        print(f"\n[ERROR] --roi values must be integers (got '{roi_str}')\n", file=sys.stderr)
        sys.exit(1)
    return (x, y, w, h)


def main() -> int:
    configure_logging()
    args = parse_args()
    roi = _parse_roi(args.roi)

    try:
        run_pipeline(args.image_path, show_windows=not args.no_preview, roi=roi)
        return 0
    except PipelineError as exc:
        print(f"\n[ERROR] {exc}\n", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n\nCancelled by user.\n")
        return 1
    except Exception as exc:  
        logging.getLogger(__name__).exception("Unexpected error")
        print(f"\n[UNEXPECTED ERROR] {exc}\n", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
