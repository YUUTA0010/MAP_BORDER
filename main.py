#!/usr/bin/env python3
"""
main.py - Entry point for the India border registration system.

Usage:
    python main.py                                        # default satellite
    python main.py --input input/satellite.png            # specific image
    python main.py --input img.png --reference ref.png    # custom reference
    python main.py --debug                                # save debug images
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from config import INPUT_DIR, OUTPUT_DIR, REFERENCE_PATH


def main():
    parser = argparse.ArgumentParser(
        description="India political border registration for satellite imagery"
    )
    parser.add_argument(
        "--input",
        default=os.path.join(INPUT_DIR, "satellite.png"),
        help="Path to satellite image (default: input/satellite.png)",
    )
    parser.add_argument(
        "--reference",
        default=REFERENCE_PATH,
        help="Path to reference India image (default: input/reference_india.png)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Save intermediate debug images to output/debug/",
    )

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.input):
        print(f"[main] ERROR: Satellite image not found: {args.input}")
        sys.exit(1)

    if not os.path.exists(args.reference):
        print(f"[main] ERROR: Reference image not found: {args.reference}")
        sys.exit(1)

    print("=" * 60)
    print("India Border Registration System")
    print("=" * 60)
    print(f"  Satellite: {args.input}")
    print(f"  Reference: {args.reference}")
    print(f"  Debug:     {args.debug}")
    print("=" * 60)

    from india_pipeline import run

    output_path, success, message = run(
        satellite_path=args.input,
        reference_path=args.reference,
        debug=args.debug,
    )

    print()
    if success:
        print(f"[main] SUCCESS: {message}")
        print(f"[main] Output: {output_path}")
    else:
        print(f"[main] FAILED: {message}")
        if output_path:
            print(f"[main] Unmodified image saved: {output_path}")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
