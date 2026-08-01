#!/usr/bin/env python3
"""Derive a character range for a redaction box from physical measurements.

The infill pipeline (unredact.py) takes min_chars/max_chars. Use this tool to
compute them from a measured box width and the document's font size.
Assumes a monospace typewriter font (Courier advance width = 0.6 * font size,
e.g. 12pt Courier = 7.2pt per char = 10 CPI) -- the right default for 70s-era
government documents.

Usage:
  python box_measure.py --width-pts 50 --font-pt 12   # box width in PDF points
  python box_measure.py --width-in 0.7 --font-pt 10   # width in inches
  python box_measure.py --width-mm 18 --font-pt 12    # width in millimeters
  python box_measure.py --cpi 10 --width-pts 72       # if you know chars/inch
"""

import argparse
import json
import math

ADVANCE = 0.6  # Courier advance = 0.6 em


def main():
    ap = argparse.ArgumentParser(description="Convert a redaction box size to a character range.")
    ap.add_argument("--width-pts", type=float, help="Box width in PDF points (1pt = 1/72 inch)")
    ap.add_argument("--width-in", type=float, help="Box width in inches")
    ap.add_argument("--width-mm", type=float, help="Box width in millimeters")
    ap.add_argument("--font-pt", type=float, default=12.0,
                    help="Font size in points (default 12)")
    ap.add_argument("--cpi", type=float, default=None,
                    help="Chars per inch (overrides font-size math, e.g. 10 or 12)")
    ap.add_argument("--tolerance", type=float, default=0.08,
                    help="Fractional slack on the estimate (default 0.08)")
    args = ap.parse_args()

    if args.width_pts is None and args.width_in is None and args.width_mm is None:
        ap.error("provide one of --width-pts, --width-in, or --width-mm")

    if args.width_pts is not None:
        width_pts = args.width_pts
    elif args.width_in is not None:
        width_pts = args.width_in * 72.0
    else:
        width_pts = args.width_mm * 72.0 / 25.4

    if args.cpi:
        n = width_pts / 72.0 * args.cpi
        how = f"{args.cpi} CPI"
    else:
        per_char = ADVANCE * args.font_pt
        n = width_pts / per_char
        how = f"{args.font_pt}pt font"

    if width_pts <= 0 or n <= 0:
        ap.error(f"degenerate box width ({width_pts:.1f}pt) -- cannot compute a character range")

    lo = max(1, int(math.floor(n * (1 - args.tolerance))))
    hi = max(lo, int(math.ceil(n * (1 + args.tolerance))))

    print(f"box width {width_pts:.1f}pt @ {how} -> ~{n:.1f} characters")
    print(f"recommended: min_chars={lo}, max_chars={hi}")
    print("paste into redactions.json:")
    print(json.dumps({"min_chars": lo, "max_chars": hi}))


if __name__ == "__main__":
    main()
