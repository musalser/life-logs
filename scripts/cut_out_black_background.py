"""Cut a subject off a black studio background into a PNG with alpha.

`process_landing_images.py` uses rembg (a segmentation network) for the landing
photographs. That is the right tool when the background is *not* known, but it is
a heavy dependency, and these plates are shot against pure black: the background
is a constant we can measure instead of guess.

The trick that makes it look clean is the last step. A pixel on the edge of the
statue is not "marble or background", it is a mix of the two:

    observed = subject * alpha + background * (1 - alpha)

Solving for `subject` (un-premultiplying against the measured background) removes
the dark halo that a plain luminance key leaves around every edge — the fringe
that becomes visible the moment the picture is composited over something that is
not black.

Usage:
    python scripts/cut_out_black_background.py IMAGE [IMAGE ...] [--out PATH]
    python scripts/cut_out_black_background.py frontend/public/landing/forge-statue.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage


def border_background(rgb: np.ndarray, width: int = 6) -> np.ndarray:
    """The colour of the background, estimated from the frame of the picture."""
    frame = np.concatenate([
        rgb[:width, :, :].reshape(-1, 3),
        rgb[-width:, :, :].reshape(-1, 3),
        rgb[:, :width, :].reshape(-1, 3),
        rgb[:, -width:, :].reshape(-1, 3),
    ])
    return np.median(frame, axis=0)


def cut_out(
    image: Image.Image,
    low: float = 10.0,
    high: float = 34.0,
    feather: float = 0.6,
    despeckle: int = 48,
) -> Image.Image:
    rgb = np.asarray(image.convert("RGB")).astype(np.float32)

    # Luminance, not the mean: the background is neutral black, so the strongest
    # channel is the most sensitive signal for "is there anything here at all".
    luminance = rgb.max(axis=2)
    background = border_background(rgb)

    # A plain luminance threshold is wrong here, and visibly so: this statue is
    # dark, so most of its body would fall below the threshold and the picture
    # would composite as a ghost. The background is not "dark pixels", it is the
    # dark region *connected to the frame* — a dark fold inside the marble is not
    # background, and a black gap between the arm and the head is. So: seed with
    # the darkest pixels, keep only the components that reach the border.
    seed = luminance <= low
    labels, count = ndimage.label(seed)
    if count:
        reaches_border = np.zeros(count + 1, dtype=bool)
        reaches_border[labels[0, :]] = True
        reaches_border[labels[-1, :]] = True
        reaches_border[labels[:, 0]] = True
        reaches_border[labels[:, -1]] = True
        reaches_border[0] = False
        outside = reaches_border[labels]
    else:
        outside = seed

    # Everything the background does not reach is the subject, solid — including
    # its shadows. Only the two-pixel ring along the edge is allowed a ramp, which
    # is where anti-aliasing and the black the picture was shot against mix in.
    alpha = np.where(outside, 0.0, 1.0)
    ring = ndimage.binary_dilation(outside, iterations=2) & ~outside
    ramp = np.clip((luminance - low) / (high - low), 0.0, 1.0)
    alpha = np.where(ring, ramp, alpha)

    # Drop specks of sensor noise that survived in the background.
    if despeckle:
        labels, count = ndimage.label(alpha > 0.35)
        if count:
            sizes = ndimage.sum(np.ones_like(labels), labels, range(1, count + 1))
            keep = np.isin(labels, np.nonzero(sizes >= despeckle)[0] + 1)
            alpha = np.where(keep, alpha, 0.0)
    if feather:
        alpha = ndimage.gaussian_filter(alpha, feather)

    # Un-premultiply against the measured background: the edge pixels recover
    # their own colour instead of carrying the black they were shot against.
    safe = np.maximum(alpha, 1e-3)[..., None]
    subject = (rgb - background * (1.0 - alpha)[..., None]) / safe
    subject = np.clip(subject, 0, 255)

    out = np.dstack([subject, alpha * 255.0]).astype(np.uint8)
    return Image.fromarray(out, "RGBA")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, help="write here instead of in place")
    parser.add_argument("--low", type=float, default=12.0, help="alpha starts above this luminance")
    parser.add_argument("--high", type=float, default=40.0, help="alpha is solid above this luminance")
    parser.add_argument("--feather", type=float, default=0.8, help="matte blur in pixels")
    args = parser.parse_args()

    if args.out and len(args.images) > 1:
        parser.error("--out makes sense for a single image")

    for source in args.images:
        result = cut_out(Image.open(source), low=args.low, high=args.high, feather=args.feather)
        target = args.out or source
        result.save(target, "PNG", optimize=True)
        alpha = np.asarray(result.getchannel("A"))
        print(
            f"{source.name} -> {target.name}: {result.width}x{result.height}, "
            f"{target.stat().st_size // 1024} KB, "
            f"transparent {np.mean(alpha < 5):.0%}, solid {np.mean(alpha > 250):.0%}"
        )


if __name__ == "__main__":
    main()
