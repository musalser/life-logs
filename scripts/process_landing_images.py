"""Convert JFIF landing assets to PNG with background removal.

Full-bleed backdrops (temple-backdrop, temple-doors) are converted to JPG
without background removal, per plan.md image spec.
"""

from pathlib import Path

from PIL import Image
from rembg import new_session, remove

LANDING_DIR = Path(__file__).resolve().parent.parent / "frontend" / "public" / "landing"
FULL_BLEED = {"temple-backdrop", "temple-doors", "stairs-ascent"}

session = new_session("isnet-general-use")

for src in sorted(LANDING_DIR.glob("*.jfif")):
    img = Image.open(src)
    if src.stem in FULL_BLEED:
        out = src.with_suffix(".jpg")
        img.convert("RGB").save(out, "JPEG", quality=90)
    else:
        out = src.with_suffix(".png")
        result = remove(img, session=session)
        result.save(out, "PNG")
    print(f"{src.name} -> {out.name} ({out.stat().st_size // 1024} KB)")

print("Done.")
