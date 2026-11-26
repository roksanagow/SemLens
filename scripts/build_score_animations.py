"""Make the short animations shown in the (?) help of each score, from hand-drawn panels.

The drawings in ``scripts/score_drawings`` hold four panels, each a finished
picture: ``amd_samd.png`` (AMD, SAMD) and ``prt_apd.png`` (PRT, APD). Blue and
orange circles are the usages of two corpora; ``usages.png`` is the same circles
alone, the first frame of every animation (lined up with each panel), since the
lines drawn over a panel's circles break them. The other layers of each picture
are told apart by colour, and shown one after another:

- AMD: the usages; then each blue usage's closest orange one (green); then each
  orange usage's closest blue one (red), each with its line of the legend.
- SAMD: the usages; then the one-to-one pairs (black).
- PRT: the usages; then each corpus's average (the crosses); then the distance
  between them (black). The crosses share the circles' colours, so they are found
  as what the PRT panel has beyond the clean circles.
- APD: the usages; then every pair (black).

Writes ``app/data/score_help/<score>.gif``. Run with the project's Python environment:

    .venv/bin/python scripts/build_score_animations.py
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DRAWINGS = ROOT / "scripts" / "score_drawings"
OUTPUT = ROOT / "app" / "data" / "score_help"
SIZE = 260          # pixels of the square animation
BORDER = 14         # the thick frame of each panel, from its outer edge
PANELS = {"amd": ("amd_samd.png", 0, 360), "samd": ("amd_samd.png", 360, 717),
          "prt": ("prt_apd.png", 0, 345), "apd": ("prt_apd.png", 345, 689),
          # The usages alone, drawn once without any lines: the first frame of every animation.
          "usages": ("usages.png", 0, 337)}
# Milliseconds each frame stays on screen; the finished picture stays longest.
TIMING = {2: [700, 1600], 3: [700, 900, 1600]}


def _load(name):
    image = Image.open(DRAWINGS / name)
    white = Image.new("RGBA", image.size, (255, 255, 255, 255))
    white.alpha_composite(image.convert("RGBA"))
    return np.asarray(white.convert("RGB")).astype(float) / 255


def _crop(name):
    """A panel cut at the outer edge of its thick frame."""
    file, x0, x1 = PANELS[name]
    a = _load(file)
    dark = a[:, x0:x1 + 1].max(axis=2) < 0.35
    cols = np.where(dark.sum(0) > 150)[0]
    rows = np.where(dark.sum(1) > 200)[0]
    return a[rows.min():rows.max() + 1, x0 + cols.min():x0 + cols.max() + 1]


def _classes(a):
    """Each pixel's ink: white, black (lines and frame), blue, orange, green or red."""
    top, bottom = a.max(axis=2), a.min(axis=2)
    saturation = np.where(top > 0, (top - bottom) / np.maximum(top, 1e-6), 0)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    spread = np.maximum(top - bottom, 1e-6)
    hue = np.where(top == r, ((g - b) / spread) % 6, np.where(top == g, (b - r) / spread + 2, (r - g) / spread + 4)) * 60
    # Dark mixtures, where a black line crosses a coloured stroke, count as the line.
    coloured = (saturation > 0.18) & (top > 0.5)
    ink = np.full(top.shape, "white", dtype=object)
    ink[~coloured & (top < 0.93)] = "black"
    ink[coloured & (hue >= 190) & (hue < 260)] = "blue"
    ink[coloured & (hue >= 18) & (hue < 55)] = "orange"
    ink[coloured & (hue >= 70) & (hue < 170)] = "green"
    ink[coloured & ((hue >= 330) | (hue < 18))] = "red"
    # Pale grey edges of a coloured stroke belong to that stroke, not to the black lines.
    pale = (ink == "black") & (top > 0.6)
    for colour in ("blue", "orange", "green", "red"):
        near = _grow(ink == colour, 1)
        ink[pale & near] = colour
    return ink


def _grow(mask, steps):
    out = mask.copy()
    for _ in range(steps):
        grown = out.copy()
        grown[1:] |= out[:-1]
        grown[:-1] |= out[1:]
        grown[:, 1:] |= out[:, :-1]
        grown[:, :-1] |= out[:, 1:]
        out = grown
    return out


def _drop_specks(mask, minimum=15):
    """Leave out pieces smaller than a drawn circle: where a red and a green line overlap, the blend looks orange."""
    from scipy import ndimage
    labels, count = ndimage.label(mask, structure=np.ones((3, 3)))
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    keep = sizes >= minimum
    keep[0] = False
    return keep[labels]


def _frame_band(shape):
    band = np.zeros(shape, dtype=bool)
    band[:BORDER], band[-BORDER:], band[:, :BORDER], band[:, -BORDER:] = True, True, True, True
    return band


def _paint(base, a, mask):
    out = base.copy()
    out[mask] = a[mask]
    return out


def _shift(mask, dy, dx):
    out = np.zeros_like(mask)
    h, w = mask.shape
    out[max(dy, 0):h + min(dy, 0), max(dx, 0):w + min(dx, 0)] = mask[max(-dy, 0):h + min(-dy, 0), max(-dx, 0):w + min(-dx, 0)]
    return out


def _resized(mask, shape):
    return np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize(shape[::-1], Image.NEAREST)) > 0


def _shift_image(a, dy, dx):
    out = np.ones_like(a)
    h, w = a.shape[:2]
    out[max(dy, 0):h + min(dy, 0), max(dx, 0):w + min(dx, 0)] = a[max(-dy, 0):h + min(-dy, 0), max(-dx, 0):w + min(-dx, 0)]
    return out


def _clean_usages(shape, points):
    """The clean drawing of the usages, scaled to a panel and lined up with its circles: (pixels, mask of the circles)."""
    clean = _crop("usages")
    image = np.asarray(Image.fromarray((clean * 255).astype(np.uint8)).resize(shape[::-1], Image.LANCZOS)).astype(float) / 255
    ink = _classes(image)
    mask = _drop_specks((ink == "blue") | (ink == "orange"))
    best = max(((int((_grow(_shift(mask, dy, dx), 1) & points).sum()), dy, dx) for dy in range(-12, 13) for dx in range(-12, 13)))
    return _shift_image(image, best[1], best[2]), _shift(mask, best[1], best[2])


def _frames(name):
    a = _crop(name)
    ink = _classes(a)
    white = np.ones_like(a)
    frame = (ink == "black") & _frame_band(ink.shape)
    points = (ink == "blue") | (ink == "orange")
    lines = (ink == "black") & ~frame
    h, w = ink.shape
    legend = np.zeros(ink.shape, dtype=bool)
    if name == "amd":
        legend[int(h * 0.74):, :int(w * 0.56)] = True
    # The first frame: the panel's own frame, with the clean drawing of the usages in place of its circles,
    # which the lines (or crosses) drawn over them have broken.
    clean, circles = _clean_usages(ink.shape, points & ~legend)
    base = _paint(_paint(white, a, frame), clean, circles)
    if name == "amd":
        first_row = legend.copy()
        first_row[int(h * 0.855):] = False
        second_row = legend & ~first_row
        green = (ink == "green") | (points & first_row)
        red = (ink == "red") | (points & second_row)
        one = _paint(base, a, green)
        return [base, one, _paint(one, a, red)]
    if name == "prt":
        # What PRT has in blue and orange beyond the circles are the crosses.
        # Small pieces are the edges of circles where the two drawings differ by a pixel, not crosses.
        crosses = _drop_specks(points & ~_grow(circles, 1), 25)
        # The crosses were drawn over the circles: near a cross, the panel's own pixels (cross on top) replace the clean circles.
        crosses |= points & _grow(crosses, 3)
        marked = _paint(base, a, crosses)
        return [base, marked, _paint(marked, a, lines)]
    return [base, _paint(base, a, lines)]


def _gif(frames, path):
    images = [Image.fromarray((f * 255).astype(np.uint8)).resize((SIZE, SIZE), Image.LANCZOS) for f in frames]
    # One palette for every frame, taken from all of them, so that colours neither flicker nor fade.
    strip = Image.new("RGB", (SIZE, SIZE * len(images)))
    for k, image in enumerate(images):
        strip.paste(image, (0, k * SIZE))
    palette = strip.quantize(colors=48, method=Image.Quantize.MEDIANCUT)
    frames = [image.quantize(palette=palette, dither=Image.Dither.NONE) for image in images]
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=TIMING[len(frames)], loop=0, optimize=True)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for name in ("amd", "samd", "prt", "apd"):
        path = OUTPUT / f"{name}.gif"
        _gif(_frames(name), path)
        print(f"{path.relative_to(ROOT)}: {path.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
