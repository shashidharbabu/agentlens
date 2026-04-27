"""Generates simple synthetic sample images so run_demo.py works
immediately on a fresh clone. The user can drop real images in
data/input_images/ with matching filenames to override these.
"""
from __future__ import annotations

import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter


def _gradient(size, top_rgb, bottom_rgb):
    w, h = size
    img = Image.new("RGB", size, top_rgb)
    draw = ImageDraw.Draw(img)
    for y in range(h):
        t = y / max(h - 1, 1)
        r = int(top_rgb[0] * (1 - t) + bottom_rgb[0] * t)
        g = int(top_rgb[1] * (1 - t) + bottom_rgb[1] * t)
        b = int(top_rgb[2] * (1 - t) + bottom_rgb[2] * t)
        draw.line([(0, y), (w, y)], fill=(r, g, b))
    return img


def _street_scene(path: Path) -> None:
    img = _gradient((768, 512), (120, 150, 200), (60, 70, 100))
    draw = ImageDraw.Draw(img)
    # buildings
    for x in range(0, 768, 90):
        h = random.randint(160, 330)
        draw.rectangle([x + 10, 512 - h, x + 80, 512], fill=(40, 45, 60))
        # windows
        for wy in range(512 - h + 20, 512 - 20, 30):
            for wx in range(x + 20, x + 75, 20):
                if random.random() > 0.3:
                    draw.rectangle([wx, wy, wx + 10, wy + 15], fill=(220, 200, 120))
    # street
    draw.rectangle([0, 490, 768, 512], fill=(30, 30, 35))
    # car
    draw.rounded_rectangle([300, 470, 420, 505], 8, fill=(180, 40, 40))
    img.save(path, "JPEG", quality=92)


def _portrait(path: Path) -> None:
    img = _gradient((512, 640), (240, 220, 200), (190, 170, 150))
    draw = ImageDraw.Draw(img)
    # face
    draw.ellipse([150, 120, 380, 380], fill=(225, 195, 170))
    # hair
    draw.chord([140, 80, 390, 250], 180, 360, fill=(60, 40, 30))
    # eyes
    draw.ellipse([200, 220, 230, 245], fill=(50, 50, 70))
    draw.ellipse([300, 220, 330, 245], fill=(50, 50, 70))
    # mouth
    draw.arc([230, 290, 300, 335], 0, 180, fill=(150, 70, 70), width=4)
    # body
    draw.rectangle([170, 380, 360, 640], fill=(70, 90, 140))
    img.save(path, "JPEG", quality=92)


def _landscape(path: Path) -> None:
    img = _gradient((768, 512), (160, 180, 210), (220, 200, 150))
    draw = ImageDraw.Draw(img)
    # mountains
    draw.polygon([(0, 380), (200, 200), (400, 330), (600, 180), (768, 360), (768, 512), (0, 512)],
                 fill=(90, 100, 110))
    # water
    draw.rectangle([0, 360, 768, 450], fill=(110, 140, 160))
    # ground
    draw.rectangle([0, 450, 768, 512], fill=(80, 110, 70))
    # sun
    draw.ellipse([560, 80, 660, 180], fill=(255, 230, 150))
    img = img.filter(ImageFilter.GaussianBlur(0.6))
    img.save(path, "JPEG", quality=92)


def ensure_sample_images(input_dir: Path) -> dict:
    """Create sample images if they don't already exist. Returns a
    dict of {key: path} suitable for feeding into run_demo.py."""
    input_dir.mkdir(parents=True, exist_ok=True)
    random.seed(7)

    paths = {
        "street_scene": input_dir / "street_scene.jpg",
        "portrait": input_dir / "portrait.jpg",
        "landscape": input_dir / "landscape.jpg",
    }

    if not paths["street_scene"].exists():
        _street_scene(paths["street_scene"])
    if not paths["portrait"].exists():
        _portrait(paths["portrait"])
    if not paths["landscape"].exists():
        _landscape(paths["landscape"])

    return paths


if __name__ == "__main__":
    from pathlib import Path as _P

    out = ensure_sample_images(_P("data/input_images"))
    for k, p in out.items():
        print(f"{k}: {p}")
