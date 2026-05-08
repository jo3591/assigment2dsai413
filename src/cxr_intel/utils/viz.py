"""Visualization helpers for retrieved evidence and ColPali heatmaps."""
from __future__ import annotations

from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def overlay_heatmap(
    image: Image.Image,
    score_grid: np.ndarray,
    alpha: float = 0.5,
    colormap: str = "jet",
) -> Image.Image:
    """Reproject a (H_p, W_p) score grid onto the image as a heatmap."""
    import matplotlib.cm as cm

    cmap = cm.get_cmap(colormap)
    grid = score_grid.astype(np.float32)
    grid = (grid - grid.min()) / (grid.max() - grid.min() + 1e-8)
    rgba = (cmap(grid) * 255).astype(np.uint8)
    heat = Image.fromarray(rgba).convert("RGBA").resize(image.size, Image.BILINEAR)

    base = image.convert("RGBA")
    blended = Image.blend(base, heat, alpha=alpha)
    return blended


def make_grid(images: Sequence[Image.Image], titles: Sequence[str] | None = None,
              cell: int = 256) -> Image.Image:
    n = len(images)
    if n == 0:
        return Image.new("RGB", (cell, cell), "white")
    grid = Image.new("RGB", (cell * n, cell + 24), "white")
    draw = ImageDraw.Draw(grid)
    try:
        font = ImageFont.load_default()
    except Exception:  # pragma: no cover
        font = None
    for i, img in enumerate(images):
        thumb = img.copy()
        thumb.thumbnail((cell, cell))
        x = i * cell + (cell - thumb.width) // 2
        y = 24 + (cell - thumb.height) // 2
        grid.paste(thumb, (x, y))
        if titles and i < len(titles) and font is not None:
            draw.text((i * cell + 4, 4), titles[i][:32], fill="black", font=font)
    return grid
