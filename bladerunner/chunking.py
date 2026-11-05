"""Token-bounded image chunking (spec Section 2.1 / 2.3).

The Orchestrator/Inspector must enforce image chunking so a single inference
request stays within the model's context window. We estimate per-image token
overhead (a rough proxy for a vision model: fixed overhead + pixel-scaled
component) and batch images until either the image count or token budget is hit.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence


@dataclass
class Chunk:
    index: int
    images: List[Path]
    estimated_tokens: int

    def __len__(self) -> int:
        return len(self.images)


def estimate_image_tokens(image_path: Path, approx_pixels_per_token: int = 256) -> int:
    """Estimate the token cost of one image.

    Uses a base overhead plus a pixel-based component. If the image cannot be
    opened (e.g. unavailable at estimate time), a conservative constant is used.
    """
    base = 64
    try:
        from PIL import Image

        with Image.open(image_path) as im:
            pixels = im.size[0] * im.size[1]
    except Exception:
        pixels = 256 * 256
    pixel_tokens = max(1, pixels // approx_pixels_per_token)
    # Cap the per-image contribution so a single huge image cannot blow the budget.
    return base + min(pixel_tokens, 1280)


def chunk_images(
    images: Sequence[Path],
    images_per_chunk: int = 6,
    max_chunk_tokens: int = 8000,
) -> List[Chunk]:
    """Split ``images`` into chunks bounded by count and estimated token budget."""
    if not images:
        return []
    images_per_chunk = max(1, int(images_per_chunk))
    chunks: List[Chunk] = []
    current: List[Path] = []
    current_tokens = 0
    idx = 0
    for img in images:
        cost = estimate_image_tokens(img)
        if current and (len(current) >= images_per_chunk or current_tokens + cost > max_chunk_tokens):
            chunks.append(Chunk(idx, current, current_tokens))
            current, current_tokens = [], 0
            idx += 1
        current.append(img)
        current_tokens += cost
    if current:
        chunks.append(Chunk(idx, current, current_tokens))
    return chunks
