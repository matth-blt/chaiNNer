"""Best Tiles node - Extract multiple best tiles from images in a sequence."""

from __future__ import annotations

from collections.abc import Iterable
from enum import Enum

import cv2
import numpy as np

from api import IteratorInputInfo, IteratorOutputInfo, Transformer
from nodes.groups import if_enum_group
from nodes.properties.inputs import EnumInput, ImageInput, NumberInput, SliderInput
from nodes.properties.outputs import ImageOutput, NumberOutput

from . import tile_group


class ComplexityMethod(Enum):
    LAPLACIAN = "laplacian"
    IC9600 = "ic9600"


@tile_group.register(
    schema_id="chainner:pepedp:best_tiles",
    name="Best Tiles",
    description=[
        "Extract multiple high-quality tiles from each image in a sequence.",
        "For each image, extracts all tiles that meet the complexity threshold.",
        "Large images will produce more tiles than small images (dynamic_n_tiles mode).",
        "Based on PepeDP by umzi2.",
    ],
    icon="BsGrid3X3",
    kind="transformer",
    inputs=[
        ImageInput().with_id(0),
        NumberInput("Tile Size", default=512, min=64, max=2048, unit="px").with_id(1),
        EnumInput(
            ComplexityMethod, "Complexity Method", default=ComplexityMethod.LAPLACIAN
        )
        .with_id(2)
        .with_docs(
            "**Laplacian**: Fast CPU-based method using Laplacian variance. Good for detecting edges and details.",
            "**IC9600**: Neural network-based complexity assessment. More accurate but requires GPU and is slower.",
        ),
        SliderInput(
            "Threshold",
            min=0,
            max=1,
            default=0.3,
            precision=2,
            step=0.01,
        )
        .with_id(3)
        .with_docs(
            "Minimum complexity score required. Tiles below this threshold are silently discarded.",
            "Set to 0 to extract all possible tiles.",
        ),
        if_enum_group(2, ComplexityMethod.LAPLACIAN)(
            NumberInput("Median Blur", default=5, min=1, max=21)
            .with_id(4)
            .with_docs(
                "Kernel size for median blur preprocessing. Higher values reduce noise but may blur fine details.",
                "Must be an odd number (will be adjusted if even).",
            ),
        ),
    ],
    outputs=[
        ImageOutput().with_id(0),
        NumberOutput("Complexity Score", output_type="0..1")
        .with_id(1)
        .with_docs(
            "The complexity score of the extracted tile. Higher values indicate more detail/information."
        ),
    ],
    iterator_inputs=IteratorInputInfo(inputs=[0], length_type="uint"),
    iterator_outputs=IteratorOutputInfo(outputs=[0, 1], length_type="uint"),
)
def best_tiles_node(
    _: None,
    tile_size: int,
    method: ComplexityMethod,
    threshold: float,
    median_blur: int,
) -> Transformer[np.ndarray, tuple[np.ndarray, float]]:
    """Extract multiple tiles from each image that meet the threshold."""
    from pepedp.scripts.utils.complexity.laplacian import LaplacianComplexity
    from pepeline import best_tile

    # Create the complexity function once (reused for all images)
    if method == ComplexityMethod.LAPLACIAN:
        complexity_fn = LaplacianComplexity(median_blur=median_blur)
    else:
        from pepedp.scripts.utils.complexity.ic9600 import IC9600Complexity

        complexity_fn = IC9600Complexity()

    def on_iterate(img: np.ndarray) -> Iterable[tuple[np.ndarray, float]]:
        h, w = img.shape[:2]
        channels = img.shape[2] if len(img.shape) == 3 else 1

        # Convert BGR -> RGB for PepeDP
        if channels == 3:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = img

        # If image is smaller than tile size, treat it as a single tile
        if h < tile_size or w < tile_size:
            complexity = complexity_fn(img_rgb)
            score = float(np.mean(complexity))
            if score >= threshold:
                yield (img, score)
            return

        # If image is exactly tile size, return it with its score
        if h == tile_size and w == tile_size:
            complexity = complexity_fn(img_rgb)
            score = float(np.mean(complexity))
            if score >= threshold:
                yield (img, score)
            return

        # Calculate the complexity map
        complexity = complexity_fn(img_rgb)

        # Calculate max number of tiles (dynamic_n_tiles formula from PepeDP)
        # n_tiles = (H*W) / (tile_size^2 * 2)
        max_tiles = (h * w) // (tile_size * tile_size * 2)
        max_tiles = max(1, max_tiles)

        for _ in range(max_tiles):
            # Find the best remaining position
            y, x = best_tile(complexity, tile_size)

            # Extract the tile and its score
            # Note: get_tile_comp_score modifies complexity in-place (sets region to -1)
            tile_rgb, complexity, score = complexity_fn.get_tile_comp_score(
                img_rgb, complexity, y, x, tile_size
            )

            # Check threshold
            if score < threshold:
                # Stop extracting tiles once we go below threshold
                break

            # Convert RGB -> BGR for chaiNNer
            if channels == 3:
                tile_bgr = cv2.cvtColor(tile_rgb, cv2.COLOR_RGB2BGR)
            else:
                tile_bgr = tile_rgb

            yield (tile_bgr, float(score))

    return Transformer(on_iterate=on_iterate)
