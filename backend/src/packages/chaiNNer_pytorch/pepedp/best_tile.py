"""Best Tile node - Extract the single best tile from an image."""

from __future__ import annotations

from enum import Enum

import cv2
import numpy as np

from nodes.groups import if_enum_group
from nodes.properties.inputs import EnumInput, ImageInput, NumberInput, SliderInput
from nodes.properties.outputs import ImageOutput, NumberOutput

from . import tile_group


class ComplexityMethod(Enum):
    LAPLACIAN = "laplacian"
    IC9600 = "ic9600"


@tile_group.register(
    schema_id="chainner:pepedp:best_tile",
    name="Best Tile",
    description=[
        "Extract the best (most complex/detailed) tile from an image.",
        "Uses complexity analysis (Laplacian variance or IC9600 neural network) to find the most informative region.",
        "Based on PepeDP by umzi2.",
    ],
    icon="BsCrop",
    inputs=[
        ImageInput(),
        NumberInput("Tile Size", default=512, min=64, max=2048, unit="px"),
        EnumInput(
            ComplexityMethod, "Complexity Method", default=ComplexityMethod.LAPLACIAN
        ).with_docs(
            "**Laplacian**: Fast CPU-based method using Laplacian variance. Good for detecting edges and details.",
            "**IC9600**: Neural network-based complexity assessment. More accurate but requires GPU and is slower.",
        ),
        SliderInput(
            "Threshold",
            min=0,
            max=1,
            default=0.0,
            precision=2,
            step=0.01,
        ).with_docs(
            "Minimum complexity score required. If the best tile's score is below this threshold, an error is raised.",
            "Set to 0 to always accept the best tile.",
        ),
        if_enum_group(2, ComplexityMethod.LAPLACIAN)(
            NumberInput("Median Blur", default=5, min=1, max=21).with_docs(
                "Kernel size for median blur preprocessing. Higher values reduce noise but may blur fine details.",
                "Must be an odd number (will be adjusted if even).",
            ),
        ),
    ],
    outputs=[
        ImageOutput(),
        NumberOutput("Complexity Score", output_type="0..1").with_docs(
            "The complexity score of the extracted tile. Higher values indicate more detail/information."
        ),
    ],
)
def best_tile_node(
    img: np.ndarray,
    tile_size: int,
    method: ComplexityMethod,
    threshold: float,
    median_blur: int,
) -> tuple[np.ndarray, float]:
    """Extract the single best tile from an image."""
    from pepedp.scripts.utils.complexity.laplacian import LaplacianComplexity
    from pepeline import best_tile

    h, w = img.shape[:2]
    channels = img.shape[2] if len(img.shape) == 3 else 1

    # Create the complexity function
    if method == ComplexityMethod.LAPLACIAN:
        complexity_fn = LaplacianComplexity(median_blur=median_blur)
    else:
        from pepedp.scripts.utils.complexity.ic9600 import IC9600Complexity

        complexity_fn = IC9600Complexity()

    # Convert BGR -> RGB for PepeDP (PepeDP expects RGB)
    if channels == 3:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = img

    # If image is smaller than tile size, return the whole image with its score
    if h < tile_size or w < tile_size:
        complexity = complexity_fn(img_rgb)
        score = float(np.mean(complexity))
        if score < threshold:
            raise ValueError(
                f"Image complexity score ({score:.4f}) is below threshold ({threshold}). "
                f"Image is smaller than tile size ({w}x{h} < {tile_size}x{tile_size})."
            )
        return img, score

    # If image is exactly tile size, return it with its score
    if h == tile_size and w == tile_size:
        complexity = complexity_fn(img_rgb)
        score = float(np.mean(complexity))
        if score < threshold:
            raise ValueError(
                f"Image complexity score ({score:.4f}) is below threshold ({threshold})."
            )
        return img, score

    # Calculate the complexity map
    complexity = complexity_fn(img_rgb)

    # Find the position of the best tile
    y, x = best_tile(complexity, tile_size)

    # Extract the tile and its score
    tile_rgb, _, score = complexity_fn.get_tile_comp_score(
        img_rgb, complexity, y, x, tile_size
    )

    # Check threshold
    if score < threshold:
        raise ValueError(
            f"Best tile complexity score ({score:.4f}) is below threshold ({threshold})."
        )

    # Convert RGB -> BGR for chaiNNer
    if channels == 3:
        tile_bgr = cv2.cvtColor(tile_rgb, cv2.COLOR_RGB2BGR)
    else:
        tile_bgr = tile_rgb

    return tile_bgr, float(score)
