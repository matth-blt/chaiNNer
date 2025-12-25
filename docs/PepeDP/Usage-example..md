## Best Tile Extraction

The `BestTile` module is designed to automatically extract the most informative and detailed tiles from input images. It can be used to prepare high-quality samples for training or analysis by selecting regions with the highest structural complexity.

#### Example usage:

```python
from pepedp.enum import ProcessType
from pepedp.scripts.utils.best_tile import BestTile
from pepedp.scripts.utils.complexity.laplacian import LaplacianComplexity
# from pepedp.scripts.utils.complexity.ic9600 import IC9600Complexity

bt = BestTile(
    in_folder="input_folder",            # Path to the folder containing source images
    out_folder="output_folder",          # Path where selected tiles will be saved
    tile_size=512,                       # Tile size in pixels (square: 512x512)
    process_type=ProcessType.THREAD,     # Processing type (e.g., multithreading)
    scale=1,                             # Downscale factor used during tile search (1 = no scaling)
    dynamic_n_tiles=True,                # If True, multiple tiles can be extracted from a single image if it's large enough
    threshold=0.0,                       # Minimum quality score (e.g., Laplacian) to keep the tile
    image_gray=True,                     # Whether to convert input images to grayscale
    func=LaplacianComplexity(median_blur=5),  # Complexity estimation method (here: Laplacian with median blur)
)

bt.run()
```

#### Complexity Functions

The `func` argument defines the complexity evaluation strategy used to rank and select tiles. Available options include:

* `LaplacianComplexity`: evaluates detail richness using a Laplacian filter with optional blurring;
* `IC9600Complexity`: utilizes a neural network model to detect areas rich in fine-grained details.

These methods help ensure that only the most structurally informative tiles are selected for downstream tasks.

## Video Frame Extraction with Embedding-Based Deduplication

The `VideoToFrame` utility allows you to extract key frames from a video by measuring visual differences between frames using image embeddings. This method ensures that only significantly different frames are retained, making it useful for dataset curation and redundancy reduction.

#### Example Usage

```python
from pepedp.embedding.embedding_class import ImgToEmbedding
from pepedp.embedding.enum import EmbeddedModel
from pepedp.scripts.utils.distance import euclid_dist
from pepedp.scripts.utils.video_to_frames import VideoToFrame

vtf = VideoToFrame(
    embedder=ImgToEmbedding(
        model=EmbeddedModel.ConvNextS,  # Embedding model (e.g., ConvNeXt Small)
        amp=True,                       # Enables automatic mixed precision (fp16) if supported
        scale=4                         # Downscales input image to speed up embedding
    ),
    threshold=0.4,                      # Distance threshold: only sufficiently different frames are saved
    distance_fn=euclid_dist            # Distance function between embeddings (Euclidean)
)

vtf("video.mp4", "extracted_video")    # Extracts and saves diverse frames from the input video
```

---

### Internal Workflow

The `VideoToFrame` class converts a video into a sequence of distinct key frames based on feature vector (embedding) differences. The algorithm proceeds as follows:

1. The video is read frame by frame.
2. Each frame is passed through an image embedder (e.g., `ConvNeXt`) to obtain a feature vector.
3. The embedding of the current frame is compared to that of the last saved frame using the specified distance function.
4. If the distance exceeds the `threshold`, the frame is saved.

---

### Key Parameters Explained

#### `embedder=ImgToEmbedding(...)`

Defines the image embedding model and pre-processing behavior. Supported models include:

* `ConvNextS`, `ConvNextL`: lightweight convolutional models pretrained on ImageNet;
* `DINOv2` (`VITS`, `VITB`, `VITL`, `VITG`): self-supervised transformer models from Facebook Research, suitable for semantic similarity.

Parameters:

* `scale=4`: input images are downscaled by a factor of 4 before embedding, reducing memory and compute costs.
* `amp=True`: activates automatic mixed precision (`float16`) for faster inference on compatible hardware.

#### `threshold=0.4`

Defines the minimum distance between the embeddings of two frames required to consider them different. A value in the range of `0.3–0.5` works well for models like `ConvNext`.

#### `distance_fn=euclid_dist`

Function to compute the distance between two embeddings:

* `euclid_dist(...)`: standard L2 (Euclidean) distance;
* `cosine_dist(...)`: cosine distance, commonly used with transformer-based models such as DINOv2.

## Duplicate Image Detection and Removal

The following example demonstrates how to detect and move duplicate images within a folder based on embedding similarity.

```python
from pepedp.embedding.embedding_class import ImgToEmbedding
from pepedp.embedding.enum import EmbeddedModel
from pepedp.scripts.utils.deduplicate import (
    create_embedd,
    filtered_pairs,
    move_duplicate_files,
)
from pepedp.scripts.utils.distance import euclid_dist

# Generate embeddings for all images in the "input" folder
embedded = create_embedd(
    img_folder="input",
    embedder=ImgToEmbedding(
        model=EmbeddedModel.ConvNextS,  # Using ConvNext Small model
        amp=True,                       # Enable mixed precision (float16) for faster processing
        scale=4                        # Downscale images by a factor of 4 before embedding
    ),
)

# Find pairs of similar images based on Euclidean distance
paired = filtered_pairs(
    embeddings=embedded,       # Embeddings dictionary: {filename: embedding}
    dist_func=euclid_dist,     # Distance metric: Euclidean distance
    threshold=1.5              # Distance threshold below which images are considered duplicates
)

# Move identified duplicate files to the "duplicate" folder
move_duplicate_files(
    duplicates_dict=paired,    # Dictionary with detected duplicate pairs
    src_dir="input",           # Source directory of images
    dst_dir="duplicate",       # Destination directory for duplicates
)
```

---

### Internal Functions Overview

#### `create_embedd(...)`

```python
def create_embedd(img_folder: str, embedder: ImgToEmbedding = ImgToEmbedding()):
```

Generates embeddings for all images in the specified folder.

* **`img_folder`** — path to the folder containing images.
* **`embedder`** — image embedding model instance (default is `ConvNextS`).
* **Returns:** a dictionary where keys are filenames and values are embeddings.

---

#### `filtered_pairs(...)`

```python
def filtered_pairs(embeddings, dist_func=euclid_dist, threshold=1.5, device_str=None):
```

Finds pairs of images whose embeddings are closer than a specified threshold.

* **`embeddings`** — dictionary of embeddings `{filename: embedding}`.
* **`dist_func`** — function to calculate distance between two embeddings (e.g., Euclidean or cosine).
* **`threshold`** — numeric threshold; pairs with distance below this are considered duplicates.
* **Operation:** embeddings are batched and compared pairwise asymmetrically (`i` vs. all `j > i`), excluding duplicates.
* **Output:** a list of pairs `(i, j, distance)` that satisfy the threshold condition.

---

#### `move_duplicate_files(...)`

```python
def move_duplicate_files(duplicates_dict, src_dir="", dst_dir=""):
```

Moves files marked as duplicates from the source to the destination directory.

* **`duplicates_dict`** — dictionary returned by `filtered_pairs` containing:

  * `names`: list of all file names.
  * `filtered_pairs`: list of tuples `(i, j, distance)` representing duplicate pairs.
* **`src_dir`** — source directory containing original images.
* **`dst_dir`** — destination directory where duplicates will be moved.
* Uses `shutil.move(...)` to perform file operations.

Additional details:

* Files not found in the source directory are logged.
* Files that already exist in the destination folder are skipped.
* The total number of successfully moved files is reported.

## Threshold-Based Image Filtering (Using IQA)

This module allows you to automatically filter out low-quality images from a directory using various Image Quality Assessment (IQA) algorithms.

#### Example Usage

```python
from pepedp.torch_enum import ThresholdAlg

thread_type = ThresholdAlg.HIPERIQA

alg = thread_type.value(
    img_dir="input",           # Directory containing images to process
    batch_size=8,              # Number of images processed in a batch
    threshold=0.5,             # Threshold for quality score (see notes below)
    median_threshold=0.5,      # Percentage of top-quality images to keep or move
    move_folder=None           # Destination directory; if None, low-quality images are deleted
)
alg()
```

---

### `ThresholdAlg` Options

`ThresholdAlg` is an enumeration that currently supports the following five algorithms:

* **HIPERIQA**, **ANIIQA**, **TOPIQ** — General-purpose IQA models that return scores in the range **0 (poor)** to **1 (excellent)**.
* **BLOCKINESS** — Detects JPEG compression artifacts. A **higher score indicates worse quality**. Thresholding is **inverted**: images **above** the threshold are discarded.
* **IC9600** — Evaluates the amount of structural detail in an image. Useful for keeping only rich-content samples **0 (poor)** to **1 (excellent)**.

---

### Common Parameters

* **`img_dir`** *(str)* — Path to the directory containing images to be evaluated.
* **`batch_size=8`** *(int)* — Number of images processed simultaneously (batches).
* **`threshold=0.5`** *(float)* — The cutoff value for IQA:

  * For most models (HIPERIQA, ANIIQA, etc.): images **below** the threshold are considered low quality.
  * For `BLOCKINESS`: images **above** the threshold are discarded due to higher compression artifacts.
* **`median_threshold=0.5`** *(float, optional)* — If set to a non-zero value, the algorithm:

  * Sorts images that passed the `threshold` by quality score.
  * Keeps (or moves) only the top `median_threshold` fraction (e.g., `0.8` = top 80%).
* **`move_folder=None`** *(str or None)* — If specified, high-quality images are **moved** to this directory instead of deleting the low-quality ones. If `None`, images failing the threshold are **deleted**.


