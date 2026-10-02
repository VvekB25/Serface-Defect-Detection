from pathlib import Path

import numpy as np
from PIL import Image

from ml.preprocessing.image_preprocessor import preprocess_image


def test_preprocess_image_returns_normalized_batch(tmp_path: Path):
    image_path = tmp_path / "sample.jpg"
    Image.new("RGB", (32, 32), color=(255, 128, 0)).save(image_path)

    result = preprocess_image(image_path)

    assert result.shape == (1, 128, 128, 3)
    assert result.dtype == np.float32
    assert float(result.min()) >= 0.0
    assert float(result.max()) <= 1.0
