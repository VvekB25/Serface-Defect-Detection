from pathlib import Path

import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image

from ml.config.settings import IMAGE_SIZE


ImageInput = str | Path | Image.Image | np.ndarray


def preprocess_image(image_input: ImageInput) -> np.ndarray:
    """Return a model-ready RGB batch normalized to float32 values in [0, 1].

    The model expects 128x128 RGB images. File paths use Keras image loading,
    matching the original notebook; PIL images and NumPy arrays are converted
    to RGB and resized to the same dimensions.
    """
    if isinstance(image_input, (str, Path)):
        loaded_image = image.load_img(
            image_input,
            target_size=IMAGE_SIZE,
            color_mode="rgb",
        )
    else:
        if isinstance(image_input, np.ndarray):
            if image_input.ndim not in (2, 3):
                raise ValueError("NumPy image input must have 2 or 3 dimensions")
            loaded_image = Image.fromarray(image_input)
        elif isinstance(image_input, Image.Image):
            loaded_image = image_input
        else:
            raise TypeError("image_input must be a path, PIL image, or NumPy array")
        loaded_image = loaded_image.convert("RGB").resize(
            IMAGE_SIZE,
            resample=Image.Resampling.NEAREST,
        )

    array = image.img_to_array(loaded_image).astype("float32") / 255.0
    return np.expand_dims(array, axis=0)
