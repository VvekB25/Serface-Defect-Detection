from tensorflow.keras.preprocessing.image import ImageDataGenerator

from ml.config.settings import BATCH_SIZE, IMAGE_SIZE, PROCESSED_DATASET_ROOT


def create_generators(dataset_root=PROCESSED_DATASET_ROOT):
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=20,
        zoom_range=0.2,
        horizontal_flip=True,
    )
    validation_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_generator = train_datagen.flow_from_directory(
        dataset_root / "train",
        target_size=IMAGE_SIZE,
        color_mode="rgb",
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=True,
    )
    validation_generator = validation_datagen.flow_from_directory(
        dataset_root / "validation",
        target_size=IMAGE_SIZE,
        color_mode="rgb",
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
    )
    return train_generator, validation_generator
