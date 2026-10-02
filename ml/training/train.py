import json

from tensorflow.keras.optimizers import Adam

from ml.config.settings import (
    BATCH_SIZE,
    CLASSES,
    EPOCHS,
    IMAGE_SIZE,
    LEARNING_RATE,
    MODEL_DIR,
    MODEL_PATH,
)
from ml.model.architecture import build_model
from ml.training.data_loader import create_generators


def train_model():
    train_generator, validation_generator = create_generators()
    model = build_model(
        input_shape=(*IMAGE_SIZE, 3),
        number_of_classes=len(CLASSES),
    )
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=EPOCHS,
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    class_mapping = {
        str(index): class_name
        for class_name, index in train_generator.class_indices.items()
    }
    with open(MODEL_DIR / "class_names.json", "w", encoding="utf-8") as file:
        json.dump(class_mapping, file, indent=2)
    return model, history


if __name__ == "__main__":
    train_model()
