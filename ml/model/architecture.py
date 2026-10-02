from tensorflow.keras.layers import (
    Add,
    BatchNormalization,
    Conv2D,
    Dense,
    DepthwiseConv2D,
    Dropout,
    GlobalAveragePooling2D,
    Input,
    MaxPooling2D,
    Multiply,
    ReLU,
    Reshape,
)
from tensorflow.keras.models import Model


def se_block(input_tensor, ratio: int = 8):
    filters = input_tensor.shape[-1]
    reduced_filters = max(int(filters) // ratio, 1)
    squeeze = GlobalAveragePooling2D()(input_tensor)
    squeeze = Dense(reduced_filters, activation="relu")(squeeze)
    squeeze = Dense(int(filters), activation="sigmoid")(squeeze)
    squeeze = Reshape((1, 1, int(filters)))(squeeze)
    return Multiply()([input_tensor, squeeze])


def mbconv_block(x, filters: int):
    shortcut = x
    x = Conv2D(filters * 2, (1, 1), padding="same")(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = DepthwiseConv2D((3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = se_block(x)
    x = Conv2D(filters, (1, 1), padding="same")(x)
    x = BatchNormalization()(x)
    if shortcut.shape[-1] == filters:
        x = Add()([x, shortcut])
    return x


def build_model(input_shape=(128, 128, 3), number_of_classes: int = 6) -> Model:
    inputs = Input(shape=input_shape)

    x = Conv2D(32, (3, 3), padding="same")(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(32, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)

    shortcut = Conv2D(64, (1, 1), padding="same")(x)
    x = Conv2D(64, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(64, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Add()([x, shortcut])
    x = ReLU()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)

    shortcut = Conv2D(128, (1, 1), padding="same")(x)
    x = Conv2D(128, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(128, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Add()([x, shortcut])
    x = ReLU()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.30)(x)

    shortcut = Conv2D(256, (1, 1), padding="same")(x)
    x = Conv2D(256, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(256, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Add()([x, shortcut])
    x = ReLU()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.40)(x)

    x = mbconv_block(x, 256)
    x = mbconv_block(x, 256)
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.50)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.30)(x)
    outputs = Dense(number_of_classes, activation="softmax")(x)

    return Model(inputs, outputs, name="hybrid_defect_detection_model")
