import tensorflow as tf
from tensorflow.keras import applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dropout, Flatten, Dense, Input


def get_model():
    input_tensor = Input(shape=(224, 224, 3))
    model = applications.MobileNetV2(
        weights="imagenet",
        input_tensor=input_tensor,
        input_shape=(224, 224, 3),
        include_top=False,
    )

    for layer in model.layers:
        layer.trainable = False

    op0 = Flatten()(model.output)
    op1 = Dense(16, activation="relu")(op0)
    op2 = Dropout(0.5)(op1)

    output_tensor = Dense(1, activation="sigmoid")(op2)

    model = Model(inputs=input_tensor, outputs=output_tensor)
    return model
