#!/usr/bin/env python

# https://www.kaggle.com/c/dogs-vs-cats/data
# https://gist.github.com/fchollet/7eb39b44eb9e16e59632d25fb3119975
# https://www.kaggle.com/c/planet-understanding-the-amazon-from-space/data

import os

import tensorflow as tf
from tensorflow.keras import applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dropout, Flatten, Dense, Input

from model import get_model


def main():
    model = get_model()

    model.compile(
        loss="binary_crossentropy",
        optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
        metrics=["accuracy"],
    )

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True
    )

    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow_from_directory(
        "./cats-dogs/train", target_size=(224, 224), batch_size=4, class_mode="binary"
    )

    validation_generator = test_datagen.flow_from_directory(
        "./cats-dogs/validation",
        target_size=(224, 224),
        batch_size=4,
        class_mode="binary",
    )

    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir="./tboard_logs")

    model.fit_generator(
        train_generator,
        steps_per_epoch=100,
        epochs=2,
        validation_data=validation_generator,
        validation_steps=10,
        callbacks=[tensorboard_cb],
    )

    model.save_weights("trained_model.chkpt")


if __name__ == "__main__":
    main()
