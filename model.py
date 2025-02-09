"""Code for training the behavior cloning model.

Usage:
python model.py \
  --batch_size 32 \
  --epochs 10 \
  --dropout 0.2 \
  --model nvidia \
  --csv ~/work/data/behavioral-cloning/tracks/driving_log.csv \
  --root ~/work/data/behavioral-cloning/tracks/img/ \
  --model_file model.h5
"""
import math
import os.path
import random

import cv2
import numpy as np
import pandas as pd
import sklearn.model_selection as skms
import sklearn.utils as sku
from absl import app
from absl import flags
from keras import layers
from keras.models import Sequential

flags.DEFINE_string("csv", "~/work/data/behavioral-cloning/driving_log.csv",
                    "Driving log CSV file.")
flags.DEFINE_string("root", "~/work/data/behavioral-cloning/IMG/",
                    "Root directory for images.")
flags.DEFINE_float("validation_proportion", 0.2, "Validation set proportion.")
flags.DEFINE_integer("epochs", 10, "Number of epochs.")
flags.DEFINE_integer("batch_size", 32, "Batch size.")
flags.DEFINE_float("correction", 0.2,
                   "Angle correction for left and right camera.")
flags.DEFINE_float("dropout", None, "Dropout rate.")
flags.DEFINE_string("model", "lenet", "Model: lenet or nvidia.")
flags.DEFINE_string("model_file", "model.h5", "Model file to save.")
FLAGS = flags.FLAGS


def read_driving_log(csv_path, root):
  df = pd.read_csv(csv_path, names=["center_img", "left_img", "right_img",
                                    "steering_angle", "throttle", "break",
                                    "speed"])
  for col in ["center_img", "left_img", "right_img"]:
    df[col] = df[col].apply(
      lambda x: os.path.join(root, os.path.basename(x)))

  return df


def images_generator(img_df, training=True, batch_size=32):
  random.seed(1)
  while True:
    img_df = sku.shuffle(img_df)
    batch_images = []
    batch_angles = []
    for _, row in img_df.iterrows():
      # Read center, left, and right images
      frame_images = [read_img(row[col]) for col in
                      ["center_img", "left_img", "right_img"]]
      angle = row["steering_angle"]
      frame_angles = [angle, angle + FLAGS.correction, angle - FLAGS.correction]

      batch_images.extend(frame_images)
      batch_angles.extend(frame_angles)

      # Flipping images
      if training:
        batch_images.extend([np.fliplr(img) for img in frame_images])
        batch_angles.extend([-angle for angle in frame_angles])
      if len(batch_images) >= batch_size:
        X_train = np.array(batch_images[:batch_size])
        y_train = np.array(batch_angles[:batch_size])
        del batch_images[:batch_size]
        del batch_angles[:batch_size]
        yield X_train, y_train


def read_img(img_path):
  img = cv2.imread(img_path)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  return img


def driving_model(img_df, batch_size=32):
  """Define and train driving model."""
  train_df, valid_df = skms.train_test_split(
    img_df, test_size=FLAGS.validation_proportion,
    random_state=1, shuffle=True)
  train_gen = images_generator(train_df, training=True, batch_size=batch_size)
  # Augmentation factor
  aug_factor = 2
  steps_per_epoch = math.ceil(len(train_df) * aug_factor / batch_size)
  valid_gen = images_generator(valid_df, training=False, batch_size=batch_size)
  valid_steps = math.ceil(len(valid_df) / batch_size)

  model = Sequential()
  model.add(layers.Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
  model.add(layers.Cropping2D(cropping=((50, 20), (0, 0))))

  if FLAGS.model == "lenet":
    lenet_model(model)
  elif FLAGS.model == "nvidia":
    nvidia_model(model)
  else:
    raise ValueError("Invalid model.")

  model.compile(loss="mse", optimizer="adam")
  print(model.summary())

  model.fit_generator(train_gen, steps_per_epoch=steps_per_epoch,
                      epochs=FLAGS.epochs,
                      validation_data=valid_gen,
                      validation_steps=valid_steps,
                      verbose=1)

  model.save(FLAGS.model_file)


def lenet_model(model):
  """LeNet model."""

  # Conv1, output shape: (156, 316, 6)
  model.add(layers.Conv2D(filters=6, kernel_size=5, strides=1,
                          activation="relu"))
  # Output shape: (78, 158, 6)
  model.add(layers.MaxPooling2D(pool_size=2, strides=2))
  if FLAGS.dropout:
    model.add(layers.Dropout(FLAGS.dropout))
  # Conv2, output shape: (74, 154, 16)
  model.add(layers.Conv2D(filters=16, kernel_size=5, strides=1,
                          activation="relu"))
  # Output shape: (37, 77, 16)
  model.add(layers.MaxPooling2D(pool_size=2, strides=2))
  if FLAGS.dropout:
    model.add(layers.Dropout(FLAGS.dropout))
  # Flatten
  model.add(layers.Flatten())
  # FC1
  model.add(layers.Dense(units=120, activation="relu"))
  # FC2
  model.add(layers.Dense(units=84, activation="relu"))
  model.add(layers.Dense(1))


def nvidia_model(model):
  """Nvidia model, https://devblogs.nvidia.com/deep-learning-self-driving-cars/ """

  model.add(layers.BatchNormalization())
  model.add(layers.Conv2D(filters=24, kernel_size=5, strides=2,
                          activation="relu"))
  if FLAGS.dropout:
    model.add(layers.Dropout(FLAGS.dropout))
  model.add(layers.Conv2D(filters=36, kernel_size=5, strides=2,
                          activation="relu"))
  if FLAGS.dropout:
    model.add(layers.Dropout(FLAGS.dropout))

  model.add(layers.Conv2D(filters=48, kernel_size=5, strides=2,
                          activation="relu"))
  if FLAGS.dropout:
    model.add(layers.Dropout(FLAGS.dropout))

  model.add(layers.Conv2D(filters=64, kernel_size=3, strides=2,
                          activation="relu"))
  if FLAGS.dropout:
    model.add(layers.Dropout(FLAGS.dropout))

  model.add(layers.Conv2D(filters=64, kernel_size=3, strides=1,
                          activation="relu"))
  if FLAGS.dropout:
    model.add(layers.Dropout(FLAGS.dropout))

  model.add(layers.Flatten())
  model.add(layers.Dense(units=1164, activation="relu"))
  model.add(layers.Dense(units=100, activation="relu"))
  model.add(layers.Dense(units=50, activation="relu"))
  model.add(layers.Dense(units=10, activation="relu"))
  model.add(layers.Dense(1))


def main(unused_argv):
  img_df = read_driving_log(os.path.expanduser(FLAGS.csv),
                            os.path.expanduser(FLAGS.root))
  driving_model(img_df, batch_size=FLAGS.batch_size)


if __name__ == "__main__":
  app.run(main)
