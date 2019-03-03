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
flags.DEFINE_integer("epochs", 10, "Number of epochs.")
flags.DEFINE_integer("batch_size", 32, "Batch size.")
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
  num = len(img_df)
  random.seed(1)
  while True:
    img_df = sku.shuffle(img_df)
    images = []
    angles = []
    for _, row in img_df.iterrows():
      img = cv2.imread(row["center_img"])
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      angle = row["steering_angle"]
      images.append(img)
      angles.append(angle)

      # Flipping image
      if training:
        images.append(np.fliplr(img))
        angles.append(-angle)
      if len(images) >= batch_size:
        X_train = np.array(images[:batch_size])
        y_train = np.array(angles[:batch_size])
        del images[:batch_size]
        del angles[:batch_size]
        yield X_train, y_train


def driving_model(img_df, batch_size=32):
  train_df, valid_df = skms.train_test_split(img_df, test_size=0.2,
                                             random_state=1, shuffle=True)
  train_gen = images_generator(train_df, training=True, batch_size=batch_size)
  # Augmentation factor
  aug_factor = 2
  steps_per_epoch = math.ceil(len(train_df) * aug_factor / batch_size)
  valid_gen = images_generator(valid_df, training=False, batch_size=batch_size)
  valid_steps = math.ceil(len(valid_df) / batch_size)

  model = Sequential()
  model.add(layers.Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))

  # Conv1, output shape: (156, 316, 6)
  model.add(layers.Conv2D(filters=6, kernel_size=5, strides=1,
                          activation="relu", input_shape=(160, 320, 3)))
  # Output shape: (78, 158, 6)
  model.add(layers.MaxPooling2D(pool_size=2, strides=2))
  # dropout?

  # Conv2, output shape: (74, 154, 16)
  model.add(
    layers.Conv2D(filters=16, kernel_size=5, strides=1, activation="relu",
                  input_shape=(78, 158, 6)))
  # Output shape: (37, 77, 16)
  model.add(layers.MaxPooling2D(pool_size=2, strides=2))
  # Dropout?

  # Flatten
  model.add(layers.Flatten())

  # FC1:
  model.add(layers.Dense(units=120, activation="relu"))
  # Dropout?

  # FC2:
  model.add(layers.Dense(units=84, activation="relu"))
  # Dropout?

  model.add(layers.Dense(1))

  model.compile(loss="mse", optimizer="adam")

  model.fit_generator(train_gen, steps_per_epoch=steps_per_epoch,
                      epochs=FLAGS.epochs,
                      validation_data=valid_gen,
                      validation_steps=valid_steps,
                      verbose=1)

  model.save("model.h5")


def main(unused_argv):
  img_df = read_driving_log(os.path.expanduser(FLAGS.csv),
                            os.path.expanduser(FLAGS.root))
  driving_model(img_df, batch_size=FLAGS.batch_size)


if __name__ == "__main__":
  app.run(main)
