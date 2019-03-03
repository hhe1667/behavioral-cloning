import numpy as np
import os.path
import pandas as pd


def read_driving_log(logs_path):
  df = pd.read_csv(logs_path, names=["center_img", "left_img", "right_img",
                                      "steering_angle", "throttle", "break",
                                      "speed"])
  for col in ["center_img", "left_img", "right_img"]:
    df[col] = df[col].apply(lambda x: os.path.basename(x))

  print(df)

read_driving_log("/home/huahai/work/data/behavioral-cloning/img/driving_log.csv")
