import argparse

import onnx
import tensorrt as trt
import os
import torch
import torchvision
import pycuda.driver as cuda
import pycuda.autoinit
from PIL import Image
import numpy as np

from utils import find_sample_data, build_engine_from_onnxmodel, \
  MNISTEntropyCalibrator, check_accuracy, \
  load_mnist_data, load_mnist_labels, CalibratorMode


def main():
  print(f'[trace] working in the main function')
  # Inference batch size can be different from calibration batch size.
  # onnxmodel = create_onnx_file()
  _, data_files = find_sample_data(
    description="Runs a Caffe MNIST network in Int8 mode",
    subfolder="mnist",
    find_files=[
      "t10k-images-idx3-ubyte",
      "t10k-labels-idx1-ubyte",
      "train-images-idx3-ubyte",
      #ModelData.DEPLOY_PATH,
      #ModelData.MODEL_PATH,
    ],
    err_msg="Please follow the README to download the MNIST dataset",
  )
  [test_set, test_labels, train_set,] = data_files

  # Now we create a calibrator and give it the location of our calibration data.
  # We also allow it to cache calibration data for faster engine building.
  calibration_cache = "mnist_calibration.cache"
  calib = MNISTEntropyCalibrator(train_set, cache_file=calibration_cache)
  onnxmodel = 'resnet50.onnx'
  #engine = build_engine_from_onnxmodel_int8(onnxmodel, calib)
  mode: CalibratorMode = CalibratorMode.FP32
  engine = build_engine_from_onnxmodel(onnxmodel, mode, calib)

  # Batch size for inference can be different than batch size used for calibration.

  '''
  batch_size = 32
  context = engine.create_execution_context()
  check_accuracy(context, batch_size, test_set=load_mnist_data(test_set), test_labels=load_mnist_labels(test_labels))
  with build_int8_engine(deploy_file, model_file, calib, batch_size) as engine, engine.create_execution_context() as context:
    # Batch size for inference can be different than batch size used for calibration.
    check_accuracy(
      context, batch_size, test_set=load_mnist_data(test_set), test_labels=load_mnist_labels(test_labels)
    )
  '''

  pass


if __name__ == '__main__':
  main()
  pass
