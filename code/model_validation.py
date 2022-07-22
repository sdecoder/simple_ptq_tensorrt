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
import torch.onnx
import utils
import argparse

import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import utils

print(f'[trace] prepare the test dataset')
transform = transforms.Compose([
  transforms.ToTensor(),
  transforms.Normalize((0.1307,), (0.3081,))
])

dataset2 = datasets.MNIST('../data', train=False,
                          transform=transform)

test_kwargs = {'batch_size': 1}
test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

def _validate_trt_engine_file(filename):
  print(f'[trace] validate the tensorrt engine file')
  if not os.path.exists(filename):
    print(f'[trace] target engine file {filename} does not exist, exit')
    return

  print(f'[trace] start to read the engine data')
  with open(filename, 'rb') as f:
    engine_data = f.read()
  TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
  trt_runtime = trt.Runtime(TRT_LOGGER)
  engine = trt_runtime.deserialize_cuda_engine(engine_data)
  if engine:
    print(f'[trace] TensorRT engine created')
  else:
    print(f'[trace] failed to create TensorRT engine, exit')
    exit(-1)

  utils.test_using_trt(engine, test_loader)
  pass

def _validate_onnx_file():
  print(f'[trace] validate the onnx file')
  file_name = './resnet50.onnx'
  if not os.path.exists(file_name):
    print(f'[trace] target onnx file {file_name} does not exist, exit')
    return

  import onnx
  print(f'[trace] start to the onnx file {file_name}')
  onnx_model = onnx.load(file_name)
  onnx.checker.check_model(onnx_model)
  print(f'[trace] the onnx file {file_name} checked')
  print(f'[trace] prepare the onnx runtime')
  import onnxruntime
  ort_session = onnxruntime.InferenceSession(file_name)
  utils.test_using_onnx_session(ort_session, test_loader)
  pass

def _main():
  print(f"[trace] working in the main function")
  '''
  [trace] Onnx test set: Average loss: 0.0256, Accuracy: 9915/10000 (99%)
  '''
  #_validate_onnx_file()

  '''
  [trace] TensorRT test set: Average loss: 0.0000, Accuracy: 9915/10000 (99%)
  _validate_trt_engine_file('./resnet50.engine')
  '''

  #[trace] TensorRT test set: Average loss: 0.0000, Accuracy: 9915/10000 (99%)
  _validate_trt_engine_file('./resnet50.int8.engine')
  pass

if __name__ == '__main__':
  _main()
  pass
