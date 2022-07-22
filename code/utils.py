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
import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from enum import Enum

# Returns a numpy buffer of shape (num_images, 1, 28, 28)
def load_mnist_data(filepath):
  with open(filepath, "rb") as f:
    raw_buf = np.fromstring(f.read(), dtype=np.uint8)
  # Make sure the magic number is what we expect
  assert raw_buf[0:4].view(">i4")[0] == 2051
  num_images = raw_buf[4:8].view(">i4")[0]
  image_c = 1
  image_h = raw_buf[8:12].view(">i4")[0]
  image_w = raw_buf[12:16].view(">i4")[0]
  # Need to scale all values to the range of [0, 1]
  return np.ascontiguousarray(
    (raw_buf[16:] / 255.0).astype(np.float32).reshape(num_images, image_c, image_h, image_w)
  )


# Returns a numpy buffer of shape (num_images)
def load_mnist_labels(filepath):
  with open(filepath, "rb") as f:
    raw_buf = np.fromstring(f.read(), dtype=np.uint8)
  # Make sure the magic number is what we expect
  assert raw_buf[0:4].view(">i4")[0] == 2049
  num_labels = raw_buf[4:8].view(">i4")[0]
  return np.ascontiguousarray(raw_buf[8:].astype(np.int32).reshape(num_labels))


class MNISTEntropyCalibrator(trt.IInt8EntropyCalibrator2):
  def __init__(self, training_data, cache_file, batch_size=64):
    # Whenever you specify a custom constructor for a TensorRT class,
    # you MUST call the constructor of the parent explicitly.
    trt.IInt8EntropyCalibrator2.__init__(self)

    self.cache_file = cache_file

    # Every time get_batch is called, the next batch of size batch_size will be copied to the device and returned.
    self.data = load_mnist_data(training_data)
    self.batch_size = batch_size
    self.current_index = 0

    # Allocate enough memory for a whole batch.
    self.device_input = cuda.mem_alloc(self.data[0].nbytes * self.batch_size)

  def get_batch_size(self):
    return self.batch_size

  # TensorRT passes along the names of the engine bindings to the get_batch function.
  # You don't necessarily have to use them, but they can be useful to understand the order of
  # the inputs. The bindings list is expected to have the same ordering as 'names'.
  def get_batch(self, names):
    if self.current_index + self.batch_size > self.data.shape[0]:
      return None

    current_batch = int(self.current_index / self.batch_size)
    if current_batch % 10 == 0:
      print("Calibrating batch {:}, containing {:} images".format(current_batch, self.batch_size))

    batch = self.data[self.current_index: self.current_index + self.batch_size].ravel()
    cuda.memcpy_htod(self.device_input, batch)
    self.current_index += self.batch_size
    return [self.device_input]

  def read_calibration_cache(self):
    # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
    print(f'[trace] MNISTEntropyCalibrator: read_calibration_cache: {self.cache_file}')
    if os.path.exists(self.cache_file):
      with open(self.cache_file, "rb") as f:
        return f.read()

  def write_calibration_cache(self, cache):
    print(f'[trace] MNISTEntropyCalibrator: write_calibration_cache: {cache}')
    with open(self.cache_file, "wb") as f:
      f.write(cache)


import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

import numpy as np
import random

TRT_LOGGER = trt.Logger()
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

def GiB(val):
  return val * 1 << 30

def build_engine_common_routine(network, builder, config, runtime, engine_file_path):

  input_batch_size = 1
  input_channel = 1
  input_image_width = 28
  input_image_height = 28
  network.get_input(0).shape = [input_batch_size, input_channel, input_image_width, input_image_height]
  plan = builder.build_serialized_network(network, config)
  if plan == None:
    print("[trace] builder.build_serialized_network failed, exit -1")
    exit(-1)
  engine = runtime.deserialize_cuda_engine(plan)
  print("[trace] Completed creating Engine")
  with open(engine_file_path, "wb") as f:
    f.write(plan)
  return engine

  pass

class CalibratorMode(Enum):
  INT8 = 0
  FP16 = 1
  TF32 = 2
  FP32 = 3

def build_engine_from_onnxmodel(onnx_file_path, mode: CalibratorMode, calib):

  '''
    if os.path.exists(engine_file_path):
    print(f'[trace] TensorRT engine {engine_file_path} exist, read it and return')
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
      obj = runtime.deserialize_cuda_engine(f.read())
      if obj == None:
        print(f'[trace] failed to deserialize_cuda_engine, exit -1')
        exit(-1)
      return obj
  '''
  calibrator_mode = mode.name.lower()
  engine_file_path = f'resnet50.{calibrator_mode}.engine'
  if not os.path.exists(onnx_file_path):
    print("ONNX file {} not found, exit -1".format(onnx_file_path))
    exit(-1)

  batch_size = 1
  with trt.Builder(TRT_LOGGER) as builder, \
      builder.create_network(EXPLICIT_BATCH) as network, \
      builder.create_builder_config() as config, \
      trt.OnnxParser(network, TRT_LOGGER) as parser, \
      trt.Runtime(TRT_LOGGER) as runtime:

    print("[trace] Loading ONNX file from path {}...".format(onnx_file_path))
    with open(onnx_file_path, "rb") as model:
      print("Beginning ONNX file parsing")
      if not parser.parse(model.read()):
        print("ERROR: Failed to parse the ONNX file.")
        for error in range(parser.num_errors):
          print(parser.get_error(error))
        return None
      print("Completed parsing of ONNX file")

    builder.max_batch_size = batch_size
    config.max_workspace_size = GiB(1)

    if mode == CalibratorMode.INT8:
      config.set_flag(trt.BuilderFlag.INT8)
    elif mode == CalibratorMode.FP16:
      config.set_flag(trt.BuilderFlag.FP16)
    elif mode == CalibratorMode.TF32:
      config.set_flag(trt.BuilderFlag.TF32)
    elif mode == CalibratorMode.FP32:
      # do nothing since this is the default branch
      #config.set_flag(trt.BuilderFlag.FP32)
      pass
    else:
      print(f'[trace] unknown calibrator mode: {mode}, exit')
      exit(-1)

    config.int8_calibrator = calib
    build_engine_common_routine(network, builder, config, runtime, engine_file_path)

  pass

def build_engine_from_onnx(onnx_file_path, engine_file_path):
  """Takes an ONNX file and creates a TensorRT engine to run inference with"""
  with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network, \
      builder.create_builder_config() as config, trt.OnnxParser(network, TRT_LOGGER) as parser, trt.Runtime(
    TRT_LOGGER) as runtime:

    config.max_workspace_size = 1 << 28  # 256MiB
    builder.max_batch_size = 1
    # Parse model file
    if not os.path.exists(onnx_file_path):
      print("ONNX file {} not found, exit -1".format(onnx_file_path))
      exit(-1)

    print("Loading ONNX file from path {}...".format(onnx_file_path))
    with open(onnx_file_path, "rb") as model:
      print("Beginning ONNX file parsing")
      if not parser.parse(model.read()):
        print("ERROR: Failed to parse the ONNX file.")
        for error in range(parser.num_errors):
          print(parser.get_error(error))
        return None
      print("Completed parsing of ONNX file")

    # The actual yolov3.onnx is generated with batch size 64. Reshape input to batch size 1
    print("Building an engine from file {}; this may take a while...".format(onnx_file_path))
    input_batch_size = 1
    input_channel = 3
    input_image_width = 224
    input_image_height = 224
    network.get_input(0).shape = [input_batch_size, input_channel, input_image_width, input_image_height]
    plan = builder.build_serialized_network(network, config)
    if plan == None:
      print("[trace] builder.build_serialized_network failed, exit -1")
      exit(-1)
    engine = runtime.deserialize_cuda_engine(plan)
    print("[trace] Completed creating Engine")
    with open(engine_file_path, "wb") as f:
      f.write(plan)
    return engine


# Simple helper data class that's a little nicer to use than a 2-tuple.
class HostDeviceMem(object):
  def __init__(self, host_mem, device_mem):
    self.host = host_mem
    self.device = device_mem

  def __str__(self):
    return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

  def __repr__(self):
    return self.__str__()


# Allocates all buffers required for an engine, i.e. host/device inputs/outputs.
def allocate_buffers(engine):
  inputs = []
  outputs = []
  bindings = []
  stream = cuda.Stream()
  for binding in engine:
    size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
    dtype = trt.nptype(engine.get_binding_dtype(binding))
    # Allocate host and device buffers
    host_mem = cuda.pagelocked_empty(size, dtype)
    device_mem = cuda.mem_alloc(host_mem.nbytes)
    # Append the device buffer to device bindings.
    bindings.append(int(device_mem))
    # Append to the appropriate list.
    if engine.binding_is_input(binding):
      inputs.append(HostDeviceMem(host_mem, device_mem))
    else:
      outputs.append(HostDeviceMem(host_mem, device_mem))
  return inputs, outputs, bindings, stream


# This function is generalized for multiple inputs/outputs.
# inputs and outputs are expected to be lists of HostDeviceMem objects.
def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
  # Transfer input data to the GPU.
  [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
  # Run inference.
  context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
  # Transfer predictions back from the GPU.
  [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
  # Synchronize the stream
  stream.synchronize()
  # Return only the host outputs.
  return [out.host for out in outputs]


def test_accuracy(context, device, test_loader):
  model.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
      data, target = data.to(device), target.to(device)
      output = model(data)
      test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
      pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
      correct += pred.eq(target.view_as(pred)).sum().item()

  test_loss /= len(test_loader.dataset)

  print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))


def check_accuracy(context, batch_size, test_set, test_labels):
  inputs, outputs, bindings, stream = allocate_buffers(context.engine)

  num_correct = 0
  num_total = 0

  batch_num = 0
  for start_idx in range(0, test_set.shape[0], batch_size):
    batch_num += 1
    if batch_num % 10 == 0:
      print("Validating batch {:}".format(batch_num))
    # If the number of images in the test set is not divisible by the batch size, the last batch will be smaller.
    # This logic is used for handling that case.
    end_idx = min(start_idx + batch_size, test_set.shape[0])
    effective_batch_size = end_idx - start_idx

    # Do inference for every batch.
    inputs[0].host = test_set[start_idx: start_idx + effective_batch_size]
    [output] = do_inference(
      context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream, batch_size=effective_batch_size
    )

    # Use argmax to get predictions and then check accuracy
    preds = np.argmax(output.reshape(batch_size, 10)[0:effective_batch_size], axis=1)
    labels = test_labels[start_idx: start_idx + effective_batch_size]
    num_total += effective_batch_size
    num_correct += np.count_nonzero(np.equal(preds, labels))

  percent_correct = 100 * num_correct / float(num_total)
  print("Total Accuracy: {:}%".format(percent_correct))

def locate_files(data_paths, filenames, err_msg=""):
  """
  Locates the specified files in the specified data directories.
  If a file exists in multiple data directories, the first directory is used.
  Args:
      data_paths (List[str]): The data directories.
      filename (List[str]): The names of the files to find.
  Returns:
      List[str]: The absolute paths of the files.
  Raises:
      FileNotFoundError if a file could not be located.
  """
  found_files = [None] * len(filenames)
  for data_path in data_paths:
    # Find all requested files.
    for index, (found, filename) in enumerate(zip(found_files, filenames)):
      if not found:
        file_path = os.path.abspath(os.path.join(data_path, filename))
        if os.path.exists(file_path):
          found_files[index] = file_path

  # Check that all files were found
  for f, filename in zip(found_files, filenames):
    if not f or not os.path.exists(f):
      raise FileNotFoundError(
        "Could not find {:}. Searched in data paths: {:}\n{:}".format(filename, data_paths, err_msg)
      )
  return found_files

def find_sample_data(description="Runs a TensorRT Python sample", subfolder="", find_files=[], err_msg=""):
  """
  Parses sample arguments.

  Args:
      description (str): Description of the sample.
      subfolder (str): The subfolder containing data relevant to this sample
      find_files (str): A list of filenames to find. Each filename will be replaced with an absolute path.

  Returns:
      str: Path of data directory.
  """

  # Standard command-line arguments for all samples.
  kDEFAULT_DATA_ROOT = os.path.join(os.sep, "usr", "src", "tensorrt", "data")
  kDEFAULT_DATA_ROOT = '../data/MNIST/raw'
  parser = argparse.ArgumentParser(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
    "-d",
    "--datadir",
    help="Location of the TensorRT sample data directory, and any additional data directories.",
    action="append",
    default=[kDEFAULT_DATA_ROOT],
  )
  args, _ = parser.parse_known_args()

  def get_data_path(data_dir):
    # If the subfolder exists, append it to the path, otherwise use the provided path as-is.
    data_path = os.path.join(data_dir, subfolder)
    if not os.path.exists(data_path):
      if data_dir != kDEFAULT_DATA_ROOT:
        print("WARNING: " + data_path + " does not exist. Trying " + data_dir + " instead.")
      data_path = data_dir
    # Make sure data directory exists.
    if not (os.path.exists(data_path)) and data_dir != kDEFAULT_DATA_ROOT:
      print(
        "WARNING: {:} does not exist. Please provide the correct data path with the -d option.".format(
          data_path
        )
      )
    return data_path

  data_paths = [get_data_path(data_dir) for data_dir in args.datadir]
  return data_paths, locate_files(data_paths, find_files, err_msg)



# This function builds an engine from a Caffe model.
def build_int8_engine(deploy_file, model_file, calib, batch_size=32):
  with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, \
      builder.create_builder_config() as config, trt.CaffeParser() as parser, trt.Runtime(TRT_LOGGER) as runtime:
    # We set the builder batch size to be the same as the calibrator's, as we use the same batches
    # during inference. Note that this is not required in general, and inference batch size is
    # independent of calibration batch size.
    builder.max_batch_size = batch_size
    config.max_workspace_size = GiB(1)
    config.set_flag(trt.BuilderFlag.INT8)
    config.int8_calibrator = calib
    # Parse Caffe model
    model_tensors = parser.parse(deploy=deploy_file, model=model_file, network=network, dtype=ModelData.DTYPE)
    network.mark_output(model_tensors.find(OUTPUT_NAME))
    # Build engine and do int8 calibration.
    plan = builder.build_serialized_network(network, config)
    return runtime.deserialize_cuda_engine(plan)

@DeprecationWarning
def create_onnx_file():
  exit(0)
  print(f'[trace] working in the func: create_onnx_file')
  onnx_file_name = 'resnet50.onnx'
  if os.path.exists(onnx_file_name):
    print(f'[trace] {onnx_file_name} exist, return')
    return onnx_file_name

  print(f'[trace] start to export the torchvision resnet50')
  input_name = ['input']
  output_name = ['output']
  from torch.autograd import Variable
  model = torchvision.models.resnet50(pretrained=True)
  import torch.nn as nn
  # model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
  model.fc = nn.Linear(2048, 10, bias=True)

  #modify the network to adapat MNIST


  input = Variable(torch.randn(1, 1, 28, 28))
  torch.onnx.export(model, input, 'resnet50.onnx', input_names=input_name, output_names=output_name, verbose=True)
  print(f'[trace] export done')

  test = onnx.load(onnx_file_name)
  onnx.checker.check_model(test)
  print(f"[trace] create onnx file Passed")
  return onnx_file_name


def to_numpy(tensor):
  return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def test_using_onnx_session(ort_session, test_loader):

  print(f'[trace] test using onnx session')
  test_loss = 0
  correct = 0
  for data, target in test_loader:

    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(data)}
    ort_outs = ort_session.run(None, ort_inputs)
    output = ort_outs[0]
    output = torch.from_numpy(output)
    test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
    pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    correct += pred.eq(target.view_as(pred)).sum().item()

  test_loss /= len(test_loader.dataset)
  print('[trace] Onnx test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))

def allocate_buffers_for_encoder(engine):
  """Allocates host and device buffer for TRT engine inference.
  This function is similair to the one in common.py, but
  converts network outputs (which are np.float32) appropriately
  before writing them to Python buffer. This is needed, since
  TensorRT plugins doesn't support output type description, and
  in our particular case, we use NMS plugin as network output.
  Args:
      engine (trt.ICudaEngine): TensorRT engine
  Returns:
      inputs [HostDeviceMem]: engine input memory
      outputs [HostDeviceMem]: engine output memory
      bindings [int]: buffer to device bindings
      stream (cuda.Stream): cuda stream for engine inference synchronization
  """
  print('[trace] reach func@allocate_buffers')
  inputs = []
  outputs = []
  bindings = []
  stream = cuda.Stream()

  binding_to_type = {}
  binding_to_type['input'] = np.float32
  binding_to_type['output'] = np.float32


  # Current NMS implementation in TRT only supports DataType.FLOAT but
  # it may change in the future, which could brake this sample here
  # when using lower precision [e.g. NMS output would not be np.float32
  # anymore, even though this is assumed in binding_to_type]


  for binding in engine:
    print(f'[trace] current binding: {str(binding)}')
    _binding_shape = engine.get_binding_shape(binding)
    _volume = trt.volume(_binding_shape)
    size = _volume * engine.max_batch_size
    print(f'[trace] current binding size: {size}')
    dtype = binding_to_type[str(binding)]
    # Allocate host and device buffers
    host_mem = cuda.pagelocked_empty(size, dtype)
    device_mem = cuda.mem_alloc(host_mem.nbytes)
    # Append the device buffer to device bindings.
    bindings.append(int(device_mem))
    # Append to the appropriate list.
    if engine.binding_is_input(binding):
      inputs.append(HostDeviceMem(host_mem, device_mem))
    else:
      outputs.append(HostDeviceMem(host_mem, device_mem))
  return inputs, outputs, bindings, stream


def test_using_trt(engine, test_loader):

  print(f'[trace] run the test using TensorRT engine')
  test_loss = 0
  correct = 0
  inputs, outputs, bindings, stream = allocate_buffers_for_encoder(engine)
  context = engine.create_execution_context()
  batch_size = 1
  for data, target in test_loader:

    np.copyto(inputs[0].host, to_numpy(data).ravel())
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    stream.synchronize()
    output = torch.from_numpy(outputs[0].host)
    output = torch.unsqueeze(output, 0)

    test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
    pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    correct += pred.eq(target.view_as(pred)).sum().item()

    # Transfer predictions back from the GPU.
    # inspect the output object here;
    pass
    test_loss /= len(test_loader.dataset)
    print('[trace] TensorRT test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
      test_loss, correct, len(test_loader.dataset),
      100. * correct / len(test_loader.dataset)))

  pass
