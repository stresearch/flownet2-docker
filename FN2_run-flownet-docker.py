#!/usr/bin/env python2.7

from __future__ import print_function

import os, sys, numpy as np
import argparse
from scipy import misc
import caffe
import tempfile
from math import ceil
from datetime import datetime, timedelta


def dockerize_filepath(path):
    """
    Convert a given filepath to be relative to the input-output folder that this
    container gets from the host system.
    """
    return os.path.join('/input-output', path)

def readFlow(name):
    if name.endswith('.pfm') or name.endswith('.PFM'):
        return readPFM(name)[0][:,:,0:2]

    f = open(name, 'rb')

    header = f.read(4)
    if header.decode("utf-8") != 'PIEH':
        raise Exception('Flow file header does not contain PIEH')

    width = np.fromfile(f, np.int32, 1).squeeze()
    height = np.fromfile(f, np.int32, 1).squeeze()

    flow = np.fromfile(f, np.float32, width * height * 2).reshape((height, width, 2))

    return flow.astype(np.float32)

def writeFlow(name, flow):
    f = open(name, 'wb')
    f.write('PIEH'.encode('utf-8'))
    np.array([flow.shape[1], flow.shape[0]], dtype=np.int32).tofile(f)
    flow = flow.astype(np.float32)
    flow.tofile(f)
    f.flush()
    f.close() 

parser = argparse.ArgumentParser()
parser.add_argument('caffemodel', help='path to model')
parser.add_argument('deployproto', help='path to deploy prototxt template')
parser.add_argument('img0', help='image 0 path')
parser.add_argument('img1', help='image 1 path')
parser.add_argument('out',  help='output filename')
parser.add_argument('--gpu',  help='gpu id to use (0, 1, ...)', default=0, type=int)
parser.add_argument('--verbose',  help='whether to output all caffe logging', action='store_true')
parser.add_argument('--skipCheck', help='whether to check for NaN in blobs after running the network', action='store_true')

args = parser.parse_args()

if(not os.path.isfile(args.caffemodel)): raise BaseException('caffemodel does not exist: '+args.caffemodel)
if(not os.path.isfile(args.deployproto)): raise BaseException('deploy-proto does not exist: '+args.deployproto)

if args.img0.endswith('.txt'):
  input_files = [[dockerize_filepath(f.strip()) for f in open(dockerize_filepath(args.img0)).readlines()],
                 [dockerize_filepath(f.strip()) for f in open(dockerize_filepath(args.img1)).readlines()]]
  output_files = [dockerize_filepath(f.strip()) for f in open(dockerize_filepath(args.out)).readlines()]
else:
  input_files = [[dockerize_filepath(args.img0),],
                 [dockerize_filepath(args.img1),]]
  output_files = [dockerize_filepath(args.out),]

totalFileSetupTime = 0.0
totalCaffeSetupTime = 0.0
totalCaffeTime = 0.0
imageCaffeTime = 0.0
totalCheckTime = 0.0
imageCheckTime = 0.0
totalFileOutputTime = 0.0
totalTime = 0.0

startTime = datetime.now()
caffeSetupStartTime = datetime.now()
caffeSetupEndTime = datetime.now()
caffeStartTime = datetime.now()
checkEndTime = datetime.now()
fileOutputTime = 0.0
for i in range(len(output_files)):
  in0 = input_files[0][i]
  in1 = input_files[1][i]
  out = output_files[i]
  imageCaffeTime = 0.0
  imageCheckTime = 0.0

  startTime = datetime.now()
  if(not os.path.isfile(in0)): raise BaseException('img0 does not exist: '+in0)
  if(not os.path.isfile(in1)): raise BaseException('img1 does not exist: '+in1)
  num_blobs = 2
  input_data = []
  img0 = misc.imread(in0)
  if len(img0.shape) < 3: input_data.append(img0[np.newaxis, np.newaxis, :, :])
  else:                   input_data.append(img0[np.newaxis, :, :, :].transpose(0, 3, 1, 2)[:, [2, 1, 0], :, :])
  img1 = misc.imread(in1)
  if len(img1.shape) < 3: input_data.append(img1[np.newaxis, np.newaxis, :, :])
  else:                   input_data.append(img1[np.newaxis, :, :, :].transpose(0, 3, 1, 2)[:, [2, 1, 0], :, :])

  if i == 0:
    width = input_data[0].shape[3]
    height = input_data[0].shape[2]
    vars = {}
    vars['TARGET_WIDTH'] = width
    vars['TARGET_HEIGHT'] = height

    divisor = 64.
    vars['ADAPTED_WIDTH'] = int(ceil(width/divisor) * divisor)
    vars['ADAPTED_HEIGHT'] = int(ceil(height/divisor) * divisor)

    vars['SCALE_WIDTH'] = width / float(vars['ADAPTED_WIDTH']);
    vars['SCALE_HEIGHT'] = height / float(vars['ADAPTED_HEIGHT']);

    tmp = tempfile.NamedTemporaryFile(mode='w', delete=True)
    proto = open(args.deployproto).readlines()
    for line in proto:
        for key, value in vars.items():
            tag = "$%s$" % key
            line = line.replace(tag, str(value))
        tmp.write(line)
    tmp.flush()

    caffeSetupStartTime = datetime.now()
    if not args.verbose:
        caffe.set_logging_disabled()
    caffe.set_device(args.gpu)
    caffe.set_mode_gpu()
    net = caffe.Net(tmp.name, args.caffemodel, caffe.TEST)
  else:
    caffeSetupStartTime = datetime.now()

  input_dict = {}
  for blob_idx in range(num_blobs):
      input_dict[net.inputs[blob_idx]] = input_data[blob_idx]

  #
  # There is some non-deterministic nan-bug in caffe
  # it seems to be a race-condition 
  #
  print('Network forward pass using %s.' % args.caffemodel)
  i = 1
  caffeSetupEndTime = datetime.now()
  while i<=5:
      i+=1

      caffeStartTime = datetime.now()
      net.forward(**input_dict)
      caffeEndTime = datetime.now()
      imageCaffeTime = imageCaffeTime + (caffeEndTime - caffeStartTime).total_seconds()

      containsNaN = False
      if not args.skipCheck:
          for name in net.blobs:
              blob = net.blobs[name]
              has_nan = np.isnan(blob.data[...]).any()
              
              if has_nan:
                  print('blob %s contains nan' % name)
                  containsNaN = True
          checkEndTime = datetime.now()
          imageCheckTime = imageCheckTime + (checkEndTime - caffeEndTime).total_seconds()
      else:
          checkEndTime = datetime.now()
      if not containsNaN:
          print('Succeeded.')
          break
      else:
          print('**************** FOUND NANs, RETRYING ****************')

  blob = np.squeeze(net.blobs['predict_flow_final'].data).transpose(1, 2, 0)

  writeFlow(out, blob)
  finished = datetime.now()
  fileSetupTime = (caffeSetupStartTime - startTime).total_seconds()
  totalFileSetupTime = totalFileSetupTime + fileSetupTime
  caffeSetupTime = (caffeSetupEndTime - caffeSetupStartTime).total_seconds()
  totalCaffeSetupTime = totalCaffeSetupTime + caffeSetupTime
  totalCaffeTime = totalCaffeTime + imageCaffeTime
  totalCheckTime = totalCheckTime + imageCheckTime
  fileOutputTime = (finished - checkEndTime).total_seconds()
  totalFileOutputTime = totalFileOutputTime + fileOutputTime
  imageTotalTime = (finished - startTime).total_seconds()
  totalTime = totalTime + imageTotalTime
  
  print('{}: File setup: {:.6f}; caffe setup: {:.6f}; caffe: {:.6f}; check: {:.6f}; file output: {:.6f}; total: {:.6f}'.format(i, fileSetupTime, caffeSetupTime, imageCaffeTime, imageCheckTime, fileOutputTime, imageTotalTime))

print('TOTAL: File setup: {:.6f}; caffe setup: {:.6f}; caffe: {:.6f}; check: {:.6f}; file output: {:.6f}; total: {:.6f}'.format(totalFileSetupTime, totalCaffeSetupTime, totalCaffeTime, totalCheckTime, totalFileOutputTime, totalTime))
