local utils = require "utils"

-- Convenience layers
function convRelu(model, inputLayers, hiddenLayers, cnnKernel, cnnStride, cnnPad)
  model:add(nn.SpatialConvolution(inputLayers, hiddenLayers, cnnKernel, cnnKernel, cnnStride, cnnStride, cnnPad, cnnPad))
  model:add(nn.ReLU())
end

function convReluPool(model, inputLayers, hiddenLayers, cnnKernel, cnnStride, cnnPad, poolKernel, poolStride, poolPad)
  model:add(nn.SpatialConvolution(inputLayers, hiddenLayers, cnnKernel, cnnKernel, cnnStride, cnnStride, cnnPad, cnnPad))
  model:add(nn.ReLU())
  model:add(nn.SpatialMaxPooling(poolKernel, poolKernel, poolStride, poolStride, poolPad, poolPad))
end

function model(kwargs)
  assert(kwargs ~= nil)

  local numClasses = utils.getKwarg(kwargs, "numClasses")
  local numChannels = utils.getKwarg(kwargs, "numChannels")
  local scale = utils.getKwarg(kwargs, "scale")

  local cnn = {
    conv1Channels = math.floor(96 * scale), -- 96; 384x384 input image -> 95; pool 95 -> 47
    conv1Kernel = 11, -- 256 -> 63; pool 63 -> 31
    conv1Stride = 4,
    conv1Pad = 2,
    pool1Kernel = 3,
    pool1Stride = 2,
    pool1Pad = 0,

    conv2Channels = math.floor(256 * scale), -- 256; 47 -> 47; pool 47 -> 23
    conv2Kernel = 5, -- 31 -> 31; pool 31 -> 15
    conv2Stride = 1,
    conv2Pad = 2,
    pool2Kernel = 3,
    pool2Stride = 2,
    pool2Pad = 0,

    conv3Channels = math.floor(384 * scale), -- 384; 23 -> 23
    conv3Kernel = 3, -- 15 -> 15
    conv3Stride = 1,
    conv3Pad = 1,

    conv4Channels = math.floor(384 * scale), -- 384; 23 -> 23
    conv4Kernel = 3, -- 15 -> 15
    conv4Stride = 1,
    conv4Pad = 1,

    conv5Channels = math.floor(256 * scale), -- 256; 23 -> 23, pool 23 -> 11
    conv5Kernel = 3, -- 15 -> 15; pool 15 -> 7
    conv5Stride = 1,
    conv5Pad = 1,
    pool5Kernel = 3,
    pool5Stride = 2,
    pool5Pad = 0,

    fc6Channels = math.floor(4096 * scale), -- 4096
    fc7Channels = math.floor(4096 * scale)  -- 4096
  }

  local model = nn.Sequential()
  convReluPool(model, numChannels, cnn.conv1Channels, cnn.conv1Kernel, cnn.conv1Stride, cnn.conv1Pad,
    cnn.pool1Kernel, cnn.pool1Stride, cnn.pool1Pad)
  convReluPool(model, cnn.conv1Channels, cnn.conv2Channels, cnn.conv2Kernel, cnn.conv2Stride, cnn.conv2Pad,
    cnn.pool2Kernel, cnn.pool2Stride, cnn.pool2Pad)
  convRelu(model, cnn.conv2Channels, cnn.conv3Channels, cnn.conv3Kernel, cnn.conv3Stride, cnn.conv3Pad)
  convRelu(model, cnn.conv3Channels, cnn.conv4Channels, cnn.conv4Kernel, cnn.conv4Stride, cnn.conv4Pad)
  convReluPool(model, cnn.conv4Channels, cnn.conv5Channels, cnn.conv5Kernel, cnn.conv5Stride, cnn.conv5Pad,
    cnn.pool5Kernel, cnn.pool5Stride, cnn.pool5Pad)
  model:add(nn.View(cnn.conv5Channels * 7 * 7))
  model:add(nn.Dropout(0.5))
  model:add(nn.Linear(cnn.conv5Channels * 7 * 7, cnn.fc6Channels))
  model:add(nn.Threshold(0, 1e-6))
  model:add(nn.Dropout(0.5))
  model:add(nn.Linear(cnn.fc6Channels, cnn.fc7Channels))
  model:add(nn.Threshold(0, 1e-6))
  model:add(nn.Linear(cnn.fc7Channels, numClasses))
  model:add(nn.LogSoftMax())

  return model
end
