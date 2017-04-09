-- DATALOADER

require "nn"
require "torch"
require "image"
require "paths"

local utils = require "utils"

GLASS = 1
PAPER = 2
CARDBOARD = 3
PLASTIC = 4
METAL = 5
TRASH = 6

local DataLoader = torch.class("DataLoader")

function DataLoader:__init(kwargs)
  self.splits = {
    train = {},
    val = {},
    test = {}
  }
  self.splits.train.list = utils.getKwarg(kwargs, "trainList")
  self.splits.test.list = utils.getKwarg(kwargs, "testList")
  self.splits.val.list = utils.getKwarg(kwargs, "valList")

  self.opt = {
    inputHeight = utils.getKwarg(kwargs, "inputHeight"),
    inputWidth = utils.getKwarg(kwargs, "inputWidth"),
    scaledHeight = utils.getKwarg(kwargs, "scaledHeight"),
    scaledWidth = utils.getKwarg(kwargs, "scaledWidth"),
    numChannels = utils.getKwarg(kwargs, "numChannels"),
    batchSize = utils.getKwarg(kwargs, "batchSize"),
    dataFolder = utils.getKwarg(kwargs, "dataFolder")
  }

  for split, _ in pairs(self.splits) do
    self.splits[split].index = 1
    self.splits[split].file = paths.basename(self.splits[split].list)
    self.splits[split].filePaths, self.splits[split].labels = loadList(self.splits[split].list, self.opt)
    self.splits[split].count = #self.splits[split].filePaths
  end

  self.meanImage = getMeanTrainingImage(self.splits.train.filePaths, self.opt)
end

function DataLoader:nextBatch(split, augment)
  assert(split == "train" or split == "val" or split == "test")

  local imageData = {}
  local imageLabels = {}

  while #imageData < self.opt.batchSize do
    local index = self.splits[split].index
    local imagePath = self.splits[split].filePaths[index]
    local imageLabel = self.splits[split].labels[index]

    local imageTensor = image.load(imagePath, self.opt.numChannels, "double")
    imageTensor = image.scale(imageTensor, "%dx%d" % {self.opt.scaledHeight, self.opt.scaledWidth})
    imageTensor = imageTensor - self.meanImage
    if split == "train" and augment == true then
      -- imageTensor = warp(imageTensor, torch.random(0, 3), 0.05)
      local transform = torch.random(1, 4)
      if transform == 1 then
        imageTensor = randomCrop(imageTensor, math.floor(self.opt.scaledHeight / 20))
      elseif transform == 2 then
        imageTensor = horizontalFlip(imageTensor, 0.5)
      elseif transform == 3 then
        imageTensor = addNoise(imageTensor, torch.uniform(-5, 5))
      end
    end

    imageTensor = imageTensor:double()
    imageTensor = imageTensor:reshape(1, self.opt.numChannels, self.opt.scaledHeight, self.opt.scaledWidth)
    table.insert(imageData, imageTensor)
    table.insert(imageLabels, torch.Tensor({imageLabel}))
    self.splits[split].index = self.splits[split].index + 1
    if self.splits[split].index > self.splits[split].count then
      self.splits[split].index = 1
      break
    end
  end

  collectgarbage()
  local batch = {
    data = torch.cat(imageData, 1):type("torch.FloatTensor"),
    labels = torch.cat(imageLabels, 1):type("torch.FloatTensor"),
  }

  setmetatable(batch,
    {__index = function(t, k)
                    return {t.data[k], t.labels[k]}
                end}
  );

  function batch:size()
    return self.data:size(1)
  end

  return batch
end

-- Scale and Rotation augmentation (warping)
function warp(input, augRot, augScale)
  -- A nice function of scale is 0.05 (stddev of scale change),
  -- and a nice value for ration is a few degrees or more if your dataset allows for it

  local width = input:size(3)
  local height = input:size(2)

  -- Scale <0=zoom in(+rand crop), >0=zoom out
  local scale_x = 0
  local scale_y = 0
  local move_x = 0
  local move_y = 0
  if augScale > 0 then
    scale_x = torch.normal(0, augScale) -- normal distribution
    -- Given a zoom in or out, we move around our canvas.
    scale_y = scale_x -- keep aspect ratio the same
    move_x = torch.uniform(-scale_x, scale_x)
    move_y = torch.uniform(-scale_y, scale_y)
  end

  -- Angle of rotation
  local rot_angle = torch.uniform(-augRot,augRot) -- (degrees) uniform distribution [-augRot : augRot)

  -- x/y grids
  local grid_x = torch.ger( torch.ones(height), torch.linspace(-1-scale_x,1+scale_x,width) )
  local grid_y = torch.ger( torch.linspace(-1-scale_y,1+scale_y,height), torch.ones(width) )

  local flow = torch.FloatTensor()
  flow:resize(2,height,width)
  flow:zero()

  -- Apply scale
  flow_scale = torch.FloatTensor()
  flow_scale:resize(2,height,width)
  flow_scale[1] = grid_y
  flow_scale[2] = grid_x
  flow_scale[1]:add(1+move_y):mul(0.5) -- move ~[-1 1] to ~[0 1]
  flow_scale[2]:add(1+move_x):mul(0.5) -- move ~[-1 1] to ~[0 1]
  flow_scale[1]:mul(height-1)
  flow_scale[2]:mul(width-1)
  flow:add(flow_scale)

  if augRot > 0 then
    -- Apply rotation through rotation matrix
    local flow_rot = torch.FloatTensor()
    flow_rot:resize(2,height,width)
    flow_rot[1] = grid_y * ((height-1)/2) * -1
    flow_rot[2] = grid_x * ((width-1)/2) * -1
    view = flow_rot:reshape(2,height*width)
    local function rmat(deg)
      local r = deg/180*math.pi
      return torch.FloatTensor{{math.cos(r), -math.sin(r)}, {math.sin(r), math.cos(r)}}
    end

    local rotmat = rmat(rot_angle)
    local flow_rotr = torch.mm(rotmat, view)
    flow_rot = flow_rot - flow_rotr:reshape( 2, height, width )
    flow:add(flow_rot)
  end

  return image.warp(input, flow, "bilinear", false)
end

function randomCrop(input, size)
  local w, h = input:size(3), input:size(2)
  if w == size and h == size then
     return input
  end

  local x1, y1 = torch.random(1, w - size), torch.random(1, h - size)
  input[{{}, {x1, x1 + size}, {y1, y1 + size}}] = 0
  return input
end

function horizontalFlip(input, prob)
  if torch.uniform() < prob then
    return image.hflip(input)
  end

  return input
end

-- Adds noise to the image
-- ref: https://github.com/brainstorm-ai/DIGITS/blob/6a150cfbed2aa7dd70992036dfbdf66ee088fba0/tools/torch/data.lua#L135
function addNoise(input, augNoise)
  -- AWGN:
  -- torch.randn makes noise with mean 0 and variance 1 (=stddev 1)
  --  so we multiply the tensor with our augNoise factor, that has a linear relation with
  --  the standard deviation (but the variance will be increased quadratically).
  return torch.add(input:float(), torch.randn(input:size()):float()*augNoise)
end

function loadList(fileListPath, opt)
  local filePaths = {}
  local fileLabels = {}
  local file, err = io.open(fileListPath, "rb")
  if err then
    utils.printTime(err)
    return
  else
    while true do
      local line = file:read()
      if line == nil then
        break
      end

      -- get tokens from line containing video path and label
      local tokens = {}
      for token in string.gmatch(line, "[^%s]+") do
        table.insert(tokens, token)
      end

      local filePath, fileLabel = unpack(tokens)
      fileLabel = tonumber(fileLabel)
      if fileLabel == GLASS then
        filePath = paths.concat(opt.dataFolder, "glass", filePath)
      elseif fileLabel == PAPER then
        filePath = paths.concat(opt.dataFolder, "paper", filePath)
      elseif fileLabel == CARDBOARD then
        filePath = paths.concat(opt.dataFolder, "cardboard", filePath)
      elseif fileLabel == PLASTIC then
        filePath = paths.concat(opt.dataFolder, "plastic", filePath)
      elseif fileLabel == METAL then
        filePath = paths.concat(opt.dataFolder, "metal", filePath)
      elseif fileLabel == TRASH then
        filePath = paths.concat(opt.dataFolder, "trash", filePath)
      end

      table.insert(filePaths, filePath)
      table.insert(fileLabels, fileLabel)
    end
  end

  return filePaths, fileLabels
end

function getMeanTrainingImage(filePaths, opt)
  local means = {0, 0, 0}
  local numImages = 0

  for i, filePath in pairs(filePaths) do
    collectgarbage()
    numImages = numImages + 1
    local img = image.load(filePath, opt.numChannels, "double")
    img = image.scale(img, "%dx%d" % {opt.scaledHeight, opt.scaledWidth})
    for channel = 1, opt.numChannels do
      means[channel] = means[channel] + (img[channel]:mean() - means[channel]) / numImages
    end
  end

  local meanImage = torch.Tensor(opt.numChannels, opt.scaledHeight, opt.scaledWidth)
  for channel = 1, opt.numChannels do
    meanImage[channel]:fill(means[channel])
  end

  collectgarbage()
  return meanImage
end
