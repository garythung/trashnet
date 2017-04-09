require "torch"
require "nn"
require "image"
require "optim"

require "model"
require "DataLoader"
local utils = require "utils"

local cmd = torch.CmdLine()

-- Options
cmd:option("-checkpoint", "checkpoints/checkpoint_final.t7")
cmd:option("-split", "", "train, val, or test. leaving blank runs all splits.")
cmd:option("-cuda", 1)

local opt = cmd:parse(arg)

assert(opt.checkpoint ~= "", "Need a trained network file to load.")
assert(opt.split == "" or opt.split == "train" or opt.split == "val" or opt.split == "test")

-- Set up GPU
opt.dtype = "torch.FloatTensor"
if opt.cuda == 1 then
	require "cunn"
  opt.dtype = "torch.CudaTensor"
end

-- Initialize model and criterion
utils.printTime("Initializing model")
local checkpoint = torch.load(opt.checkpoint)
local model = checkpoint.model
model:type(opt.dtype)
local criterion = nn.ClassNLLCriterion():type(opt.dtype)

-- Initialize DataLoader to receive batch data
utils.printTime("Initializing DataLoader")
local loader = DataLoader(checkpoint.opt)

--[[
  Inputs:
    - model: a CNN
    - split: "train", "val", or "test"

  Outputs:
    - loss: average loss per item in this split
    - accuracy: accuracy on this split
    - confusion: an optim.ConfusionMatrix object

  Performs image classification using a given nn module.
]]--
function test(model, split)
  assert(split == "train" or split == "val" or split == "test")
  collectgarbage()
  utils.printTime("Starting evaluation on the %s split" % split)

  -- Turn off Dropout
  model:evaluate()

  local confusion = optim.ConfusionMatrix(checkpoint.opt.numClasses)
  local evalData = {
    predictedLabels = {},
    trueLabels = {},
    loss = {}
  }

  local numIterations = math.ceil(loader.splits[split].count / checkpoint.opt.batchSize)
  for i = 1, numIterations do
    local batch = loader:nextBatch(split, false)
    if opt.cuda == 1 then
      batch.data = batch.data:cuda()
      batch.labels = batch.labels:cuda()
    end

    local scores = model:forward(batch.data) -- batchSize x numClasses
    local _, predictedLabels = torch.max(scores, 2)
    table.insert(evalData.predictedLabels, predictedLabels:double())
    table.insert(evalData.trueLabels, batch.labels:reshape(batch:size(), 1):double())
    local loss = criterion:forward(scores, batch.labels)
    table.insert(evalData.loss, loss)

    collectgarbage()
  end

  evalData.predictedLabels = torch.cat(evalData.predictedLabels, 1)
  evalData.trueLabels = torch.cat(evalData.trueLabels, 1)
  confusion:batchAdd(evalData.predictedLabels, evalData.trueLabels)
  local loss = torch.mean(torch.Tensor(evalData.loss))
  local accuracy = torch.sum(torch.eq(evalData.predictedLabels, evalData.trueLabels)) / evalData.trueLabels:size()[1]

  return loss, accuracy, confusion
end

if opt.split == "" then
  for _, split in pairs({"train", "val", "test"}) do
    local _, acc, _ = test(model, split)
    utils.printTime("Accuracy on the %s split: %f" % {split, acc})
  end
else
  local _, acc, _ = test(model, opt.split)
  utils.printTime("Accuracy on the %s split: %f" % {opt.split, acc})
end
