-- TRAIN

require "torch"
require "nn"
require "image"
require "optim"

require "model"
require "DataLoader"
local utils = require "utils"

local cmd = torch.CmdLine()

-- Dataset options
cmd:option("-trainList", "data/one-indexed-files-notrash_train.txt") -- necessary
cmd:option("-valList", "data/one-indexed-files-notrash_val.txt") -- necessary
cmd:option("-testList", "data/one-indexed-files-notrash_test.txt") -- necessary
cmd:option("-numClasses", 5) -- necessary
cmd:option("-inputHeight", "384")
cmd:option("-inputWidth", "384")
cmd:option("-scaledHeight", "256") -- uses original height if unprovided
cmd:option("-scaledWidth", "256") -- uses original width if unprovided
cmd:option("-numChannels", 3)
cmd:option("-batchSize", 32)
cmd:option("-dataFolder", "data/pics")

-- Optimization options
cmd:option("-numEpochs", 100)
cmd:option("-learningRate", 1.25e-5) -- 2.5e-5 works well; 1e-5 is second best
cmd:option("-lrDecayFactor", 0.9, "newLR = oldLR * <lrDecayFactor>")
cmd:option("-lrDecayEvery", 20, "learning rate is decayed every <lrDecayEver> epochs")
cmd:option("-weightDecay", 2.5e-2, "L2 regularization")
cmd:option("-weightInitializationMethod", "kaiming", "heuristic, xavier, xavier_caffe, or none")

-- Output options
cmd:option("-printEvery", 1, "prints and saves the train and val acc and loss every <printEvery> epochs")
cmd:option("-checkpointEvery", 20, "saves a snapshot of the model every <checkpointEvery> epochs")
cmd:option("-checkpointName", "checkpoints/checkpoint", "checkpoint will be saved at ./<checkpointName>_#.t7")

-- Backend options
cmd:option("-cuda", 1)
cmd:option("-gpu", 0)
cmd:option("-scale", 1, "proportion of filters used in the architecture")

local opt = cmd:parse(arg)

-- Torch cmd parses user input as strings so we need to convert number strings to numbers
for k, v in pairs(opt) do
  if tonumber(v) then
    opt[k] = tonumber(v)
  end
end

assert(opt.trainList ~= "", "Need a list of train items.")
assert(opt.valList ~= "", "Need a list of val items.")
assert(opt.testList ~= "", "Need a list of test items.")
assert(opt.numClasses ~= "", "Need the number of image classes.")
assert(opt.dataFolder ~= "", "Need the folder relative to this file where the pictures are stored.")

if opt.scaledHeight == "" then
  opt.scaledHeight = opt.inputHeight
end

if opt.scaledWidth == "" then
  opt.scaledWidth = opt.inputWidth
end

-- Set up GPU
opt.dtype = "torch.FloatTensor"
if opt.gpu >= 0 and opt.cuda == 1 then
  require "cunn"
  require "cutorch"
  opt.dtype = "torch.CudaTensor"
  cutorch.setDevice(opt.gpu + 1)
end

-- Initialize DataLoader to receive batch data
utils.printTime("Initializing DataLoader")
local loader = DataLoader(opt)

-- Initialize model and criterion
utils.printTime("Initializing model and criterion")
local model = model(opt):type(opt.dtype)
if opt.weightInitializationMethod ~= "none" then
  model = require("weight-init")(model, opt.weightInitializationMethod)
end
local criterion = nn.ClassNLLCriterion():type(opt.dtype)

-- Initialize history tables
trainLossHistory = {}
trainAccHistory = {}
valLossHistory = {}
valAccHistory = {}
epochs = {}

--[[
  Input:
    - model: a CNN

  Trains a fresh CNN from end to end. Uses the opt parameters declared above.
]]--
function train(model)
  utils.printTime("Starting training for %d epochs" % opt.numEpochs)

  local config = {
    learningRate = opt.learningRate,
    weightDecay = opt.weightDecay
  }

  local params, gradParams = model:getParameters()

  local feval = function(x)
    assert(x == params)
    gradParams:zero()

    local batch = loader:nextBatch("train", true)
    if opt.cuda == 1 then
      batch.data = batch.data:cuda()
      batch.labels = batch.labels:cuda()
    end

    local scores = model:forward(batch.data) -- opt.batchSize x NUM_CLASSES
    local loss = criterion:forward(scores, batch.labels)
    local gradScores = criterion:backward(scores, batch.labels) -- opt.batchSize x NUM_CLASSES
    model:backward(batch.data, gradScores)

    return loss, gradParams
  end

  local epochLoss = {}
  local iterationsPerEpoch = math.ceil(loader.splits.train.count / opt.batchSize)
  local numIterations = opt.numEpochs * iterationsPerEpoch

  -- Turn on Dropout
  model:training()

  for i = 1, numIterations do
    collectgarbage()

    local epoch = math.floor((i - 1) / iterationsPerEpoch) + 1
    local _, loss = optim.adam(feval, params, config)
    table.insert(epochLoss, loss[1])

    local iterationCompleted = i % iterationsPerEpoch
    if iterationCompleted == 0 then
      iterationCompleted = iterationsPerEpoch
    end

    if iterationCompleted % 10 == -1 then
      utils.printTime("Epoch %d/%d: finished %d/%d iterations" % {epoch, opt.numEpochs, iterationCompleted, iterationsPerEpoch})
    end

    -- end of an epoch
    if #epochLoss % iterationsPerEpoch == 0 then
      if epoch % opt.lrDecayEvery == 0 then
        local oldLearningRate = config.learningRate
        config = {
          learningRate = oldLearningRate * opt.lrDecayFactor,
          weightDecay = opt.weightDecay
        }
      end

      -- Calculate and print the epoch loss
      epochLoss = torch.mean(torch.Tensor(epochLoss))

      if (opt.printEvery > 0 and epoch % opt.printEvery == 0) then

        -- Add current epoch number to history
        table.insert(epochs, epoch)

        local _, trainAcc, _ = test(model, "train")
        table.insert(trainLossHistory, epochLoss)
        table.insert(trainAccHistory, trainAcc)

        local valLoss, valAcc, _ = test(model, "val")
        table.insert(valLossHistory, valLoss)
        table.insert(valAccHistory, valAcc)
        utils.printTime("Epoch %d/%d: train acc: %f, train loss: %f, val acc: %f, val loss: %f" % {epoch, opt.numEpochs, trainAcc, epochLoss, valAcc, valLoss})

        -- Turn Dropout back on
        model:training()
      end

      -- Clear this table for the next epoch
      epochLoss = {}

      -- Save a checkpoint of the model, its opt parameters, the training loss history, and the testing loss history
      if (opt.checkpointEvery > 0 and epoch % opt.checkpointEvery == 0) or epoch == opt.numEpochs then
        local checkpoint = {
          opt = opt,
          trainLossHistory = trainLossHistory,
          trainAccHistory = trainAccHistory,
          valLossHistory = valLossHistory,
          valAccHistory = valAccHistory,
          epochs = epochs
        }

        local filename
        if epoch == opt.numEpochs then
          filename = "%s_%s.t7" % {opt.checkpointName, "final"}
        else
          filename = "%s_%d.t7" % {opt.checkpointName, epoch}
        end

        -- Make sure the output directory exists before we try to write it
        paths.mkdir(paths.dirname(filename))

        -- Clear intermediate states in the model before saving to disk to save memory
        model:clearState()

        -- Cast model to float so it can be used on CPU
        model:float()
        checkpoint.model = model
        torch.save(filename, checkpoint)

        -- Cast model back so that it can continue to be used
        model:type(opt.dtype)
        params, gradParams = model:getParameters()
        -- utils.printTime("Saved checkpoint model for epoch %d and opt at %s" % {epoch, filename})
        collectgarbage()
      end
    end
  end

  utils.printTime("Finished training")
end

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
  -- utils.printTime("Starting evaluation on the %s split" % split)

  -- Turn off Dropout
  model:evaluate()

  local confusion = optim.ConfusionMatrix(opt.numClasses)
  local evalData = {
    predictedLabels = {},
    trueLabels = {},
    loss = {}
  }

  local numIterations = math.ceil(loader.splits[split].count / opt.batchSize)
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

for k, v in pairs(opt) do
   utils.printTime("%s = %s" % {k, v})
end

train(model)
utils.printTime("Final accuracy on the train set: %f" % trainAccHistory[#trainAccHistory])
utils.printTime("Final accuracy on the val set: %f" % valAccHistory[#valAccHistory])

local _, testAcc, testConfusion = test(model, "test", True)
utils.printTime("Final accuracy on the test set: %f" % testAcc)
print(testConfusion)
