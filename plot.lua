-- PLOT RESULTS

require "torch"
require "nn"
require "gnuplot"

local cmd = torch.CmdLine()

-- Options
cmd:option("-checkpoint", "checkpoints/checkpoint11/checkpoint_final.t7")
cmd:option("-outputDir", "checkpoints/checkpoint11")

local opt = cmd:parse(arg)

assert(opt.checkpoint ~= "", "Need a trained network file to load.")

local checkpoint = torch.load(opt.checkpoint)
local trainLossHistory = torch.Tensor(checkpoint.trainLossHistory)
local trainAccHistory = torch.Tensor(checkpoint.trainAccHistory)
local valLossHistory = torch.Tensor(checkpoint.valLossHistory)
local valAccHistory = torch.Tensor(checkpoint.valAccHistory)
local epochs = torch.Tensor(checkpoint.epochs)

assert(epochs:size()[1] == trainLossHistory:size()[1], "The number of epochs must correspond to the number of train loss history points.")
assert(epochs:size()[1] == trainAccHistory:size()[1], "The number of epochs must correspond to the number of train accuracy history points.")
assert(epochs:size()[1] == valLossHistory:size()[1], "The number of epochs must correspond to the number of val loss history points.")
assert(epochs:size()[1] == valAccHistory:size()[1], "The number of epochs must correspond to the number of val accuracy history points.")

gnuplot.pngfigure(paths.concat(opt.outputDir, "training-loss.png"))
gnuplot.title("Training Loss")
gnuplot.xlabel("epoch")
gnuplot.ylabel("loss")
gnuplot.plot(epochs, trainLossHistory, "-")
gnuplot.plotflush()

gnuplot.pngfigure(paths.concat(opt.outputDir, "accuracy.png"))
gnuplot.title("Accuracy Fitting")
gnuplot.xlabel("epoch")
gnuplot.ylabel("accuracy")
gnuplot.plot(
  {"Training", epochs, trainAccHistory, "-"},
  {"Validation", epochs, valAccHistory, "-"})
gnuplot.plotflush()
