local utils = require 'utils'

local cmd = torch.CmdLine()

cmd:option('-filename', 'data/one-indexed-files.txt')
cmd:option('-format', 'one-indexed-files-notrash_') -- if specified, output filenames will be "%s%s.txt" % {format, split}
cmd:option('-outputDir', 'data') -- usually ''
cmd:option('-train', 0.70)
cmd:option('-val', 0.13)
cmd:option('-test', 0.17)

local opt = cmd:parse(arg)
assert(opt.filename ~= '', "Need a text file consisting of filenames and labels.")
assert(opt.train ~= '' or opt.train ~= 0, "Must have train examples.")
assert(opt.val ~= '' or opt.val ~= 0, "Must have val examples.")
assert(opt.test ~= '' or opt.test ~= 0, "Must have test examples.")

function shuffle(opt)
  local allLines = {}
  local splits = {
    train = {},
    val = {},
    test = {}
  }

  local file, err = io.open(opt.filename, 'r')
  if err then
    utils.printTime(err)
    return
  else
    while true do
      local line = file:read()
      if line == nil then
        break
      end

      table.insert(allLines, line)
    end
  end

  splits.train.count = math.floor(#allLines * opt.train)
  splits.val.count = math.floor(#allLines * opt.val)
  splits.test.count = #allLines - splits.train.count - splits.val.count
  local shufflePerm = torch.randperm(#allLines)
  for split, _ in pairs(splits) do
    local startIndex, endIndex
    if split == 'train' then
      startIndex = 1
      endIndex = splits.train.count
    elseif split == 'val' then
      startIndex = splits.train.count + 1
      endIndex = splits.train.count + splits.val.count
    elseif split == 'test' then
      startIndex = splits.train.count + splits.val.count + 1
      endIndex = -1
    end

    splits[split].indices = shufflePerm[{{startIndex, endIndex}}]
    local file, err = io.open(paths.concat(opt.outputDir, "%s%s.txt" % {opt.format, split}), 'w+')
    if err then
      utils.printTime(err)
      return
    else
      for i = 1, splits[split].count do
        local index = splits[split].indices[i]
        file:write("%s\n" % allLines[index])
      end
      file:close()
    end
  end
end

shuffle(opt)
