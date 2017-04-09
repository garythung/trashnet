local utils = {}

function utils.getKwarg(kwargs, name, default)
  if kwargs == nil then kwargs = {} end
  if kwargs[name] == nil and default == nil then
    assert(false, string.format("'%s' expected and not given", name))
  elseif kwargs[name] == nil then
    return default
  else
    return kwargs[name]
  end
end

--[[
  Prints the time and a message in the form of

  <time> <message>

  example: 08:58:23 Hello World!
]]--
function utils.printTime(message)
  local timeObject = os.date("*t")
  local currTime = ("%02d:%02d:%02d"):format(timeObject.hour, timeObject.min, timeObject.sec)
  print("%s %s" % {currTime, message})
end

return utils
