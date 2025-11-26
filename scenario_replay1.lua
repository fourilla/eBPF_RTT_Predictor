math.randomseed(os.time())

local payload_1kb = string.rep("X", 1024)
local payload_10kb = string.rep("Y", 10 * 1024)
local payload_100kb = string.rep("Z", 100 * 1024)

request = function()
   local headers = {}
   
   if math.random(100) <= 70 then
      headers["Connection"] = "close"
   else
      headers["Connection"] = "keep-alive"
   end

   local r = math.random(100)
   
   if r <= 10 then
       headers["Content-Type"] = "application/octet-stream"
       return wrk.format("POST", "/upload_heavy", headers, payload_100kb)
   elseif r <= 40 then
       return wrk.format("POST", "/upload_mid", headers, payload_10kb)
   else
       return wrk.format("GET", "/fast_query", headers, nil)
   end
end