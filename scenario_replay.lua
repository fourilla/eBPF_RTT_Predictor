math.randomseed(os.time())

local upload_body = string.rep("X", 10 * 1024) 

local domains = {
   { name = "www.muni.cz", weight = 302 },      
   { name = "cdn.muni.cz", weight = 297 },     
   { name = "www.fss.muni.cz", weight = 48 },  
   { name = "www.econ.muni.cz", weight = 41 },  
   { name = "webcentrum.muni.cz", weight = 28 },
   { name = "other.muni.cz", weight = 284 }    
}

local function get_weighted_domain()
   local rand = math.random(1000) 
   local current = 0
   for _, d in ipairs(domains) do
      current = current + d.weight
      if rand <= current then return d.name end
   end
   return "www.muni.cz"
end

request = function()
   local headers = {}
   headers["Host"] = get_weighted_domain()
   
   if math.random(100) <= 62 then
      headers["Connection"] = "close"
   else
      headers["Connection"] = "keep-alive"
   end

   if math.random(100) <= 5 then
      headers["Content-Type"] = "application/octet-stream"
      return wrk.format("POST", "/upload_huge", headers, upload_body)
   else
      return wrk.format("GET", "/", headers, nil)
   end
end