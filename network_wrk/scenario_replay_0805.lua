math.randomseed(os.time())

local upload_body = string.rep("X", 10 * 1024) 

local domains = {
   { name = "cdn.muni.cz", weight = 1364 },      
   { name = "www.muni.cz", weight = 1050 },   
   { name = "www.econ.muni.cz", weight = 130 },
   { name = "www.fss.muni.cz", weight = 128 },  
   { name = "poradna.fss.muni.cz", weight = 79 },
   { name = "it.muni.cz", weight = 72 },   
   { name = "other.muni.cz", weight = 1486 }     
}

local function get_weighted_domain()
   local rand = math.random(4309)
   local current = 0
   for _, d in ipairs(domains) do
      current = current + d.weight
      if rand <= current then return d.name end
   end
   return "cdn.muni.cz"
end

request = function()
   local headers = {}
   headers["Host"] = get_weighted_domain()

   if math.random(100) <= 36 then
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