math.randomseed(os.time())

local large_body = string.rep("X", 10 * 1024) 

[cite_start]
local domains = {
   { name = "cdn.muni.cz", weight = 1184 },      
   { name = "www.muni.cz", weight = 990 },       
   { name = "www.fss.muni.cz", weight = 125 },    
   { name = "www.econ.muni.cz", weight = 120 },  
   { name = "poradna.fss.muni.cz", weight = 80 }, 
   { name = "it.muni.cz", weight = 70 },         
   { name = "other.muni.cz", weight = 1433 }  
}

local function get_weighted_domain()
   local rand = math.random(4000) 
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
   
   [cite_start]
   if math.random(100) <= 36 then
      headers["Connection"] = "close"
   else
      headers["Connection"] = "keep-alive"
   end

   if math.random(100) <= 5 then
      headers["Content-Type"] = "application/octet-stream"
      return wrk.format("POST", "/upload_sync", headers, large_body)
   else
      return wrk.format("GET", "/", headers, nil)
   end
end