math.randomseed(os.time())

local large_body = string.rep("X", 10 * 1024) 

local domains = {
   { name = "cdn.muni.cz", weight = 771 },       
   { name = "www.muni.cz", weight = 683 },    
   { name = "www.econ.muni.cz", weight = 86 },   
   { name = "www.fss.muni.cz", weight = 84 },  
   { name = "poradna.fss.muni.cz", weight = 51 },
   { name = "other.muni.cz", weight = 125 }      
}

local function get_weighted_domain()
   local rand = math.random(1800) 
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
   
   if math.random(100) <= 50 then
      headers["Connection"] = "close"
   else
      headers["Connection"] = "keep-alive"
   end

   if math.random(100) <= 5 then
      headers["Content-Type"] = "application/octet-stream"
      return wrk.format("POST", "/upload", headers, large_body)
   else
      return wrk.format("GET", "/", headers, nil)
   end
end