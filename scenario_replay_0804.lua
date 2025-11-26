math.randomseed(os.time())

local bidirectional_body = string.rep("X", 10 * 1024) 

local domains = {
   { name = "cdn.muni.cz", weight = 1356 },       
   { name = "www.muni.cz", weight = 1051 },       
   { name = "www.fss.muni.cz", weight = 133 },    
   { name = "www.econ.muni.cz", weight = 130 },   
   { name = "poradna.fss.muni.cz", weight = 83 }, 
   { name = "it.muni.cz", weight = 78 },          
   { name = "other.muni.cz", weight = 1633 }      
}

local function get_weighted_domain()
   local rand = math.random(4464) 
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

   if math.random(100) <= 10 then
      headers["Content-Type"] = "application/octet-stream"
      return wrk.format("POST", "/bidirectional_transfer", headers, bidirectional_body)
   else
      return wrk.format("GET", "/", headers, nil)
   end
end