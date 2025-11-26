math.randomseed(os.time())

local domains = {
   { name = "cdn.muni.cz", weight = 1505 },      
   { name = "www.muni.cz", weight = 1047 },     
   { name = "www.skm.muni.cz", weight = 136 },    
   { name = "www.econ.muni.cz", weight = 135 },   
   { name = "www.fss.muni.cz", weight = 130 },   
   { name = "webcentrum.muni.cz", weight = 85 },  
   { name = "other.muni.cz", weight = 1584 }     
}

local function get_weighted_domain()
   local rand = math.random(4622) 
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

   if math.random(100) <= 45 then
      headers["Connection"] = "close"
   else
      headers["Connection"] = "keep-alive"
   end

   if math.random(100) <= 10 then
      return wrk.format("GET", "/large_asset_download_2GB_simulation", headers, nil)
   else
      return wrk.format("GET", "/", headers, nil)
   end
end