math.randomseed(os.time())

local backup_body = string.rep("X", 10 * 1024) 

local domains = {
   { name = "www.muni.cz", weight = 923 },        
   { name = "cdn.muni.cz", weight = 889 },        
   { name = "www.econ.muni.cz", weight = 125 }, 
   { name = "www.fss.muni.cz", weight = 122 },   
   { name = "poradna.fss.muni.cz", weight = 81 }, 
   { name = "it.muni.cz", weight = 71 },         
   { name = "other.muni.cz", weight = 390 }      
}

local function get_weighted_domain()
   local rand = math.random(2600)
   local current = 0
   for _, d in ipairs(domains) do
      current = current + d.weight
      if rand <= current then return d.name end
   end
   return "www.muni.cz" -- 기본값
end

request = function()
   local headers = {}
   headers["Host"] = get_weighted_domain()
   
   if math.random(100) <= 40 then
      headers["Connection"] = "close"
   else
      headers["Connection"] = "keep-alive"
   end

   if math.random(100) <= 5 then
      headers["Content-Type"] = "application/octet-stream"
      return wrk.format("POST", "/upload_backup", headers, backup_body)
   else
      return wrk.format("GET", "/", headers, nil)
   end
end