-- NestedTensor: Utility module for nested tensor operations
-- Provides utilities for working with nested tensor structures in nnn (nested neural nets)

local NestedTensor = {}

-- Clone a nested tensor structure
function NestedTensor.clone(obj)
   if torch.isTensor(obj) then
      return obj:clone()
   end
   
   if type(obj) ~= 'table' then
      return obj
   end
   
   local result = {}
   for k, v in pairs(obj) do
      result[k] = NestedTensor.clone(v)
   end
   return result
end

-- Fill nested tensor structure with value
function NestedTensor.fill(obj, value)
   if torch.isTensor(obj) then
      obj:fill(value)
      return obj
   end
   
   if type(obj) == 'table' then
      for k, v in pairs(obj) do
         NestedTensor.fill(v, value)
      end
   end
   return obj
end

-- Resize nested tensor structure
function NestedTensor.resizeAs(output, input)
   if torch.isTensor(output) and torch.isTensor(input) then
      output:resizeAs(input)
      return output
   end
   
   if type(output) == 'table' and type(input) == 'table' then
      for k, v in pairs(input) do
         if not output[k] then
            output[k] = NestedTensor.clone(v)
         else
            NestedTensor.resizeAs(output[k], v)
         end
      end
      -- Remove extra keys
      for k in pairs(output) do
         if not input[k] then
            output[k] = nil
         end
      end
   end
   return output
end

-- Copy nested tensor structure
function NestedTensor.copy(output, input)
   if torch.isTensor(output) and torch.isTensor(input) then
      output:copy(input)
      return output
   end
   
   if type(output) == 'table' and type(input) == 'table' then
      for k, v in pairs(input) do
         if output[k] then
            NestedTensor.copy(output[k], v)
         end
      end
   end
   return output
end

-- Add nested tensors
function NestedTensor.add(output, input)
   if torch.isTensor(output) and torch.isTensor(input) then
      output:add(input)
      return output
   end
   
   if type(output) == 'table' and type(input) == 'table' then
      for k, v in pairs(input) do
         if output[k] then
            NestedTensor.add(output[k], v)
         end
      end
   end
   return output
end

-- Get depth of nested structure
function NestedTensor.depth(obj)
   if torch.isTensor(obj) then
      return 0
   end
   
   if type(obj) ~= 'table' then
      return 0
   end
   
   local maxDepth = 0
   for _, v in pairs(obj) do
      local d = NestedTensor.depth(v)
      if d > maxDepth then
         maxDepth = d
      end
   end
   return maxDepth + 1
end

-- Count total number of tensors in nested structure
function NestedTensor.count(obj)
   if torch.isTensor(obj) then
      return 1
   end
   
   if type(obj) ~= 'table' then
      return 0
   end
   
   local total = 0
   for _, v in pairs(obj) do
      total = total + NestedTensor.count(v)
   end
   return total
end

-- Flatten nested structure to array of tensors
function NestedTensor.flatten(obj, result)
   result = result or {}
   
   if torch.isTensor(obj) then
      table.insert(result, obj)
      return result
   end
   
   if type(obj) == 'table' then
      for _, v in pairs(obj) do
         NestedTensor.flatten(v, result)
      end
   end
   
   return result
end

-- Apply function to all tensors in nested structure
function NestedTensor.map(obj, func)
   if torch.isTensor(obj) then
      return func(obj)
   end
   
   if type(obj) == 'table' then
      local result = {}
      for k, v in pairs(obj) do
         result[k] = NestedTensor.map(v, func)
      end
      return result
   end
   
   return obj
end

-- Check if two nested structures have same shape
function NestedTensor.sameStructure(obj1, obj2)
   local t1 = type(obj1)
   local t2 = type(obj2)
   
   if t1 ~= t2 then
      return false
   end
   
   if torch.isTensor(obj1) and torch.isTensor(obj2) then
      return obj1:isSameSizeAs(obj2)
   end
   
   if t1 == 'table' then
      -- Check same keys
      for k in pairs(obj1) do
         if not obj2[k] then
            return false
         end
      end
      for k in pairs(obj2) do
         if not obj1[k] then
            return false
         end
      end
      
      -- Check recursively
      for k, v in pairs(obj1) do
         if not NestedTensor.sameStructure(v, obj2[k]) then
            return false
         end
      end
      return true
   end
   
   return true
end

-- Create nested tensor structure from specification
function NestedTensor.create(spec)
   if type(spec) == 'table' then
      if spec.type == 'tensor' then
         -- Create tensor with specified size
         return torch.Tensor(unpack(spec.size))
      elseif spec.type == 'nested' then
         -- Create nested structure
         local result = {}
         for k, v in pairs(spec.children) do
            result[k] = NestedTensor.create(v)
         end
         return result
      end
   end
   error("Invalid specification for nested tensor creation")
end

return NestedTensor
