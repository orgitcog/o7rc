-- PrimeFactorType: Type system based on prime factorization of tensor shapes
-- Provides a unique type signature for tensors based on dimensional decomposition

local PrimeFactorType = {}

-- Compute prime factorization of a number
function PrimeFactorType.factorize(n)
   if n < 1 then
      return {}
   end
   
   local factors = {}
   local d = 2
   while d * d <= n do
      while n % d == 0 do
         table.insert(factors, d)
         n = n / d
      end
      d = d + 1
   end
   if n > 1 then
      table.insert(factors, n)
   end
   return factors
end

-- Get prime factorization signature for tensor shape
function PrimeFactorType.getShapeSignature(shape)
   local signature = {}
   for i = 1, #shape do
      signature[i] = PrimeFactorType.factorize(shape[i])
   end
   return signature
end

-- Get type signature from tensor
function PrimeFactorType.getTensorSignature(tensor)
   local shape = tensor:size():totable()
   return PrimeFactorType.getShapeSignature(shape)
end

-- Compare two type signatures
function PrimeFactorType.compareSignatures(sig1, sig2)
   if #sig1 ~= #sig2 then
      return false
   end
   
   for i = 1, #sig1 do
      local factors1 = sig1[i]
      local factors2 = sig2[i]
      
      if #factors1 ~= #factors2 then
         return false
      end
      
      for j = 1, #factors1 do
         if factors1[j] ~= factors2[j] then
            return false
         end
      end
   end
   
   return true
end

-- Get unique type ID from signature (hash-like)
function PrimeFactorType.getTypeId(signature)
   local id = ""
   for i = 1, #signature do
      local factors = signature[i]
      id = id .. "["
      for j = 1, #factors do
         id = id .. tostring(factors[j])
         if j < #factors then
            id = id .. "."
         end
      end
      id = id .. "]"
   end
   return id
end

-- Check if tensor has a specific type signature
function PrimeFactorType.hasType(tensor, signature)
   local tensorSig = PrimeFactorType.getTensorSignature(tensor)
   return PrimeFactorType.compareSignatures(tensorSig, signature)
end

-- Product of all elements in array
local function product(arr)
   local p = 1
   for _, v in ipairs(arr) do
      p = p * v
   end
   return p
end

-- Compute metagraph-like type structure
-- Represents tensor as typed hypergraph node
function PrimeFactorType.getMetagraphType(tensor)
   local shape = tensor:size():totable()
   local signature = PrimeFactorType.getShapeSignature(shape)
   local typeId = PrimeFactorType.getTypeId(signature)
   
   -- Compute dimensional embeddings
   local dimEmbeddings = {}
   for i = 1, #shape do
      dimEmbeddings[i] = {
         size = shape[i],
         primeFactors = signature[i],
         factorProduct = product(signature[i])
      }
   end
   
   return {
      typeId = typeId,
      signature = signature,
      shape = shape,
      dimensionalEmbeddings = dimEmbeddings,
      totalElements = tensor:nElement(),
      totalFactors = PrimeFactorType.factorize(tensor:nElement())
   }
end

-- Check if two tensors are type-compatible (have same prime structure)
function PrimeFactorType.isCompatible(tensor1, tensor2)
   local sig1 = PrimeFactorType.getTensorSignature(tensor1)
   local sig2 = PrimeFactorType.getTensorSignature(tensor2)
   return PrimeFactorType.compareSignatures(sig1, sig2)
end

-- Get nested type structure for nested tensor/table
function PrimeFactorType.getNestedType(obj, depth)
   depth = depth or 0
   
   if torch.isTensor(obj) then
      local metagraphType = PrimeFactorType.getMetagraphType(obj)
      metagraphType.depth = depth
      return metagraphType
   end
   
   if type(obj) == 'table' then
      local nestedType = {
         depth = depth,
         isNested = true,
         children = {}
      }
      
      for k, v in pairs(obj) do
         nestedType.children[k] = PrimeFactorType.getNestedType(v, depth + 1)
      end
      
      return nestedType
   end
   
   return {
      depth = depth,
      isScalar = true
   }
end

-- Convert type structure to string representation
function PrimeFactorType.typeToString(typeStruct)
   if typeStruct.isNested then
      local s = string.format("Nested(depth=%d, children={", typeStruct.depth)
      local first = true
      for k, v in pairs(typeStruct.children) do
         if not first then
            s = s .. ", "
         end
         s = s .. tostring(k) .. ":" .. PrimeFactorType.typeToString(v)
         first = false
      end
      s = s .. "})"
      return s
   elseif typeStruct.isScalar then
      return "Scalar"
   else
      return string.format("Tensor(type=%s, shape=[%s])", 
         typeStruct.typeId or "unknown",
         table.concat(typeStruct.shape or {}, "x"))
   end
end

return PrimeFactorType
