-- NestedEmbedding: Extends tensor embeddings to nested neural nets (nnn)
-- Supports tuples of tuples of tuples as generalized rooted-tree-like embeddings
-- Similar to metagraph (typed hypergraph) structures where tensor shape
-- as prime factor product of dimensional embeddings specifies a unique type

local NestedEmbedding, parent = torch.class('nn.NestedEmbedding', 'nn.Module')

NestedEmbedding.__version = 1

function NestedEmbedding:__init(nIndex, nOutput, maxDepth)
   parent.__init(self)
   
   -- Basic embedding parameters
   self.nIndex = nIndex  -- vocabulary size
   self.nOutput = nOutput  -- embedding dimension
   self.maxDepth = maxDepth or 3  -- maximum nesting depth
   
   -- Weight tables for each level of nesting
   self.weights = {}
   self.gradWeights = {}
   
   -- Initialize weights for each nesting level
   for depth = 1, self.maxDepth do
      self.weights[depth] = torch.Tensor(nIndex, nOutput)
      self.gradWeights[depth] = torch.Tensor(nIndex, nOutput):zero()
   end
   
   -- Type signature based on prime factorization
   self.typeSignature = nil
   self.primeFactors = {}
   
   self:reset()
end

function NestedEmbedding:reset(stdv)
   stdv = stdv or 1 / math.sqrt(self.nOutput)
   for depth = 1, self.maxDepth do
      self.weights[depth]:normal(0, stdv)
   end
end

-- Compute prime factorization of a number
local function primeFactorize(n)
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

-- Get type signature from tensor shape
function NestedEmbedding:getTypeSignature(shape)
   local signature = {}
   for i = 1, #shape do
      signature[i] = primeFactorize(shape[i])
   end
   return signature
end

-- Check if input is a nested structure (table of tensors/tables)
local function isNested(input)
   return type(input) == 'table'
end

-- Get depth of nested structure
local function getDepth(input)
   if torch.isTensor(input) then
      return 0
   elseif type(input) ~= 'table' then
      return 0
   end
   
   local maxChildDepth = 0
   for _, v in pairs(input) do
      local childDepth = getDepth(v)
      if childDepth > maxChildDepth then
         maxChildDepth = childDepth
      end
   end
   return maxChildDepth + 1
end

-- Process nested input recursively
function NestedEmbedding:processNested(input, depth)
   depth = depth or 1
   
   if depth > self.maxDepth then
      error(string.format('Nesting depth %d exceeds maximum depth %d', depth, self.maxDepth))
   end
   
   -- Base case: tensor input
   if torch.isTensor(input) then
      -- Use standard embedding lookup at this depth
      local weight = self.weights[depth]
      local output
      if input:dim() == 1 then
         output = weight:index(1, input:long())
      elseif input:dim() == 2 then
         output = weight:index(1, input:long():view(-1))
         output = output:view(input:size(1), input:size(2), weight:size(2))
      else
         error("tensor input must be 1D or 2D")
      end
      return output
   end
   
   -- Recursive case: nested table structure
   if type(input) == 'table' then
      local outputs = {}
      for k, v in pairs(input) do
         outputs[k] = self:processNested(v, depth + 1)
      end
      return outputs
   end
   
   error("input must be a tensor or table")
end

function NestedEmbedding:updateOutput(input)
   -- Compute type signature from input structure
   if torch.isTensor(input) then
      local shape = input:size():totable()
      self.typeSignature = self:getTypeSignature(shape)
      self.primeFactors = primeFactorize(input:nElement())
   end
   
   -- Process nested structure
   self.output = self:processNested(input, 1)
   
   return self.output
end

-- Process nested gradient recursively
function NestedEmbedding:processNestedGrad(input, gradOutput, depth)
   depth = depth or 1
   
   -- Base case: tensor input
   if torch.isTensor(input) then
      if torch.type(self.gradInput) ~= torch.type(input) then
         self.gradInput = input.new()
      end
      if not self.gradInput:isSameSizeAs(input) then
         self.gradInput:resizeAs(input):zero()
      end
      return self.gradInput
   end
   
   -- Recursive case: nested table structure
   if type(input) == 'table' then
      local gradInputs = {}
      for k, v in pairs(input) do
         gradInputs[k] = self:processNestedGrad(v, gradOutput[k], depth + 1)
      end
      return gradInputs
   end
   
   error("input must be a tensor or table")
end

function NestedEmbedding:updateGradInput(input, gradOutput)
   self.gradInput = self:processNestedGrad(input, gradOutput, 1)
   return self.gradInput
end

-- Accumulate gradients for nested structure
function NestedEmbedding:accGradNested(input, gradOutput, depth, scale)
   depth = depth or 1
   scale = scale or 1
   
   -- Base case: tensor input
   if torch.isTensor(input) then
      local weight = self.weights[depth]
      local gradWeight = self.gradWeights[depth]
      
      local inputFlat = input:long()
      if input:dim() == 2 then
         inputFlat = inputFlat:view(-1)
      end
      
      local gradOutputFlat = gradOutput
      if gradOutput:dim() == 3 then
         gradOutputFlat = gradOutput:view(-1, gradOutput:size(3))
      end
      
      -- Accumulate gradients using index_add
      gradWeight:indexAdd(1, inputFlat, gradOutputFlat:mul(scale))
      return
   end
   
   -- Recursive case: nested table structure
   if type(input) == 'table' then
      for k, v in pairs(input) do
         self:accGradNested(v, gradOutput[k], depth + 1, scale)
      end
      return
   end
   
   error("input must be a tensor or table")
end

function NestedEmbedding:accGradParameters(input, gradOutput, scale)
   self:accGradNested(input, gradOutput, 1, scale)
end

function NestedEmbedding:type(type, tensorCache)
   parent.type(self, type, tensorCache)
   return self
end

function NestedEmbedding:clearState()
   return parent.clearState(self)
end

-- Get embedding weight for specific depth level
function NestedEmbedding:getWeightAtDepth(depth)
   if depth < 1 or depth > self.maxDepth then
      error(string.format('Depth %d out of range [1, %d]', depth, self.maxDepth))
   end
   return self.weights[depth]
end

-- Get type signature information
function NestedEmbedding:getType()
   return {
      signature = self.typeSignature,
      primeFactors = self.primeFactors
   }
end
