-- NestedNeuralNet (NNN): Complete nested neural network module
-- Combines NestedEmbedding, NestedTensor utilities, and PrimeFactorType system
-- Provides a complete framework for working with rooted-tree-like tensor embeddings

local NestedNeuralNet, parent = torch.class('nn.NestedNeuralNet', 'nn.Module')

NestedNeuralNet.__version = 1

function NestedNeuralNet:__init(config)
   parent.__init(self)
   
   config = config or {}
   
   -- Configuration
   self.maxDepth = config.maxDepth or 3
   self.useTypeSystem = config.useTypeSystem ~= false  -- default true
   self.embedders = {}  -- Embedding modules at each depth
   self.processors = {}  -- Processing modules at each depth
   
   -- Type system
   self.primeFactorType = require('nn.PrimeFactorType')
   self.nestedTensor = require('nn.NestedTensor')
   
   -- Tree structure tracking
   self.treeStructure = nil
   self.typeSignatures = {}
end

-- Add an embedding layer at specific depth
function NestedNeuralNet:addEmbedder(depth, embedder)
   if depth < 1 or depth > self.maxDepth then
      error(string.format('Depth %d out of range [1, %d]', depth, self.maxDepth))
   end
   self.embedders[depth] = embedder
   return self
end

-- Add a processing module at specific depth
function NestedNeuralNet:addProcessor(depth, processor)
   if depth < 1 or depth > self.maxDepth then
      error(string.format('Depth %d out of range [1, %d]', depth, self.maxDepth))
   end
   self.processors[depth] = processor
   return self
end

-- Analyze input structure and create type signatures
function NestedNeuralNet:analyzeStructure(input)
   -- Get nested type structure
   self.treeStructure = self.primeFactorType.getNestedType(input, 0)
   
   -- Extract type signatures at each level
   local function extractSignatures(node, depth, signatures)
      if not node.isNested and not node.isScalar then
         -- It's a tensor node
         signatures[depth] = signatures[depth] or {}
         table.insert(signatures[depth], node.typeId)
      elseif node.isNested then
         for k, child in pairs(node.children) do
            extractSignatures(child, depth + 1, signatures)
         end
      end
   end
   
   self.typeSignatures = {}
   extractSignatures(self.treeStructure, 0, self.typeSignatures)
   
   return self.treeStructure
end

-- Process nested structure recursively
function NestedNeuralNet:processNested(input, depth)
   depth = depth or 0
   
   if depth >= self.maxDepth then
      error(string.format('Nesting depth %d exceeds maximum depth %d', depth, self.maxDepth))
   end
   
   -- Base case: tensor input
   if torch.isTensor(input) then
      local output = input
      
      -- Apply embedder if available at this depth
      if self.embedders[depth + 1] then
         output = self.embedders[depth + 1]:forward(output)
      end
      
      -- Apply processor if available at this depth
      if self.processors[depth + 1] then
         output = self.processors[depth + 1]:forward(output)
      end
      
      return output
   end
   
   -- Recursive case: nested table structure
   if type(input) == 'table' then
      local outputs = {}
      for k, v in pairs(input) do
         outputs[k] = self:processNested(v, depth + 1)
      end
      
      -- Apply processor at this nesting level if available
      if self.processors[depth + 1] then
         -- Process the collection of outputs
         outputs = self.processors[depth + 1]:forward(outputs)
      end
      
      return outputs
   end
   
   error("input must be a tensor or table")
end

function NestedNeuralNet:updateOutput(input)
   -- Analyze structure if using type system
   if self.useTypeSystem then
      self:analyzeStructure(input)
   end
   
   -- Process nested structure
   self.output = self:processNested(input, 0)
   
   return self.output
end

-- Backward pass through nested structure
function NestedNeuralNet:processNestedBackward(input, gradOutput, depth)
   depth = depth or 0
   
   -- Base case: tensor input
   if torch.isTensor(input) then
      local gradInput = gradOutput
      
      -- Backward through processor
      if self.processors[depth + 1] then
         gradInput = self.processors[depth + 1]:backward(input, gradInput)
      end
      
      -- Backward through embedder
      if self.embedders[depth + 1] then
         local embeddedInput = self.embedders[depth + 1].output
         gradInput = self.embedders[depth + 1]:backward(input, gradInput)
      end
      
      return gradInput
   end
   
   -- Recursive case: nested table structure
   if type(input) == 'table' then
      local currentGradOutput = gradOutput
      
      -- Backward through processor at this level
      if self.processors[depth + 1] then
         currentGradOutput = self.processors[depth + 1]:backward(input, currentGradOutput)
      end
      
      local gradInputs = {}
      for k, v in pairs(input) do
         gradInputs[k] = self:processNestedBackward(v, currentGradOutput[k], depth + 1)
      end
      return gradInputs
   end
   
   error("input must be a tensor or table")
end

function NestedNeuralNet:updateGradInput(input, gradOutput)
   self.gradInput = self:processNestedBackward(input, gradOutput, 0)
   return self.gradInput
end

function NestedNeuralNet:accGradParameters(input, gradOutput, scale)
   -- Accumulate gradients for all embedders and processors
   local function accumulateNested(inp, gradOut, depth)
      depth = depth or 0
      
      if torch.isTensor(inp) then
         if self.processors[depth + 1] then
            self.processors[depth + 1]:accGradParameters(inp, gradOut, scale)
         end
         if self.embedders[depth + 1] then
            local embeddedInput = inp
            if self.processors[depth + 1] then
               -- Need to pass through processor to get correct input
               embeddedInput = self.embedders[depth + 1].output
            end
            self.embedders[depth + 1]:accGradParameters(inp, gradOut, scale)
         end
      elseif type(inp) == 'table' then
         if self.processors[depth + 1] then
            self.processors[depth + 1]:accGradParameters(inp, gradOut, scale)
         end
         for k, v in pairs(inp) do
            accumulateNested(v, gradOut[k], depth + 1)
         end
      end
   end
   
   accumulateNested(input, gradOutput, 0)
end

-- Get type information for current input
function NestedNeuralNet:getTypeInfo()
   return {
      treeStructure = self.treeStructure,
      typeSignatures = self.typeSignatures,
      maxDepth = self.maxDepth
   }
end

-- Get string representation of tree structure
function NestedNeuralNet:getStructureString()
   if not self.treeStructure then
      return "No structure analyzed yet"
   end
   return self.primeFactorType.typeToString(self.treeStructure)
end

-- Reset all parameters
function NestedNeuralNet:reset(stdv)
   for _, embedder in pairs(self.embedders) do
      if embedder.reset then
         embedder:reset(stdv)
      end
   end
   for _, processor in pairs(self.processors) do
      if processor.reset then
         processor:reset(stdv)
      end
   end
end

function NestedNeuralNet:type(type, tensorCache)
   parent.type(self, type, tensorCache)
   for _, embedder in pairs(self.embedders) do
      embedder:type(type, tensorCache)
   end
   for _, processor in pairs(self.processors) do
      processor:type(type, tensorCache)
   end
   return self
end

function NestedNeuralNet:clearState()
   for _, embedder in pairs(self.embedders) do
      embedder:clearState()
   end
   for _, processor in pairs(self.processors) do
      processor:clearState()
   end
   return parent.clearState(self)
end

-- Helper: Create a simple nested neural net with embeddings at each level
function NestedNeuralNet.createSimple(nIndex, nOutput, maxDepth)
   local nnn = nn.NestedNeuralNet({maxDepth = maxDepth})
   
   -- Add embeddings at each depth
   for depth = 1, maxDepth do
      local embedder = nn.LookupTable(nIndex, nOutput)
      nnn:addEmbedder(depth, embedder)
   end
   
   return nnn
end
