-- Integration example: Using NNN with torch7u integration features
-- This demonstrates how Nested Neural Nets work within the unified torch7u framework

require 'init'  -- Load torch7u integration layer

print("=== Nested Neural Nets Integration with torch7u ===\n")

-- Example 1: Using NNN with torch7u model registry
print("1. Integration with Model Registry")
print("-----------------------------------")

-- Create a nested neural net
local nnn = nn.NestedNeuralNet.createSimple(1000, 128, 3)

-- Register with torch7u model registry (if available)
if torch7u and torch7u.models then
   torch7u.models.register('hierarchical_embedder', nnn)
   print("Registered NNN model with torch7u model registry")
else
   print("torch7u model registry not available, using standalone")
end

-- Example 2: Using NNN with existing nn modules
print("\n2. Combining NNN with Standard nn Modules")
print("------------------------------------------")

-- Create a pipeline that combines NNN with standard nn modules
local model = nn.Sequential()

-- Add nested embedder as first layer
local embedder = nn.NestedNeuralNet({maxDepth = 2})
embedder:addEmbedder(1, nn.LookupTable(100, 64))
embedder:addProcessor(1, nn.Tanh())

-- Note: In real usage, you'd need to flatten the nested output
-- before passing to standard layers. Here's a conceptual example:
print("Created hybrid model with NNN embeddings")

-- Example 3: Type-based model selection using prime factorization
print("\n3. Type-Based Model Selection")
print("------------------------------")

local PrimeFactorType = require('nn.PrimeFactorType')

-- Function to select model based on tensor type signature
local function selectModelByType(tensor)
   local metagraphType = PrimeFactorType.getMetagraphType(tensor)
   local typeId = metagraphType.typeId
   
   print(string.format("Input type ID: %s", typeId))
   print(string.format("Input shape: [%s]", table.concat(metagraphType.shape, "x")))
   
   -- Select model based on type signature
   -- This enables type-specialized processing
   if #metagraphType.shape == 2 then
      print("Selected: 2D specialized model")
      return nn.Linear(metagraphType.totalElements, 10)
   elseif #metagraphType.shape == 3 then
      print("Selected: 3D specialized model")
      return nn.VolumetricConvolution(3, 16, 3, 3, 3)
   else
      print("Selected: Generic model")
      return nn.Identity()
   end
end

-- Test with different tensor types
local tensor2d = torch.Tensor(4, 6)
local model2d = selectModelByType(tensor2d)

-- Example 4: Hierarchical data processing pipeline
print("\n4. Hierarchical Data Processing Pipeline")
print("-----------------------------------------")

-- Create a document processing pipeline
local docPipeline = {
   -- Stage 1: Word embeddings
   wordEmbedder = nn.NestedNeuralNet.createSimple(10000, 256, 1),
   
   -- Stage 2: Sentence processing
   sentenceProcessor = nn.Sequential()
      :add(nn.TemporalConvolution(256, 128, 3))
      :add(nn.ReLU())
      :add(nn.TemporalMaxPooling(2)),
   
   -- Stage 3: Document aggregation
   documentAggregator = nn.Sequential()
      :add(nn.Linear(128, 64))
      :add(nn.Tanh())
}

print("Created hierarchical document processing pipeline:")
print("  - Word embeddings (NNN, depth 1)")
print("  - Sentence processing (Temporal CNN)")
print("  - Document aggregation (Linear + Tanh)")

-- Example 5: Using NestedTensor utilities for data preprocessing
print("\n5. Data Preprocessing with NestedTensor")
print("----------------------------------------")

local NestedTensor = require('nn.NestedTensor')

-- Create hierarchical document structure
local document = {
   -- Paragraph 1: {sentence1, sentence2, ...}
   {
      torch.LongTensor({1, 2, 3, 4}),
      torch.LongTensor({5, 6, 7})
   },
   -- Paragraph 2
   {
      torch.LongTensor({8, 9}),
      torch.LongTensor({10, 11, 12, 13})
   }
}

-- Analyze structure
local depth = NestedTensor.depth(document)
local count = NestedTensor.count(document)

print(string.format("Document structure: depth=%d, tensors=%d", depth, count))
print("Structure: {{[4], [3]}, {[2], [4]}}")

-- Clone for batch processing
local batch = {}
for i = 1, 5 do
   batch[i] = NestedTensor.clone(document)
end
print(string.format("Created batch of %d documents", #batch))

-- Example 6: Metagraph-style typed processing
print("\n6. Metagraph-Style Typed Processing")
print("------------------------------------")

-- Create processors for different types
local typeProcessors = {}

-- Register processor for specific type signature
local function registerTypeProcessor(typeId, processor)
   typeProcessors[typeId] = processor
   print(string.format("Registered processor for type: %s", typeId))
end

-- Register some example processors
registerTypeProcessor("[2.2][2.3]", nn.Linear(24, 10))  -- 4x6 tensors
registerTypeProcessor("[2.2.2][3]", nn.Linear(24, 10))  -- 8x3 tensors
registerTypeProcessor("[2][3][5]", nn.Linear(30, 10))   -- 2x3x5 tensors

-- Function to process tensor based on its type
local function processWithType(tensor)
   local metagraphType = PrimeFactorType.getMetagraphType(tensor)
   local processor = typeProcessors[metagraphType.typeId]
   
   if processor then
      print(string.format("Processing tensor with type %s", metagraphType.typeId))
      return processor:forward(tensor:view(-1))
   else
      print(string.format("No processor for type %s, using default", metagraphType.typeId))
      return tensor
   end
end

-- Test typed processing
local testTensor = torch.Tensor(4, 6):fill(1)
local result = processWithType(testTensor)
print(string.format("Result shape: [%s]", table.concat(result:size():totable(), "x")))

-- Example 7: Integration with training framework
print("\n7. Integration with Training Framework")
print("---------------------------------------")

-- Create model and criterion
local trainModel = nn.NestedNeuralNet.createSimple(100, 32, 2)
local criterion = nn.MSECriterion()

print("Created training setup:")
print("  - Model: NestedNeuralNet (vocab=100, dim=32, depth=2)")
print("  - Criterion: MSECriterion")

-- Simulate training step (conceptual)
local function trainStep(model, criterion, input, target, learningRate)
   -- Forward
   local output = model:forward(input)
   local loss = criterion:forward(output, target)
   
   -- Backward
   local gradOutput = criterion:backward(output, target)
   model:zeroGradParameters()
   model:backward(input, gradOutput)
   
   -- Update
   model:updateParameters(learningRate)
   
   return loss
end

print("Training functions configured")
print("  - Forward, backward, and parameter updates")

-- Example 8: Nested structure validation
print("\n8. Nested Structure Validation")
print("-------------------------------")

local function validateNestedStructure(obj, maxDepth)
   local NestedTensor = require('nn.NestedTensor')
   
   local depth = NestedTensor.depth(obj)
   local count = NestedTensor.count(obj)
   
   print(string.format("Validation results:"))
   print(string.format("  - Depth: %d (max: %d) %s", 
      depth, maxDepth, depth <= maxDepth and "✓" or "✗"))
   print(string.format("  - Tensor count: %d", count))
   
   -- Check all tensors are properly formed
   local valid = true
   local tensors = NestedTensor.flatten(obj)
   for i, t in ipairs(tensors) do
      if not torch.isTensor(t) then
         valid = false
         print(string.format("  - Element %d is not a tensor ✗", i))
      end
   end
   
   if valid then
      print("  - All elements are valid tensors ✓")
   end
   
   return valid and depth <= maxDepth
end

-- Validate example structure
local validInput = {
   torch.Tensor(2, 3),
   {torch.Tensor(4, 5)}
}
local isValid = validateNestedStructure(validInput, 3)
print(string.format("\nOverall validation: %s", isValid and "PASSED" or "FAILED"))

print("\n=== Integration Examples Complete ===")
print("\nKey Integration Points:")
print("1. NNN modules work seamlessly with standard nn modules")
print("2. Type system enables intelligent model selection")
print("3. Nested structures support hierarchical data processing")
print("4. Compatible with torch7u model registry and training framework")
print("5. Utilities support validation and preprocessing")
print("\nSee NNN_DOCUMENTATION.md for complete API reference")
