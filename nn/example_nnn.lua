-- Example usage of Nested Neural Nets (NNN)
-- This demonstrates the key features of the nnn extension

require 'nn'

print("=== Nested Neural Nets (NNN) Examples ===\n")

-- Example 1: Basic Prime Factorization Type System
print("1. Prime Factorization Type System")
print("-----------------------------------")
local PrimeFactorType = require('nn.PrimeFactorType')

-- Factorize some numbers
print("Prime factors of 12:", table.concat(PrimeFactorType.factorize(12), ", "))
print("Prime factors of 24:", table.concat(PrimeFactorType.factorize(24), ", "))
print("Prime factors of 17:", table.concat(PrimeFactorType.factorize(17), ", "))

-- Get tensor type signature
local tensor1 = torch.Tensor(4, 6, 8)
local signature = PrimeFactorType.getTensorSignature(tensor1)
print("\nTensor shape [4, 6, 8] type signature:")
for i, factors in ipairs(signature) do
   print(string.format("  Dimension %d: [%s]", i, table.concat(factors, ", ")))
end

-- Get metagraph type
local metagraphType = PrimeFactorType.getMetagraphType(tensor1)
print("\nMetagraph type information:")
print("  Type ID:", metagraphType.typeId)
print("  Total elements:", metagraphType.totalElements)
print("  Total prime factors:", table.concat(metagraphType.totalFactors, ", "))

-- Example 2: NestedTensor Utilities
print("\n2. NestedTensor Utilities")
print("-------------------------")
local NestedTensor = require('nn.NestedTensor')

local t1 = torch.Tensor(2, 3):fill(1)
local t2 = torch.Tensor(2, 3):fill(2)
local nested = {t1, {t1:clone(), t2:clone()}}

print("Nested structure depth:", NestedTensor.depth(nested))
print("Number of tensors:", NestedTensor.count(nested))

local flattened = NestedTensor.flatten(nested)
print("Flattened structure has", #flattened, "tensors")

-- Example 3: NestedEmbedding
print("\n3. NestedEmbedding")
print("------------------")
local nIndex = 100       -- vocabulary size
local nOutput = 32       -- embedding dimension
local maxDepth = 3       -- maximum nesting depth

local embedder = nn.NestedEmbedding(nIndex, nOutput, maxDepth)
print(string.format("Created NestedEmbedding (vocab=%d, dim=%d, depth=%d)", 
   nIndex, nOutput, maxDepth))

-- Simple tensor input
local input1 = torch.LongTensor({1, 2, 3, 4, 5})
local output1 = embedder:forward(input1)
print(string.format("Input shape: [%d] -> Output shape: [%d, %d]", 
   input1:size(1), output1:size(1), output1:size(2)))

-- Nested input
local input2 = {
   torch.LongTensor({10, 20}),
   torch.LongTensor({30, 40, 50})
}
local output2 = embedder:forward(input2)
print("Nested input (2 tensors) -> Nested output (2 tensors)")
print(string.format("  Branch 1: [%d] -> [%d, %d]", 
   input2[1]:size(1), output2[1]:size(1), output2[1]:size(2)))
print(string.format("  Branch 2: [%d] -> [%d, %d]", 
   input2[2]:size(1), output2[2]:size(1), output2[2]:size(2)))

-- Example 4: NestedNeuralNet
print("\n4. NestedNeuralNet")
print("------------------")

-- Create a simple nested neural net
local nnn = nn.NestedNeuralNet.createSimple(100, 32, 3)
print("Created NestedNeuralNet with embeddings at each depth")

-- Process nested input
local nestedInput = {
   torch.LongTensor({1, 2, 3}),
   {
      torch.LongTensor({4, 5}),
      torch.LongTensor({6, 7, 8})
   }
}

local nestedOutput = nnn:forward(nestedInput)
print("Processed nested input structure:")
print("  Input: { Tensor[3], { Tensor[2], Tensor[3] } }")

-- Analyze structure
local structure = nnn:getStructureString()
print("Structure analysis:", structure)

local typeInfo = nnn:getTypeInfo()
print("Maximum depth:", typeInfo.maxDepth)

-- Example 5: Custom Pipeline
print("\n5. Custom Processing Pipeline")
print("------------------------------")

local customNNN = nn.NestedNeuralNet({maxDepth = 3})

-- Add embedder at depth 1
local embedder1 = nn.LookupTable(100, 64)
customNNN:addEmbedder(1, embedder1)

-- Add processor at depth 1
local processor1 = nn.Tanh()
customNNN:addProcessor(1, processor1)

-- Add embedder at depth 2
local embedder2 = nn.LookupTable(100, 32)
customNNN:addEmbedder(2, embedder2)

print("Created custom NNN with:")
print("  - Depth 1: LookupTable(100, 64) + Tanh")
print("  - Depth 2: LookupTable(100, 32)")

-- Example 6: Type Compatibility
print("\n6. Type Compatibility Check")
print("---------------------------")

local tensor_a = torch.Tensor(4, 6)
local tensor_b = torch.Tensor(4, 6)
local tensor_c = torch.Tensor(4, 8)

local compatible_ab = PrimeFactorType.isCompatible(tensor_a, tensor_b)
local compatible_ac = PrimeFactorType.isCompatible(tensor_a, tensor_c)

print(string.format("Tensor[4,6] compatible with Tensor[4,6]: %s", 
   tostring(compatible_ab)))
print(string.format("Tensor[4,6] compatible with Tensor[4,8]: %s", 
   tostring(compatible_ac)))

-- Example 7: Nested Structure Analysis
print("\n7. Nested Structure Type Analysis")
print("----------------------------------")

local complexNested = {
   torch.Tensor(2, 3),
   {
      torch.Tensor(4, 5),
      {torch.Tensor(6, 7)}
   }
}

local nestedType = PrimeFactorType.getNestedType(complexNested)
print("Analyzed complex nested structure:")
print("  Is nested:", nestedType.isNested)
print("  Depth:", nestedType.depth)
print("  Number of children:", #nestedType.children)

-- Type string representation
local typeString = PrimeFactorType.typeToString(nestedType)
print("  Type string:", typeString)

print("\n=== Examples Complete ===")
print("\nFor more information, see nn/NNN_DOCUMENTATION.md")
