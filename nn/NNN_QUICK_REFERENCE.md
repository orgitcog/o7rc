# Nested Neural Nets (NNN) - Quick Reference

## Installation
NNN is integrated into the nn package. Simply require nn:
```lua
require 'nn'
```

## Quick Start

### 1. Simple Nested Embedding
```lua
-- Create nested embedding with vocabulary size 1000, dimension 128, max depth 3
local embedder = nn.NestedEmbedding(1000, 128, 3)

-- Simple tensor input
local input = torch.LongTensor({1, 2, 3})
local output = embedder:forward(input)  -- [3 x 128]

-- Nested input
local nestedInput = {
   torch.LongTensor({1, 2}),
   torch.LongTensor({3, 4, 5})
}
local nestedOutput = embedder:forward(nestedInput)  -- {[2x128], [3x128]}
```

### 2. Complete Nested Neural Net
```lua
-- Quick creation with embeddings at each level
local nnn = nn.NestedNeuralNet.createSimple(
   1000,  -- vocab size
   128,   -- embedding dim
   3      -- max depth
)

-- Process nested structure
local input = {
   torch.LongTensor({1, 2, 3}),
   {torch.LongTensor({4, 5})}
}
local output = nnn:forward(input)
```

### 3. Custom Pipeline
```lua
-- Create with custom configuration
local nnn = nn.NestedNeuralNet({maxDepth = 3})

-- Add embedder at depth 1
nnn:addEmbedder(1, nn.LookupTable(1000, 64))

-- Add processor at depth 1
nnn:addProcessor(1, nn.Tanh())

-- Add embedder at depth 2
nnn:addEmbedder(2, nn.LookupTable(1000, 32))

-- Forward pass
local output = nnn:forward(input)
```

## Prime Factorization Type System

### Get Type Signature
```lua
local PrimeFactorType = require('nn.PrimeFactorType')

-- Factorize number
local factors = PrimeFactorType.factorize(12)  -- {2, 2, 3}

-- Get tensor type signature
local tensor = torch.Tensor(4, 6, 8)
local signature = PrimeFactorType.getTensorSignature(tensor)
-- signature = {{2,2}, {2,3}, {2,2,2}}

-- Get metagraph type
local metagraphType = PrimeFactorType.getMetagraphType(tensor)
print(metagraphType.typeId)           -- "[2.2][2.3][2.2.2]"
print(metagraphType.totalElements)    -- 192
```

### Check Type Compatibility
```lua
local t1 = torch.Tensor(4, 6)
local t2 = torch.Tensor(4, 6)
local t3 = torch.Tensor(4, 8)

PrimeFactorType.isCompatible(t1, t2)  -- true
PrimeFactorType.isCompatible(t1, t3)  -- false
```

## Nested Tensor Utilities

### Basic Operations
```lua
local NestedTensor = require('nn.NestedTensor')

local nested = {
   torch.Tensor(2, 3),
   {torch.Tensor(4, 5), torch.Tensor(6, 7)}
}

-- Get depth
local depth = NestedTensor.depth(nested)  -- 2

-- Count tensors
local count = NestedTensor.count(nested)  -- 3

-- Clone
local cloned = NestedTensor.clone(nested)

-- Flatten
local flat = NestedTensor.flatten(nested)  -- {tensor1, tensor2, tensor3}

-- Fill all tensors
NestedTensor.fill(nested, 0)

-- Map function to all tensors
local scaled = NestedTensor.map(nested, function(t) return t:mul(2) end)
```

## Common Patterns

### Hierarchical Text Processing
```lua
-- Document -> Paragraphs -> Sentences -> Words
local docProcessor = nn.NestedNeuralNet.createSimple(10000, 256, 3)

local document = {
   -- Paragraph 1
   {
      torch.LongTensor({1, 2, 3}),    -- Sentence 1
      torch.LongTensor({4, 5})        -- Sentence 2
   },
   -- Paragraph 2
   {
      torch.LongTensor({6, 7, 8, 9})  -- Sentence 1
   }
}

local embeddings = docProcessor:forward(document)
```

### Tree-Structured Data
```lua
-- Process AST or parse tree
local treeProcessor = nn.NestedNeuralNet({maxDepth = 4})
treeProcessor:addEmbedder(1, nn.LookupTable(500, 64))

local tree = {
   torch.LongTensor({1}),  -- root
   {                        -- children
      torch.LongTensor({2, 3}),
      {torch.LongTensor({4})}
   }
}

local result = treeProcessor:forward(tree)
```

### Type Analysis
```lua
local nnn = nn.NestedNeuralNet({maxDepth = 3, useTypeSystem = true})

-- Analyze structure
nnn:analyzeStructure(input)

-- Get type info
local typeInfo = nnn:getTypeInfo()
print("Structure:", nnn:getStructureString())
print("Max depth:", typeInfo.maxDepth)
```

## Module Methods

### NestedEmbedding
- `NestedEmbedding(nIndex, nOutput, maxDepth)` - Constructor
- `:forward(input)` - Forward pass (tensor or nested table)
- `:backward(input, gradOutput)` - Backward pass
- `:getWeightAtDepth(depth)` - Get weights at specific depth
- `:getType()` - Get type signature information

### NestedNeuralNet
- `NestedNeuralNet(config)` - Constructor with config table
- `NestedNeuralNet.createSimple(nIndex, nOutput, maxDepth)` - Quick creation
- `:addEmbedder(depth, embedder)` - Add embedding at depth
- `:addProcessor(depth, processor)` - Add processor at depth
- `:forward(input)` - Forward pass
- `:backward(input, gradOutput)` - Backward pass
- `:analyzeStructure(input)` - Analyze and store type structure
- `:getTypeInfo()` - Get type information
- `:getStructureString()` - Get string representation of structure

### PrimeFactorType (Module Functions)
- `factorize(n)` - Prime factorization of number
- `getTensorSignature(tensor)` - Type signature from tensor
- `getMetagraphType(tensor)` - Full metagraph type structure
- `isCompatible(t1, t2)` - Check type compatibility
- `getNestedType(obj, depth)` - Analyze nested structure
- `typeToString(typeStruct)` - Convert type to string

### NestedTensor (Module Functions)
- `clone(obj)` - Deep clone
- `fill(obj, value)` - Fill all tensors
- `depth(obj)` - Get nesting depth
- `count(obj)` - Count tensors
- `flatten(obj)` - Flatten to array
- `map(obj, func)` - Apply function to all tensors
- `sameStructure(obj1, obj2)` - Check structural equality

## Configuration

### NestedNeuralNet Config Table
```lua
local config = {
   maxDepth = 3,          -- Maximum nesting depth (default: 3)
   useTypeSystem = true   -- Enable type system (default: true)
}
local nnn = nn.NestedNeuralNet(config)
```

## Training

### Basic Training Loop
```lua
local nnn = nn.NestedNeuralNet.createSimple(1000, 128, 3)
local criterion = nn.MSECriterion()

-- Forward
local output = nnn:forward(input)
local loss = criterion:forward(output, target)

-- Backward
local gradOutput = criterion:backward(output, target)
nnn:zeroGradParameters()
nnn:backward(input, gradOutput)

-- Update
nnn:updateParameters(learningRate)
```

## Examples

See the following files for more examples:
- `example_nnn.lua` - Comprehensive examples
- `test_nnn.lua` - Test suite with usage patterns
- `NNN_DOCUMENTATION.md` - Complete documentation

## Performance Tips

1. **Depth**: Keep maxDepth as low as needed (reduces memory)
2. **Batch Processing**: Use batched tensors at leaf level
3. **Type System**: Disable if not needed (`useTypeSystem = false`)
4. **Reuse Structures**: Clone and reuse nested structures when possible

## Common Issues

**Issue**: "Nesting depth exceeds maximum depth"
**Solution**: Increase maxDepth in constructor

**Issue**: Gradient not flowing correctly
**Solution**: Ensure gradOutput structure matches output structure

**Issue**: Memory usage high
**Solution**: Reduce maxDepth or embedding dimension

## Links

- [Full Documentation](NNN_DOCUMENTATION.md)
- [Implementation Summary](NNN_IMPLEMENTATION_SUMMARY.md)
- [Test Suite](test_nnn.lua)
- [Examples](example_nnn.lua)
