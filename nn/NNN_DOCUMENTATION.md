# Nested Neural Nets (NNN) - Documentation

## Overview

The Nested Neural Nets (NNN) extension provides a framework for working with rooted-tree-like tensor embeddings in neural networks. This system supports arbitrary nesting of tensors (tuples of tuples of tuples...) and implements a type system based on prime factorization of tensor shapes, similar to metagraph (typed hypergraph) structures.

## Core Concepts

### 1. Nested Structures

NNN supports arbitrary nesting of tensors and tables:
- **Depth 0**: Single tensor
- **Depth 1**: Table of tensors `{tensor1, tensor2, ...}`
- **Depth 2**: Table of tables of tensors `{{tensor1}, {tensor2, tensor3}, ...}`
- **Depth N**: Arbitrary nesting up to configured maximum depth

### 2. Type System via Prime Factorization

Each tensor's shape is decomposed into prime factors, creating a unique type signature:
- Shape `[4, 6]` → Prime factors `[[2, 2], [2, 3]]`
- Shape `[8, 3, 5]` → Prime factors `[[2, 2, 2], [3], [5]]`

This creates a metagraph-like type structure where dimensional embeddings specify unique types.

### 3. Tree-like Embeddings

Embeddings can be applied at each level of nesting, creating a hierarchical representation that preserves the tree structure of the input.

## Modules

### nn.PrimeFactorType

Utility module for prime factorization-based type system.

```lua
local PrimeFactorType = require('nn.PrimeFactorType')

-- Factorize a number
local factors = PrimeFactorType.factorize(12)  -- {2, 2, 3}

-- Get tensor type signature
local tensor = torch.Tensor(4, 6, 8)
local signature = PrimeFactorType.getTensorSignature(tensor)
-- signature = {{2,2}, {2,3}, {2,2,2}}

-- Get metagraph type structure
local metagraphType = PrimeFactorType.getMetagraphType(tensor)
-- Returns: {typeId, signature, shape, dimensionalEmbeddings, totalElements, totalFactors}

-- Check tensor compatibility
local t1 = torch.Tensor(4, 6)
local t2 = torch.Tensor(4, 6)
local compatible = PrimeFactorType.isCompatible(t1, t2)  -- true

-- Get nested type structure
local nested = {tensor, {tensor, tensor}}
local nestedType = PrimeFactorType.getNestedType(nested)
```

### nn.NestedTensor

Utility module for operations on nested tensor structures.

```lua
local NestedTensor = require('nn.NestedTensor')

local t1 = torch.Tensor(2, 3):fill(1)
local t2 = torch.Tensor(2, 3):fill(2)
local nested = {t1, {t1, t2}}

-- Clone nested structure
local cloned = NestedTensor.clone(nested)

-- Get depth
local depth = NestedTensor.depth(nested)  -- 2

-- Count tensors
local count = NestedTensor.count(nested)  -- 3

-- Flatten to array
local flat = NestedTensor.flatten(nested)  -- {t1, t1, t2}

-- Fill all tensors
NestedTensor.fill(nested, 5)

-- Apply function to all tensors
local scaled = NestedTensor.map(nested, function(t) return t:mul(2) end)

-- Check structure compatibility
local same = NestedTensor.sameStructure(nested, cloned)  -- true
```

### nn.NestedEmbedding

Embedding module that supports nested structures with different weights at each nesting level.

```lua
local nIndex = 1000      -- vocabulary size
local nOutput = 128      -- embedding dimension
local maxDepth = 3       -- maximum nesting depth

local embedder = nn.NestedEmbedding(nIndex, nOutput, maxDepth)

-- Forward with simple tensor
local input1 = torch.LongTensor({1, 2, 3, 4})
local output1 = embedder:forward(input1)  -- [4 x 128]

-- Forward with nested structure
local input2 = {
   torch.LongTensor({1, 2}),
   torch.LongTensor({3, 4, 5})
}
local output2 = embedder:forward(input2)  -- {[2 x 128], [3 x 128]}

-- Access weights at specific depth
local weightsDepth1 = embedder:getWeightAtDepth(1)
local weightsDepth2 = embedder:getWeightAtDepth(2)

-- Get type information
local typeInfo = embedder:getType()
-- typeInfo = {signature, primeFactors}
```

### nn.NestedNeuralNet

Complete framework for nested neural networks with embeddings and processing at each level.

```lua
-- Create with configuration
local nnn = nn.NestedNeuralNet({
   maxDepth = 3,
   useTypeSystem = true
})

-- Add embedder at specific depth
local embedder1 = nn.LookupTable(1000, 128)
nnn:addEmbedder(1, embedder1)

local embedder2 = nn.LookupTable(1000, 64)
nnn:addEmbedder(2, embedder2)

-- Add processor at specific depth
local processor = nn.Linear(128, 64)
nnn:addProcessor(1, processor)

-- Forward pass
local input = {
   torch.LongTensor({1, 2, 3}),
   {
      torch.LongTensor({4, 5}),
      torch.LongTensor({6, 7, 8})
   }
}
local output = nnn:forward(input)

-- Get structure analysis
local structure = nnn:getStructureString()
print(structure)  -- "Nested(depth=0, children={...})"

local typeInfo = nnn:getTypeInfo()
-- typeInfo = {treeStructure, typeSignatures, maxDepth}

-- Backward pass
local gradOutput = NestedTensor.clone(output)
NestedTensor.fill(gradOutput, 1)
local gradInput = nnn:backward(input, gradOutput)

-- Quick creation with embeddings at each level
local simpleNNN = nn.NestedNeuralNet.createSimple(
   1000,  -- vocabulary size
   128,   -- embedding dimension
   3      -- max depth
)
```

## Use Cases

### 1. Hierarchical Text Embeddings

```lua
-- Embed sentences as nested structure: document -> sentences -> words
local docEmbedder = nn.NestedNeuralNet.createSimple(10000, 128, 3)

local document = {
   -- Sentence 1: word indices
   torch.LongTensor({45, 123, 67, 890}),
   -- Sentence 2
   torch.LongTensor({12, 456, 789}),
   -- Nested paragraph
   {
      torch.LongTensor({111, 222, 333}),
      torch.LongTensor({444, 555})
   }
}

local embeddings = docEmbedder:forward(document)
```

### 2. Tree-Structured Data

```lua
-- Process tree structures like ASTs, parse trees, etc.
local treeProcessor = nn.NestedNeuralNet({maxDepth = 5})
treeProcessor:addEmbedder(1, nn.LookupTable(500, 64))
treeProcessor:addProcessor(1, nn.Tanh())

local tree = {
   root = torch.LongTensor({1}),  -- root node
   children = {
      torch.LongTensor({2, 3}),    -- left subtree
      {                             -- right subtree
         torch.LongTensor({4}),
         torch.LongTensor({5, 6})
      }
   }
}

local treeEmbedding = treeProcessor:forward(tree)
```

### 3. Metagraph/Hypergraph Representations

```lua
-- Use prime factorization type system for typed hypergraphs
local PrimeFactorType = require('nn.PrimeFactorType')

-- Create tensors with specific shapes
local node1 = torch.Tensor(4, 6)    -- Type: [[2,2], [2,3]]
local node2 = torch.Tensor(8, 3)    -- Type: [[2,2,2], [3]]

-- Check type compatibility
local compatible = PrimeFactorType.isCompatible(node1, node2)

-- Get metagraph type
local type1 = PrimeFactorType.getMetagraphType(node1)
print("Type ID:", type1.typeId)
print("Dimensional embeddings:", type1.dimensionalEmbeddings)
```

### 4. Multi-Scale Feature Processing

```lua
-- Process features at different scales/resolutions
local multiScale = nn.NestedNeuralNet({maxDepth = 3})

-- Different processing at each scale
multiScale:addProcessor(1, nn.SpatialConvolution(3, 64, 3, 3))
multiScale:addProcessor(2, nn.SpatialConvolution(3, 32, 3, 3))
multiScale:addProcessor(3, nn.SpatialConvolution(3, 16, 3, 3))

local features = {
   torch.Tensor(3, 64, 64),  -- High resolution
   {
      torch.Tensor(3, 32, 32),  -- Medium resolution
      torch.Tensor(3, 16, 16)   -- Low resolution
   }
}

local processed = multiScale:forward(features)
```

## Advanced Features

### Type Signature Analysis

```lua
local nnn = nn.NestedNeuralNet({maxDepth = 3, useTypeSystem = true})

local input = {
   torch.Tensor(4, 6),
   {torch.Tensor(8, 9), torch.Tensor(2, 3, 5)}
}

-- Analyze structure
nnn:analyzeStructure(input)

-- Get detailed type information
local typeInfo = nnn:getTypeInfo()
print("Tree structure:", nnn:getStructureString())
print("Type signatures by depth:", typeInfo.typeSignatures)
```

### Custom Processing Pipelines

```lua
-- Build custom pipeline with different operations at each level
local pipeline = nn.NestedNeuralNet({maxDepth = 4})

-- Depth 1: Embedding
pipeline:addEmbedder(1, nn.LookupTable(5000, 256))
pipeline:addProcessor(1, nn.Tanh())

-- Depth 2: Transformation
pipeline:addProcessor(2, nn.Linear(256, 128))
pipeline:addProcessor(2, nn.ReLU())

-- Depth 3: Attention-like mechanism
local attention = nn.Sequential()
   :add(nn.Linear(128, 64))
   :add(nn.SoftMax())
pipeline:addProcessor(3, attention)

local complexInput = {
   {
      {torch.LongTensor({1, 2, 3})}
   }
}

local result = pipeline:forward(complexInput)
```

## Implementation Notes

1. **Memory Efficiency**: Each depth level maintains separate weights, allowing independent optimization but increasing memory usage with depth.

2. **Type System**: The prime factorization-based type system provides unique identifiers for tensor shapes, enabling type checking and metagraph-like structures.

3. **Gradient Flow**: Gradients flow through the nested structure recursively, maintaining the tree structure during backpropagation.

4. **Flexibility**: Modules can be mixed and matched at different depths, allowing arbitrary processing pipelines.

## Limitations

- Maximum nesting depth must be specified in advance
- Memory scales with `maxDepth * nIndex * nOutput` for embeddings
- Very deep nesting may impact performance due to recursive operations

## Future Extensions

- Dynamic depth adaptation
- Attention mechanisms across nesting levels
- Graph neural network integration for non-tree structures
- Automatic type inference and shape verification
- Parallel processing of independent branches

## References

This implementation draws inspiration from:
- Nested structures in nngraph
- Metagraph and typed hypergraph concepts
- Prime factorization for type systems
- Tree-structured neural networks
