# NNN Quick Reference

## What is NNN?

**NNN (Nested Neural Nets)** is a functional operator system that extends any `nn.*` module to work with nested tensors (nestors). It provides metagraph embeddings based on prime factorization type systems.

**Core Idea**: `nn.*` works with tensors, `nnn.*` works with nested tensors (trees of tensors).

## Basic Usage

### Import
```lua
require 'nn'
local nnn = require 'nnn'
```

### Transform Any Module
```lua
-- Take any nn module
local linear = nn.Linear(10, 5)

-- Transform to work with nested tensors
local nestedLinear = nnn.transform(linear)

-- Use with nested input
local input = {torch.randn(10), torch.randn(10)}
local output = nestedLinear:forward(input)  -- {Tensor[5], Tensor[5]}
```

## Pre-built Modules

### Containers
- `nnn.Sequential()` - Sequential container for nested tensors

### Linear Layers
- `nnn.Linear(in, out, bias)` - Linear transformation

### Activations
- `nnn.Tanh()` - Hyperbolic tangent
- `nnn.ReLU(inplace)` - Rectified linear unit
- `nnn.Sigmoid()` - Sigmoid activation
- `nnn.SoftMax()` - Softmax activation

### Example: Build a Model
```lua
local model = nnn.Sequential()
    :add(nnn.Linear(20, 15))
    :add(nnn.ReLU())
    :add(nnn.Linear(15, 10))
    :add(nnn.Tanh())
```

## Criteria (Modal Classifiers)

### Transform Any Criterion
```lua
-- Wrap any criterion
local criterion = nnn.Criterion(nn.MSECriterion())

-- Or use pre-built
local criterion = nnn.MSECriterion()
```

### Pre-built Criteria
- `nnn.MSECriterion()` - Mean squared error
- `nnn.ClassNLLCriterion(weights)` - Negative log-likelihood
- `nnn.BCECriterion(weights)` - Binary cross-entropy
- `nnn.CrossEntropyCriterion(weights)` - Cross-entropy

### Example: Training with Nested Data
```lua
local model = nnn.Sequential():add(nnn.Linear(10, 5))
local criterion = nnn.MSECriterion()

-- Nested input and target
local input = {torch.randn(10), torch.randn(10)}
local target = {torch.randn(5), torch.randn(5)}

-- Forward
local output = model:forward(input)
local loss = criterion:forward(output, target)

-- Backward
local gradOutput = criterion:backward(output, target)
model:zeroGradParameters()
model:backward(input, gradOutput)
model:updateParameters(0.01)
```

## Utility Functions

### Structure Operations
```lua
-- Check if nested
nnn.isNested(input)  -- true/false

-- Get depth
nnn.depth(input)  -- integer

-- Flatten to array
nnn.flatten(input)  -- {tensor1, tensor2, ...}

-- Clone structure
nnn.clone(input)  -- deep copy

-- Map function to all tensors
nnn.map(input, function(t) return t * 2 end)
```

## Advanced Features

### Factory Function
```lua
-- Create nnn version of any nn module by name
local sigmoid = nnn.fromNN('Sigmoid')
local conv = nnn.fromNN('SpatialConvolution', 3, 64, 3, 3)
```

### Wrap Existing Models
```lua
-- Transform a pre-trained model
local pretrainedModel = torch.load('model.t7')
local nestedModel = nnn.transform(pretrainedModel)

-- Now works with nested inputs
local output = nestedModel:forward({tensor1, tensor2})
```

## Type System Integration

### Access Prime Factorization Types
```lua
local PrimeFactorType = nnn.PrimeFactorType

-- Get metagraph type
local tensor = torch.Tensor(4, 6)
local typeInfo = PrimeFactorType.getMetagraphType(tensor)

print(typeInfo.typeId)        -- "[2.2][2.3]"
print(typeInfo.signature)     -- {{2,2}, {2,3}}

-- Check compatibility
local compatible = PrimeFactorType.isCompatible(tensor1, tensor2)
```

## Common Patterns

### Pattern 1: Hierarchical Text Processing
```lua
-- Document -> Paragraphs -> Sentences -> Words
local model = nnn.Sequential()
    :add(nn.NestedEmbedding(10000, 128, 3))
    :add(nnn.Linear(128, 64))
    :add(nnn.Tanh())

local document = {
    {torch.LongTensor({1,2,3}), torch.LongTensor({4,5})},  -- Para 1
    {torch.LongTensor({6,7,8})}                             -- Para 2
}

local output = model:forward(document)
```

### Pattern 2: Multi-branch Classification
```lua
local classifier = nnn.Sequential()
    :add(nnn.Linear(100, 50))
    :add(nnn.ReLU())
    :add(nnn.Linear(50, 10))
    :add(nnn.SoftMax())

-- Classify multiple branches independently
local inputs = {
    torch.randn(100),  -- Branch 1
    torch.randn(100),  -- Branch 2
    torch.randn(100)   -- Branch 3
}

local predictions = classifier:forward(inputs)
-- predictions = {prob1, prob2, prob3}
```

### Pattern 3: Tree-Structured Data
```lua
-- Process tree structures (AST, parse trees, etc.)
local treeProcessor = nnn.Sequential()
    :add(nnn.Linear(50, 25))
    :add(nnn.ReLU())

local tree = {
    root = torch.randn(50),
    children = {
        torch.randn(50),
        {torch.randn(50), torch.randn(50)}
    }
}

local processed = treeProcessor:forward(tree)
```

## Key Concepts

### Nested Tensors (Nestors)
A nested tensor is a tree structure where:
- **Leaves**: Regular tensors
- **Internal nodes**: Tables/arrays
- **Depth**: Maximum nesting level

Example:
```lua
-- Depth 0: Single tensor
torch.randn(10)

-- Depth 1: Array of tensors
{torch.randn(10), torch.randn(10)}

-- Depth 2: Nested arrays
{{torch.randn(10)}, torch.randn(10)}
```

### Structure Preservation
**Key Property**: Output structure always matches input structure

```lua
local input = {
    {tensor1, tensor2},
    tensor3
}
local output = model:forward(input)
-- output = {{result1, result2}, result3}
```

### Metagraph Embeddings
Each tensor has a type signature based on prime factorization:
- Shape `[4, 6]` → Type `"[2.2][2.3]"`
- Shape `[8, 3]` → Type `"[2.2.2][3]"`

This creates a typed hypergraph structure where shape determines type.

## Comparison: nn vs nnn

| Aspect | nn.* | nnn.* |
|--------|------|-------|
| Input | Single tensor | Nested tensors (tree) |
| Output | Single tensor | Nested tensors (same structure) |
| Use case | Standard data | Hierarchical/tree data |
| Embedding | Tensor | Nestor (metagraph) |

## Performance Tips

1. **Reuse transformations**: Cache transformed modules
2. **Limit depth**: Set appropriate `maxDepth` to avoid deep recursion
3. **Batch within branches**: Use batched tensors at leaf level
4. **Profile first**: Test with single tensors before nesting

## Common Mistakes

### ❌ Wrong: Mismatched structure in training
```lua
local input = {tensor1, tensor2}
local target = {tensor1}  -- Different structure!
-- This will fail
```

### ✓ Right: Matching structure
```lua
local input = {tensor1, tensor2}
local target = {target1, target2}  -- Same structure
-- This works correctly
```

### ❌ Wrong: Forgetting to transform criterion
```lua
local model = nnn.Linear(10, 5)
local criterion = nn.MSECriterion()  -- Not transformed!
-- Won't work with nested outputs
```

### ✓ Right: Transform both model and criterion
```lua
local model = nnn.Linear(10, 5)
local criterion = nnn.MSECriterion()  -- Transformed
-- Works with nested outputs
```

## API Quick Lookup

### Core Functions
- `nnn.transform(module, config)` - Transform any module
- `nnn.wrap(module)` - Alias for transform
- `nnn.fromNN(name, ...)` - Create by name

### Modules
- `nnn.Sequential()`
- `nnn.Linear(in, out, bias)`
- `nnn.Tanh()`, `nnn.ReLU()`, `nnn.Sigmoid()`, `nnn.SoftMax()`

### Criteria
- `nnn.Criterion(criterion)` - Transform any criterion
- `nnn.MSECriterion()`, `nnn.ClassNLLCriterion()`, `nnn.BCECriterion()`

### Utilities
- `nnn.isNested(input)` - Check if nested
- `nnn.depth(input)` - Get depth
- `nnn.flatten(input)` - Flatten structure
- `nnn.clone(input)` - Clone structure
- `nnn.map(input, func)` - Map function

### Type System
- `nnn.PrimeFactorType` - Access type utilities
- `nnn.NestedTensor` - Access nested tensor utilities

## Future Extensions

### Operad Gadgets (Planned)
```lua
-- Future: Operad structures indexed by prime factorizations
nnn.operad.compose(op1, op2, signature)
nnn.operad.symmetry(typeId)
```

The `nnn.operad` namespace is reserved for composable operations with orbifold symmetries based on prime factorizations.

## Resources

- **Full Documentation**: [nnn/README.md](README.md)
- **Examples**: [nnn/example.lua](example.lua)
- **Tests**: [nnn/test.lua](test.lua)
- **Related**: [nn/NNN_DOCUMENTATION.md](../nn/NNN_DOCUMENTATION.md)

## Help & Support

For issues or questions:
1. Check examples in `nnn/example.lua`
2. Run tests with `th nnn/test.lua`
3. Review full documentation in `nnn/README.md`
4. See integration examples in `nn/integration_example_nnn.lua`

---

**Quick Start Summary**:
```lua
local nnn = require 'nnn'

-- Transform any module
local model = nnn.transform(nn.Linear(10, 5))

-- Or use pre-built
local model = nnn.Linear(10, 5)

-- Use with nested input
local input = {torch.randn(10), torch.randn(10)}
local output = model:forward(input)
```

**Remember**: `nnn.*` extends `nn.*` to work with tree-structured nested tensors!
