# NNN: Nested Neural Nets Functional Operator System

## Overview

The `nnn` module provides a **functional operator system** that extends any `nn.*` operation to work with nested tensors (nestors). Similar to how the prefix `nn.*` transforms `*` with tensor embeddings, `nnn.*` transforms `*` with nestor (nested tensor) metagraph embeddings.

## Core Concept

```
nn.*  + tensor embedding     → neural operation
nnn.* + nestor embedding     → nested neural operation (metagraph)
```

**Key Insight**: `nnn` acts as a **functional operator** that lifts any `nn` module to work with arbitrarily nested tensor structures while preserving the tree structure of the input.

## Installation

The `nnn` module is automatically available when you require the nn package:

```lua
require 'nn'  -- This makes nnn available
local nnn = require 'nnn'
```

## Quick Start

### Basic Usage: Transform any nn.* module

```lua
local nnn = require 'nnn'

-- Create a standard nn module
local linear = nn.Linear(10, 5)

-- Transform it to work with nested tensors
local nestedLinear = nnn.transform(linear)

-- Use with nested input
local input = {
    torch.randn(3, 10),  -- First branch
    torch.randn(2, 10)   -- Second branch
}

local output = nestedLinear:forward(input)
-- output = {Tensor[3,5], Tensor[2,5]}
```

### Pre-built nnn.* modules

```lua
-- Direct nnn versions of common modules
local model = nnn.Sequential()
    :add(nnn.Linear(10, 20))
    :add(nnn.ReLU())
    :add(nnn.Linear(20, 5))
    :add(nnn.Tanh())

-- Works with nested tensors automatically
local nestedInput = {
    {torch.randn(10), torch.randn(10)},
    torch.randn(10)
}
local output = model:forward(nestedInput)
```

### Modal Classifiers: Extend any criterion

```lua
-- Wrap any criterion to work with nested tensors
local criterion = nnn.Criterion(nn.MSECriterion())

-- Or use pre-built versions
local criterion = nnn.MSECriterion()

-- Use with nested predictions and targets
local predictions = {torch.randn(5), torch.randn(5)}
local targets = {torch.randn(5), torch.randn(5)}

local loss = criterion:forward(predictions, targets)
local gradInput = criterion:backward(predictions, targets)
```

## API Reference

### Core Functions

#### nnn.transform(module, config)

Transform any `nn` module to work with nested tensors.

**Parameters:**
- `module` (nn.Module): Any nn module to transform
- `config` (table, optional): Configuration options
  - `maxDepth` (number, default: 10): Maximum nesting depth
  - `aggregation` (string, default: 'preserve'): How to aggregate outputs

**Returns:** Transformed module that works with nested tensors

**Example:**
```lua
local linear = nn.Linear(10, 5)
local nestedLinear = nnn.transform(linear)
```

#### nnn.wrap(module)

Convenience function for `nnn.transform(module)`.

#### nnn.fromNN(moduleName, ...)

Factory function to create nnn version of any nn module by name.

**Parameters:**
- `moduleName` (string): Name of nn module (without 'nn.' prefix)
- `...`: Arguments to pass to the nn module constructor

**Example:**
```lua
local nestedConv = nnn.fromNN('SpatialConvolution', 3, 64, 3, 3)
-- Equivalent to: nnn.transform(nn.SpatialConvolution(3, 64, 3, 3))
```

### Pre-built Modules

All pre-built modules follow the pattern: `nnn.<ModuleName>` mirrors `nn.<ModuleName>`

#### Container Modules

- **nnn.Sequential()**: Sequential container for nested tensors

#### Linear Modules

- **nnn.Linear(inputSize, outputSize, bias)**: Linear transformation

#### Activation Functions

- **nnn.Tanh()**: Hyperbolic tangent activation
- **nnn.ReLU(inplace)**: Rectified linear unit
- **nnn.Sigmoid()**: Sigmoid activation
- **nnn.SoftMax()**: Softmax activation

### Criteria (Modal Classifiers)

#### nnn.Criterion(criterion)

Transform any criterion to work with nested tensors.

**Example:**
```lua
local nestedCriterion = nnn.Criterion(nn.MarginRankingCriterion())
```

#### Pre-built Criteria

- **nnn.MSECriterion()**: Mean squared error for nested tensors
- **nnn.ClassNLLCriterion(weights)**: Negative log-likelihood
- **nnn.BCECriterion(weights)**: Binary cross-entropy
- **nnn.CrossEntropyCriterion(weights)**: Cross-entropy loss

### Utility Functions

#### nnn.isNested(input)

Check if input is a nested structure.

**Returns:** `true` if input is a table (not a tensor), `false` otherwise

#### nnn.depth(input)

Get nesting depth of structure.

**Returns:** Maximum depth of nesting

#### nnn.flatten(input)

Flatten nested structure to array of tensors.

**Returns:** Array containing all tensors in the structure

#### nnn.clone(input)

Deep clone nested structure.

**Returns:** Independent copy of the nested structure

#### nnn.map(input, func)

Apply function to all tensors in nested structure.

**Parameters:**
- `input`: Nested tensor structure
- `func` (function): Function to apply to each tensor

**Returns:** New nested structure with transformed tensors

**Example:**
```lua
local nested = {torch.ones(3), {torch.ones(2)}}
local scaled = nnn.map(nested, function(t) return t * 2 end)
```

## Advanced Usage

### Hierarchical Document Processing

```lua
local nnn = require 'nnn'

-- Build a model for hierarchical text processing
local docModel = nnn.Sequential()
    :add(nn.NestedEmbedding(10000, 128, 3))  -- Word embeddings at each level
    :add(nnn.Linear(128, 64))                 -- Process embeddings
    :add(nnn.Tanh())                          -- Non-linearity

-- Document structure: paragraphs -> sentences -> words
local document = {
    -- Paragraph 1
    {
        torch.LongTensor({1, 2, 3, 4}),      -- Sentence 1
        torch.LongTensor({5, 6, 7})          -- Sentence 2
    },
    -- Paragraph 2
    {
        torch.LongTensor({8, 9}),            -- Sentence 1
        torch.LongTensor({10, 11, 12, 13})   -- Sentence 2
    }
}

local output = docModel:forward(document)
```

### Transform Existing Models

```lua
-- Take an existing pre-trained model
local pretrainedModel = torch.load('model.t7')

-- Transform it to work with nested inputs
local nestedModel = nnn.transform(pretrainedModel)

-- Now it works with both single tensors and nested structures
local singleOutput = nestedModel:forward(torch.randn(10))
local nestedOutput = nestedModel:forward({torch.randn(10), torch.randn(10)})
```

### Custom Aggregation

```lua
-- Transform with custom configuration
local module = nnn.transform(nn.Linear(10, 5), {
    maxDepth = 5,
    aggregation = 'preserve'  -- Keep nested structure
})
```

### Training with Nested Data

```lua
local nnn = require 'nnn'

-- Model
local model = nnn.Sequential()
    :add(nnn.Linear(784, 256))
    :add(nnn.ReLU())
    :add(nnn.Linear(256, 10))

-- Criterion
local criterion = nnn.MSECriterion()

-- Training loop
local function train(input, target, learningRate)
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

-- Works with nested inputs/targets
local nestedInput = {
    {torch.randn(784), torch.randn(784)},
    torch.randn(784)
}
local nestedTarget = {
    {torch.randn(10), torch.randn(10)},
    torch.randn(10)
}

local loss = train(nestedInput, nestedTarget, 0.01)
```

## Integration with Metagraph Type System

The `nnn` module integrates with the prime factorization type system:

```lua
local nnn = require 'nnn'

-- Access type utilities
local PrimeFactorType = nnn.PrimeFactorType

-- Analyze tensor types
local tensor = torch.Tensor(4, 6)
local metagraphType = PrimeFactorType.getMetagraphType(tensor)

print("Type ID:", metagraphType.typeId)           -- "[2.2][2.3]"
print("Prime factors:", metagraphType.signature)  -- {{2,2}, {2,3}}

-- Check compatibility
local tensor2 = torch.Tensor(4, 6)
local compatible = PrimeFactorType.isCompatible(tensor, tensor2)  -- true
```

## How It Works

### Functional Operator Pattern

The `nnn.transform` function creates a **functional operator** that:

1. **Recursively processes nested structures**: Traverses the tree of tensors
2. **Applies the wrapped module to each leaf**: Preserves semantics
3. **Maintains tree structure**: Output has same nesting as input
4. **Handles gradients correctly**: Backpropagation through nested structures

### Architecture

```
Input (nested)
    │
    ├─ nnn.NestedOperator (wrapper)
    │    │
    │    ├─ Recursive processing
    │    │
    │    └─ Wrapped nn.Module
    │         │
    │         └─ Applied to each tensor
    │
    └─ Output (nested, same structure as input)
```

## Comparison: nn vs nnn

| Feature | nn.* | nnn.* |
|---------|------|-------|
| Input type | Single tensor | Nested tensors (tree structure) |
| Output type | Single tensor | Nested tensors (preserves structure) |
| Use case | Standard neural networks | Hierarchical/tree-structured data |
| Embedding | Tensor embedding | Nestor (nested tensor) metagraph embedding |
| Type system | Standard tensors | Prime factorization metagraph types |

## Examples

### Example 1: Simple Transformation

```lua
local nnn = require 'nnn'

-- Standard nn module
local linear = nn.Linear(10, 5)

-- Transform to nnn
local nestedLinear = nnn.wrap(linear)

-- Single tensor (works as normal)
local out1 = nestedLinear:forward(torch.randn(10))  -- Tensor[5]

-- Nested tensor (preserves structure)
local out2 = nestedLinear:forward({
    torch.randn(10),
    torch.randn(10)
})  -- {Tensor[5], Tensor[5]}
```

### Example 2: Building Complex Models

```lua
-- Create a complex nested model
local model = nnn.Sequential()

-- Layer 1: Nested embedding
model:add(nn.NestedEmbedding(5000, 256, 3))

-- Layer 2: Nested linear transformation
model:add(nnn.Linear(256, 128))

-- Layer 3: Nested activation
model:add(nnn.Tanh())

-- Layer 4: Another transformation
model:add(nnn.Linear(128, 64))

-- Use with deeply nested input
local input = {
    {
        torch.LongTensor({1, 2, 3}),
        torch.LongTensor({4, 5})
    },
    {
        torch.LongTensor({6, 7, 8, 9})
    }
}

local output = model:forward(input)
-- Output preserves the same nested structure
```

### Example 3: Modal Classifier

```lua
-- Train a classifier with nested outputs
local classifier = nnn.Sequential()
    :add(nnn.Linear(100, 50))
    :add(nnn.ReLU())
    :add(nnn.Linear(50, 10))

local criterion = nnn.ClassNLLCriterion()

-- Multi-branch prediction
local input = {
    torch.randn(100),  -- Branch 1
    torch.randn(100),  -- Branch 2
    torch.randn(100)   -- Branch 3
}

local target = {
    torch.LongTensor({3}),  -- Label for branch 1
    torch.LongTensor({7}),  -- Label for branch 2
    torch.LongTensor({1})   -- Label for branch 3
}

local predictions = classifier:forward(input)
local loss = criterion:forward(predictions, target)
```

## GU (Geometric Unity) Integration

The NNN module provides deep integration with the GU (Geometric Unity) framework, allowing nested neural networks to work with ObserverseTensor structures.

### Overview

The NNN-GU integration enables:
- Processing nested structures of ObserverseTensors
- Applying neural operations to base and/or fiber components
- GU-aware criteria for loss computation
- Seamless composition of nested geometric structures

### Quick Start with GU

```lua
local nnn = require 'nnn'
local gu = require 'gu'

-- Create a GU-aware linear layer
local guLinear = nnn.GULinear(10, 20, {applyTo = 'fiber'})

-- Works with ObserverseTensors
local obs = gu.randomObserverse(4)
local output = guLinear:forward(obs)

-- Works with nested ObserverseTensors
local nested = {
    gu.randomObserverse(4),
    {gu.randomObserverse(4), gu.randomObserverse(4)}
}
local nested_output = guLinear:forward(nested)
```

### GU-Aware Transform

```lua
-- Transform any nn module for GU support
local guModule = nnn.transformGU(nn.Linear(10, 10), {
    applyTo = 'fiber',    -- 'base', 'fiber', or 'both'
    preserveGU = true,     -- Preserve ObserverseTensor type
    maxDepth = 10          -- Maximum nesting depth
})
```

### Hybrid GU Modules

Pre-built modules for nested ObserverseTensors:

| Module | Description |
|--------|-------------|
| `nnn.GULinear(in, out, config)` | Linear layer for nested ObserverseTensors |
| `nnn.GUReLU(config)` | ReLU activation |
| `nnn.GUTanh(config)` | Tanh activation |
| `nnn.GUSigmoid(config)` | Sigmoid activation |
| `nnn.GUSoftMax(config)` | SoftMax activation |
| `nnn.GUDropout(p, config)` | Dropout |
| `nnn.GUBatchNormalization(n, config)` | Batch normalization |
| `nnn.GULayer(fiber_dim, config)` | Full GU layer (Swerve + Gauge) |
| `nnn.GUSequential()` | Sequential container for GU models |

### GU-Specific Criteria

```lua
-- MSE criterion for nested ObserverseTensors
local criterion = nnn.GUMSECriterion({
    applyTo = 'fiber',      -- Apply to fiber only
    aggregation = 'mean'    -- 'mean', 'sum', or 'max'
})

-- Weighted Observerse MSE (different weights for base/fiber)
local weighted = nnn.ObserverseMSECriterion({
    baseWeight = 0.3,
    fiberWeight = 0.7
})

-- Other criteria
nnn.GUBCECriterion(weights, config)
nnn.GUClassNLLCriterion(weights, config)
nnn.GUCrossEntropyCriterion(weights, config)
```

### GU Utility Functions

```lua
-- Check if input is ObserverseTensor
nnn.gu.isObserverse(input)  -- true/false

-- Flatten nested ObserverseTensors to list
local list = nnn.gu.flatten(nested_input)

-- Count ObserverseTensors in structure
local count = nnn.gu.count(nested_input)

-- Get nesting depth
local depth = nnn.gu.depth(nested_input)

-- Map function over all ObserverseTensors
local mapped = nnn.gu.map(input, function(obs)
    return gu.ObserverseTensor.create(obs.base * 2, obs.fiber)
end)

-- Clone nested structure
local cloned = nnn.gu.clone(nested_input)

-- Generate random nested structure from template
local template = {branch1 = 1, branch2 = {a = 1, b = 1}}
local random = nnn.gu.randomNested(template, batch_size)
```

### Building GU Networks

```lua
-- Complete GU neural network
local model = nnn.GUSequential()
model:add(nnn.GULinear(10, 20, {applyTo = 'fiber'}))
model:add(nnn.GUReLU({applyTo = 'fiber'}))
model:add(nnn.GUDropout(0.1, {applyTo = 'fiber'}))
model:add(nnn.GULinear(20, 10, {applyTo = 'fiber'}))

-- Using full GU layers with Swerve and Gauge
local gu_model = nnn.GUSequential()
gu_model:add(nnn.GULayer(10, {
    use_swerve = true,
    use_gauge = true,
    gauge_type = 'tilted',
    use_residual = true
}))
```

### Training with NNN-GU

```lua
local model = nnn.GUSequential()
model:add(nnn.GULinear(10, 10, {applyTo = 'fiber'}))

local criterion = nnn.GUMSECriterion({applyTo = 'fiber'})
local params, gradParams = model:getParameters()

-- Training loop
for epoch = 1, num_epochs do
    gradParams:zero()
    local output = model:forward(input)
    local loss = criterion:forward(output, target)
    local gradOutput = criterion:backward(output, target)
    model:backward(input, gradOutput)
    params:add(-learning_rate, gradParams)
end
```

### GU Integration Info

Display available GU integration features:

```lua
nnn.gu.info()
```

### Geonestor Neuroglyph

A **Geonestor Neuroglyph** is a geometric nested tensor neural gauge-awareness symmetry structure - the unified formalism for NNN-GU integration.

#### Components

| Component | Meaning |
|-----------|---------|
| **GEO** | Geometric Unity (Observerse, gauge transformations, fiber bundles) |
| **NESTOR** | Nested Tensor structures (recursive tree-shaped data) |
| **NEURO** | Neural network operations (learnable transformations) |
| **GLYPH** | Symbolic representation (type signatures, symmetry invariants) |

#### Creating Neuroglyphs

```lua
local Neuroglyph = nnn.Neuroglyph

-- Create directly
local glyph = Neuroglyph.create({
    name = 'my_glyph',
    baseDim = 4,
    fiberDim = 10,
    gaugeGroup = 'SO',
    depth = 2
})

-- From nested structure
local nested = {gu.randomObserverse(4), {gu.randomObserverse(4)}}
local glyph = nnn.gu.neuroglyph(nested, {name = 'derived'})

-- From model
local model = nnn.GUSequential()
model:add(nnn.GULinear(10, 10))
local glyph = nnn.gu.neuroglyphFromModel(model)
```

#### Neuroglyph API

```lua
-- Signatures and invariants
glyph:signature()       -- "G[name:4|10|SO|d2]"
glyph:primeSignature()  -- {base = {2,2}, fiber = {2,5}}

-- Compatibility checks
glyph:isGaugeCompatible(other)
glyph:isStructurallyCompatible(other)

-- Composition (tensor product)
local composed = glyph:compose(other, {name = 'product'})

-- Model generation
local model = glyph:createModel({
    numLayers = 3,
    activation = 'tanh',
    dropout = 0.1
})

-- Create specific layers
local layer = glyph:createLayer('full', {use_swerve = true})

-- Visualization
glyph:visualize()  -- ASCII art display
```

#### Visualization Output

```
╔══════════════════════════════════════════════════════╗
║            GEONESTOR NEUROGLYPH                      ║
╠══════════════════════════════════════════════════════╣
║  Name:      my_glyph                                 ║
║  Type:      composite                                ║
║  Signature: G[my_glyph:4|10|SO|d2]                   ║
╠══════════════════════════════════════════════════════╣
║  GEOMETRY                                            ║
║    Base Space:    4-dimensional                      ║
║    Fiber Space:   10-dimensional                     ║
║    Chimeric Dim:  14-dimensional                     ║
╠══════════════════════════════════════════════════════╣
║  SYMMETRY                                            ║
║    Gauge Group:   SO (Special Orthogonal)            ║
║    Nesting Depth: 2                                  ║
╠══════════════════════════════════════════════════════╣
║  PRIME SIGNATURE                                     ║
║    Base:  [2·2]                                      ║
║    Fiber: [2·5]                                      ║
╚══════════════════════════════════════════════════════╝
```

### See Also

- [gu_integration_example.lua](gu_integration_example.lua) - Comprehensive GU examples
- [Neuroglyph.lua](Neuroglyph.lua) - Neuroglyph implementation
- [GU README](../gu/README.md) - Geometric Unity module documentation

## Future Extensions

### Operad Gadgets (Planned)

The `nnn.operad` namespace is reserved for future extensions involving:

- **Operad structures**: Composable operations indexed by tree shapes
- **Prime factorization indexing**: Operations organized by type signatures
- **Orbifold symmetries**: Symmetry properties based on prime factorizations

```lua
-- Future API (planned)
nnn.operad.compose(op1, op2, signature)
nnn.operad.symmetry(typeId)
```

## Best Practices

1. **Use nnn.* when working with hierarchical data**: Documents, trees, graphs
2. **Transform existing models**: Use `nnn.transform` to adapt pre-trained models
3. **Leverage type system**: Use `PrimeFactorType` for type-aware processing
4. **Preserve structure**: Nested operations maintain input tree structure
5. **Test with both**: Ensure models work with single tensors and nested inputs

## Performance Notes

- **Memory**: Nested operations scale linearly with number of tensors
- **Computation**: Recursive processing has small overhead
- **Gradients**: Backpropagation through nested structures is efficient
- **Depth limit**: Default max depth is 10 (configurable)

## Related Modules

- **nn.NestedEmbedding**: Embeddings for nested structures
- **nn.NestedNeuralNet**: Complete nested neural network framework
- **nn.PrimeFactorType**: Type system based on prime factorization
- **nn.NestedTensor**: Utility functions for nested tensor operations

## See Also

- [NNN_DOCUMENTATION.md](../nn/NNN_DOCUMENTATION.md) - Complete NNN documentation
- [example_nnn.lua](../nn/example_nnn.lua) - Usage examples
- [test_nnn.lua](../nn/test_nnn.lua) - Test suite

## License

Part of the torch7u unified framework. See individual component licenses.
