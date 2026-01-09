# Tensor Logic - Neuro-Symbolic AI in Lua

A Lua implementation of **Tensor Logic**, a programming paradigm that unifies neural and symbolic AI at a fundamental level.

Based on the paper: [Tensor Logic: The Language of AI](https://arxiv.org/abs/2510.12269) by Prof. Pedro Domingos (University of Washington)

Reference implementation: https://github.com/MrBesterTester/tensor-logic

## The Core Insight

The key insight of Tensor Logic is that **logical rules and Einstein summation are essentially the same operation**:

| Logic Programming | Tensor Algebra |
|------------------|----------------|
| `Ancestor(x,z) ← Ancestor(x,y), Parent(y,z)` | `Ancestor[x,z] = Σ_y Ancestor[x,y] · Parent[y,z]` |
| JOIN on y | Contract over index y |
| PROJECT onto (x,z) | Keep indices x, z |

The only difference is the atomic data type:
- **Boolean (0/1)** → Symbolic logic
- **Real numbers** → Neural networks

This unification allows expressing both symbolic AI and neural networks in the same language.

## Quick Start

```lua
-- Load the tensor logic module
local tl = require 'tensor-logic'

-- Print module information
tl.info()

-- Run a quick demo
tl.demo()

-- Example: Logic Programming
local result = tl.examples.logic.runLogicProgramExample()
print(result.description)

-- Example: Neural Network (MLP)
local mlp_result = tl.examples.mlp.runMLPExample()
print(mlp_result.description)
```

## Features

### Core Tensor Operations

- **Tensor Creation**: `createTensor`, `fromMatrix`, `fromVector`
- **Einstein Summation**: `einsum` - the heart of tensor logic
- **Activation Functions**: `threshold`, `sigmoid`, `relu`, `softmax`
- **Tensor Operations**: `add`, `multiply`, `scale`, `transpose`
- **Utilities**: Element access, cloning, identity matrices

### Examples Included

1. **Logic Programming (Symbolic AI)**: Family relationships with ancestor inference
2. **Multi-Layer Perceptron (Neural Networks)**: XOR problem with batch processing

## Usage Examples

### Logic Programming

```lua
local tl = require 'tensor-logic'

-- Define parent relationships as a Boolean tensor
local Parent = tl.fromMatrix('Parent', {'x', 'y'}, {
    {0, 1, 0, 0},  -- Alice is parent of Bob
    {0, 0, 1, 1},  -- Bob is parent of Charlie and Diana
    {0, 0, 0, 0},
    {0, 0, 0, 0}
})

-- Compute transitive closure using einsum
local Ancestor = tl.clone(Parent)
local newAncestors = tl.einsum('xy,yz->xz', Ancestor, Parent)
local combined = tl.add(Ancestor, tl.threshold(newAncestors))

-- Result: Alice is now ancestor of Charlie and Diana!
```

### Neural Network (MLP)

```lua
local tl = require 'tensor-logic'

-- Define input
local Input = tl.fromVector('Input', 'i', {1, 0})

-- Define weights and biases
local W1 = tl.createTensor('W1', {'h', 'i'}, {2, 2}, {1, 1, 1, 1})
local B1 = tl.fromVector('B1', 'h', {-1.5, -0.5})

-- Forward pass
local preHidden = tl.einsum('hi,i->h', W1, Input)
local Hidden = tl.relu(tl.add(preHidden, B1))

-- Output layer
local W2 = tl.createTensor('W2', {'o', 'h'}, {1, 2}, {-2, 2})
local Output = tl.sigmoid(tl.einsum('oh,h->o', W2, Hidden))

print('XOR(1,0) =', Output.data[1])  -- Should be ~0.73 (rounds to 1)
```

### Einstein Summation Examples

```lua
local tl = require 'tensor-logic'

-- Matrix multiplication: C[i,k] = Σ_j A[i,j] · B[j,k]
local A = tl.fromMatrix('A', {'i', 'j'}, {{1, 2}, {3, 4}})
local B = tl.fromMatrix('B', {'j', 'k'}, {{5, 6}, {7, 8}})
local C = tl.einsum('ij,jk->ik', A, B)

-- Vector dot product: d = Σ_i v1[i] · v2[i]
local v1 = tl.fromVector('v1', 'i', {1, 2, 3})
local v2 = tl.fromVector('v2', 'i', {4, 5, 6})
local dot = tl.einsum('i,i->', v1, v2)

-- Batch matrix multiplication: C[b,i,k] = Σ_j A[b,i,j] · B[j,k]
local batched = tl.einsum('bij,jk->bik', batchA, B)
```

## What This Unifies

### Symbolic AI
- Logic Programming (Datalog, Prolog)
- Theorem Proving
- Knowledge Graphs
- Rule-based Systems

### Neural Networks
- Multi-Layer Perceptrons
- Convolutional Networks
- Transformers (GPT, BERT)
- Attention Mechanisms

### Probabilistic AI
- Bayesian Networks
- Markov Random Fields
- Hidden Markov Models
- Probabilistic Programs

### Hybrid Methods
- Kernel Machines (SVM)
- Graph Neural Networks
- Embedding-based Reasoning

## API Reference

### Core Functions

#### Tensor Creation
- `createTensor(name, indices, shape, initializer)` - Create a new tensor
- `fromMatrix(name, indices, matrix)` - Create tensor from 2D array
- `fromVector(name, index, values)` - Create tensor from 1D array
- `identity(name, indices, size)` - Create identity matrix

#### Einstein Summation
- `einsum(notation, ...)` - Perform Einstein summation over tensors
  - Example: `einsum('ij,jk->ik', A, B)` for matrix multiplication

#### Activation Functions
- `threshold(tensor, t)` - Boolean threshold (values > t become 1)
- `sigmoid(tensor, temperature)` - Smooth sigmoid activation
- `relu(tensor)` - ReLU activation: max(0, x)
- `softmax(tensor, axis)` - Softmax normalization

#### Tensor Operations
- `add(...)` - Element-wise addition
- `multiply(...)` - Element-wise multiplication (Hadamard product)
- `scale(tensor, scalar)` - Scalar multiplication
- `transpose(tensor)` - Matrix transpose
- `clone(tensor)` - Deep copy of tensor

#### Element Access
- `getElement(tensor, ...)` - Get element at indices
- `setElement(tensor, value, ...)` - Set element at indices

#### Utilities
- `tensorToString(tensor, precision)` - Convert to readable string
- `broadcastAdd(tensor, bias, dimIndex)` - Broadcast addition
- `broadcastMultiply(tensor, weights, dimIndex)` - Broadcast multiplication
- `extractSlice(tensor, dimIndex, sliceIndex)` - Extract slice

## Running Tests

```bash
cd tensor-logic/test
lua test.lua
```

The test suite covers:
- Tensor creation and manipulation
- Einstein summation operations
- Activation functions
- Logic programming examples
- Neural network examples

## Why Does This Matter?

- **Unified Language**: Express neural nets, logic programs, and probabilistic models in the same notation
- **Sound Reasoning**: At temperature T=0, embedding-based reasoning becomes exact deduction—no hallucinations
- **Learnable Logic**: Make logical programs differentiable and trainable with gradient descent
- **Transparent AI**: Extract interpretable rules from neural representations

## Integration with Torch7u

This module is designed to work seamlessly with the Torch7u framework. It can be used alongside other torch modules like `nn`, `optim`, and provides a foundation for neuro-symbolic AI research and applications.

```lua
-- In main torch7u system
require 'init'
local tl = torch7u.tensor_logic or require 'tensor-logic'

-- Use with torch tensors
-- (Note: This implementation uses pure Lua for portability)
```

## References

- Domingos, P. (2025). *Tensor Logic: The Language of AI*. [arXiv:2510.12269](https://arxiv.org/abs/2510.12269)
- Website: [tensor-logic.org](https://tensor-logic.org)
- Reference implementation: [github.com/MrBesterTester/tensor-logic](https://github.com/MrBesterTester/tensor-logic)

## License

MIT

---

*Part of the Torch7u unified framework*
*Implementation date: January 2026*
