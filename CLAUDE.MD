# Torch7u - Claude Code Context

## Project Overview

Torch7u is a unified repository containing all Torch7 components deeply integrated as a cohesive framework. It extends the original Torch7 ecosystem with modern features including nested neural nets (nnn), Geometric Unity (gu) extensions, and a comprehensive integration layer.

## Directory Structure

```
torch7u/
├── init.lua           # Main integration layer and entry point
├── torch7/            # Core Torch7 scientific computing framework
├── nn/                # Neural network modules (with NNN extensions)
├── nnn/               # Nested Neural Nets - functional operators for nested tensors
├── gu/                # Geometric Unity extension module
├── optim/             # Optimization algorithms
├── cutorch/           # CUDA backend
├── cunn/              # CUDA neural network modules
├── rnn/               # Recurrent neural network modules
├── nngraph/           # Neural network graph module
├── image/             # Image processing library
├── integrations/      # Module-specific integration bridges
├── demos/             # Demo applications
└── tutorials/         # Torch tutorials
```

## Key Modules

### NNN (Nested Neural Nets)
Location: `nnn/init.lua`

The nnn module transforms nn.* operations to work with nested tensors (nestors). Key concepts:
- `nnn.transform(module)` - Wraps any nn module for nested tensor support
- `nnn.NestedTensor` - Core nested tensor data structure
- `nnn.PrimeFactorType` - Type system based on prime factorizations
- Supports recursive processing up to configurable depth

### GU (Geometric Unity)
Location: `gu/`

Implements computational structures from Eric Weinstein's Geometric Unity theory:

**Constants:**
- `gu.BASE_DIM = 4` - Base spacetime dimension (X^4)
- `gu.FIBER_DIM = 10` - Metric fiber dimension
- `gu.CHIMERIC_DIM = 14` - Total Chimeric Bundle dimension
- `gu.SPINOR_DIM = 128` - GU spinor dimension (2^7)

**Components:**
- `gu.ObserverseTensor` - Two-space structure (base + fiber)
- `gu.ChimericBundle` - 14D bundle vectors
- `gu.ShiabOperator` - Ship-in-a-bottle operator (k-forms to (k+2)-forms)
- `gu.SwerveModule` - Computes Swervature tensor
- `gu.GaugeTransformer` - Gauge transformations (tilted, standard, inhomogeneous)
- `gu.GeneralizedGaugeTransformer` - Full generalized gauge transformer architecture
- `gu.GULayer` - Composite GU dynamics layer

## Code Patterns

### Loading the Framework
```lua
require 'init'  -- Load integrated torch7u system
require 'gu'    -- Load Geometric Unity extension
```

### Creating ObserverseTensor
```lua
local base = torch.randn(batch, 4)
local fiber = torch.randn(batch, 10)
local obs = gu.ObserverseTensor.create(base, fiber)
```

### Using Gauge Transformers
```lua
-- Standard gauge transformer
local gauge = gu.GaugeTransformer(10, {gauge_type = 'tilted', learnable = true})

-- Generalized gauge transformer with attention
local gen_gauge = gu.GeneralizedGaugeTransformer(10, {
    num_heads = 4,
    structure_group = 'SO',
    use_attention = true
})
```

### Building GU Models
```lua
local model = nn.Sequential()
model:add(gu.GULayer(10, {activation = 'tanh', use_residual = true}))
model:add(gu.GeneralizedGaugeTransformer(10, {num_heads = 4}))
```

## Testing

```bash
# Run integration tests
th test_integration.lua

# Run example
th example_integration.lua
```

## Build Commands

The project uses LuaJIT and LuaRocks for building:
```bash
# From luajit-rocks directory
mkdir build && cd build
cmake ..
make
```

## Integration Layer

The integration layer (`init.lua`) provides:
- Module registry with dependency management
- Event system for cross-module communication
- Shared configuration (`torch7u.configure({...})`)
- Model registry (`torch7u.models.register(name, model)`)
- Unified logging and metrics
- Plugin system for extensibility

## Conventions

1. **Module Pattern**: Each module returns a table with functions and classes
2. **Class Definition**: Uses `torch.class('ClassName', 'ParentClass')` pattern
3. **nn.Module Interface**: Custom layers implement `updateOutput`, `updateGradInput`, `accGradParameters`
4. **ObserverseTensor**: Check with `type(input) == 'table' and input._type == 'ObserverseTensor'`

## Mathematical Background

### Geometric Unity Concepts
- **Observerse**: Coupled two-space structure replacing single spacetime
- **Chimeric Bundle**: 14D space combining base (4D) and fiber (10D)
- **Shiab Operator**: Maps ad-valued k-forms to (k+2)-forms
- **Swervature**: Tensor from GU field equations: `Shiab(F_A) + star(Torsion) = 0`
- **Tilted Gauge Group**: Non-standard gauge group action on both components

### Generalized Gauge Transformer Architecture
The generalized gauge transformer extends standard gauge transformations with:
- Multi-head attention over gauge connections
- Structure group constraints (SO(n), SU(n), Spin(n))
- Lie algebra parameterization for smooth gauge group manifold
- Parallel transport along fiber directions
- Curvature-aware transformations

## File Naming

- `*.lua` - Lua source files
- Component classes: PascalCase (e.g., `GaugeTransformer.lua`)
- Modules: lowercase with underscores in integration bridges

## Dependencies

- torch7 (core tensor library)
- nn (neural network modules)
- optim (optimization)
- (Optional) cutorch/cunn for CUDA support
