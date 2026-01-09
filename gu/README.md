# GU: Geometric Unity Extension for Torch7u

## Overview

The `gu` module provides a computational implementation of key mathematical structures from Eric Weinstein's **Geometric Unity** (GU) theory. This extension leverages torch7u's nested tensor (`nnn`) system to represent the complex geometric objects central to GU, enabling researchers to explore computational implementations of this theoretical physics framework.

## Core Concepts

### The Observerse

The Observerse represents a fundamental departure from standard spacetime models by replacing a single 4-dimensional spacetime manifold with a coupled two-space structure. In the `gu` module, this is represented by the `ObserverseTensor` data structure, which contains:

- **Base space** (4 dimensions): Represents the familiar spacetime coordinates
- **Fiber space** (10 dimensions): Represents the space of possible metrics

### The Chimeric Bundle

The Chimeric Bundle is a 14-dimensional structure that combines the base and fiber spaces. It provides a natural home for GU spinors (128-dimensional objects) and is the arena where the unified field equations operate.

### The Shiab Operator

The "Ship-in-a-Bottle" (Shiab) operator is a family of differential operators that acts on forms valued in the adjoint bundle. It maps ad-valued k-forms to ad-valued (k+2)-forms and is central to constructing gauge-invariant quantities in GU.

### The Swerve and Swervature

The Swerve is a specific instance of the Shiab operator that computes the Swervature tensor—a central component of the GU field equations:

```
Shiab_{ε,σ}(F_A) + ⋆(Augmented Torsion) = 0
```

## Installation

The `gu` module is automatically available when you require torch7u:

```lua
require 'init'  -- Load torch7u
require 'gu'    -- Load Geometric Unity extension
```

## Quick Start

### Basic Usage

```lua
require 'init'
require 'gu'

-- Display module information
gu.info()

-- Create an ObserverseTensor
local base = torch.randn(4)    -- 4D spacetime vector
local fiber = torch.randn(10)  -- 10D fiber vector
local observerse = gu.ObserverseTensor.create(base, fiber)

print(observerse)
```

### Using GU Operators

```lua
-- Create a Shiab operator
local shiab = gu.ShiabOperator(10, 10)

-- Apply to fiber component
local output = shiab:forward(observerse)

-- Create a Swerve module (computes Swervature)
local swerve = gu.SwerveModule(10)
local swervature = swerve:forward(observerse)
```

### Building GU Models

```lua
-- Create a GU layer
local gu_layer = gu.GULayer(10, {
    base_dim = 4,
    use_swerve = true,
    use_gauge = true,
    activation = 'tanh',
    use_residual = true
})

-- Apply to observerse
local output = gu_layer:forward(observerse)

-- Create a multi-layer GU model
local model = gu.createModel({
    num_layers = 3,
    hidden_dim = 10,
    activation = true
})

-- Create a Generalized Gauge Transformer model
local transformer_model = gu.createGaugeTransformerModel({
    num_layers = 3,
    fiber_dim = 10,
    base_dim = 4,
    num_heads = 4,
    structure_group = 'SO',
    use_attention = true,
    use_residual = true,
    use_layernorm = true,
    dropout = 0.1
})

-- Create a hybrid model (GU layers + Gauge Transformer layers)
local hybrid_model = gu.createHybridModel({
    num_gu_layers = 2,
    num_transformer_layers = 2,
    fiber_dim = 10,
    num_heads = 4,
    structure_group = 'SO'
})
```

## API Reference

### Data Structures

#### `gu.ObserverseTensor`

```lua
-- Create from tensors
local obs = gu.ObserverseTensor.create(base_tensor, fiber_tensor)

-- Create from Chimeric vector
local obs = gu.ObserverseTensor.fromChimeric(chimeric_tensor, base_dim)

-- Factory methods
local obs = gu.ObserverseTensor.zeros({batch, 4}, {batch, 10})
local obs = gu.ObserverseTensor.randn({batch, 4}, {batch, 10})

-- Instance methods
obs:clone()
obs:toChimeric()
obs:map(func)
obs:type('torch.CudaTensor')
obs:norm()
```

#### `gu.ChimericBundle`

```lua
-- Create vectors and spinors
local vec = gu.ChimericBundle.randomVector(batch_size)
local spinor = gu.ChimericBundle.randomSpinor(batch_size)

-- Access components
vec:base()   -- First 4 dimensions
vec:fiber()  -- Last 10 dimensions

-- Convert to ObserverseTensor
local obs = vec:toObserverse()
```

### Operators

#### `gu.ShiabOperator`

```lua
local shiab = gu.ShiabOperator(input_dim, output_dim, config)
local output = shiab:forward(input)
local gradInput = shiab:backward(input, gradOutput)
```

#### `gu.SwerveModule`

```lua
local swerve = gu.SwerveModule(dim, {use_torsion = true})
local swervature = swerve:forward(observerse)
```

#### `gu.GaugeTransformer`

```lua
local gauge = gu.GaugeTransformer(dim, {
    gauge_type = 'tilted',  -- 'tilted', 'standard', 'inhomogeneous', 'lie_algebra', 'parallel_transport'
    structure_group = 'SO',  -- 'GL', 'SO', 'SU', 'Spin', 'U'
    learnable = true,
    coupling_strength = 0.1  -- For tilted gauge
})
local transformed = gauge:forward(observerse)

-- Get the current gauge group element
local g = gauge:getGaugeElement()

-- Compute curvature (field strength)
local F = gauge:curvature()
```

#### `gu.GeneralizedGaugeTransformer`

The GeneralizedGaugeTransformer is a full transformer architecture for gauge field transformations:

```lua
local gen_gauge = gu.GeneralizedGaugeTransformer(dim, {
    base_dim = 4,
    num_heads = 4,
    structure_group = 'SO',      -- 'GL', 'SO', 'SU', 'Spin', 'U'
    use_attention = true,        -- Multi-head gauge attention
    use_connection = false,      -- Learn gauge connection
    use_curvature_reg = false,   -- Curvature regularization
    use_residual = true,         -- Residual connections
    use_layernorm = true,        -- Layer normalization
    dropout = 0.1
})

local output = gen_gauge:forward(observerse)

-- Get attention weights
local attn = gen_gauge:getAttentionWeights()

-- Compute curvature loss for regularization
local curv_loss = gen_gauge:curvatureLoss()
```

### Layers

#### `gu.GULayer`

```lua
local layer = gu.GULayer(fiber_dim, {
    base_dim = 4,
    use_swerve = true,
    use_gauge = true,
    use_residual = false,
    activation = 'tanh',  -- 'tanh', 'relu', 'sigmoid', or nil
    dropout = 0.1,
    use_layernorm = false
})
```

### Utility Functions

```lua
-- Check if object is ObserverseTensor
gu.isObserverse(obj)

-- Create random structures
gu.randomObserverse(batch_size)
gu.randomChimeric(batch_size)
gu.randomSpinor(batch_size)

-- Conversions
gu.chimericToObserverse(chimeric)
gu.observerseToChimeric(observerse)

-- Transform nn modules for GU
gu.transform(nn_module, {applyTo = 'fiber'})
```

## Constants

```lua
gu.BASE_DIM = 4       -- Dimension of base spacetime
gu.FIBER_DIM = 10     -- Dimension of metric fiber
gu.CHIMERIC_DIM = 14  -- Total Chimeric Bundle dimension
gu.SPINOR_DIM = 128   -- GU spinor dimension (2^7)
```

## Integration with NNN

The `gu` module integrates seamlessly with torch7u's `nnn` system:

```lua
local nnn = require 'nnn'

-- Transform any nn module to work with ObserverseTensors
local relu = gu.transform(nn.ReLU(), {applyTo = 'fiber'})
local output = relu:forward(observerse)

-- Use nnn utilities
local flattened = nnn.flatten(observerse)
local cloned = nnn.clone(observerse)
```

## Training Example

```lua
require 'init'
require 'gu'

-- Create model
local model = nn.Sequential()
model:add(gu.GULayer(10, {activation = 'tanh'}))
model:add(gu.GULayer(10, {activation = 'tanh'}))

-- Create criterion (custom for ObserverseTensor)
local criterion = nn.MSECriterion()

-- Training loop
local learning_rate = 0.01
for epoch = 1, 100 do
    -- Generate random data
    local input = gu.randomObserverse(32)
    local target = gu.randomObserverse(32)
    
    -- Forward
    local output = model:forward(input)
    
    -- Compute loss on fiber component
    local loss = criterion:forward(output.fiber, target.fiber)
    
    -- Backward
    local gradOutput = gu.ObserverseTensor.create(
        torch.zeros(32, 4),
        criterion:backward(output.fiber, target.fiber)
    )
    model:zeroGradParameters()
    model:backward(input, gradOutput)
    
    -- Update
    model:updateParameters(learning_rate)
    
    if epoch % 10 == 0 then
        print(string.format("Epoch %d, Loss: %.4f", epoch, loss))
    end
end
```

## The Complete Solution

The `gu.Solution` module provides the complete computational solution to Geometric Unity:

### Solving GU

```lua
require 'init'
require 'gu'

-- Create a solver
local solver = gu.Solution.create()

-- Define initial conditions
local initial = {
    curvature = torch.randn(10, 10) * 0.1,  -- Initial fiber curvature
    torsion = torch.zeros(10, 10),           -- Torsion tensor
    matter = torch.zeros(10),                -- Matter current
    spinor = torch.randn(128)                -- 128D GU spinor
}

-- Solve the GU field equations
local result = solver:solve(initial, {
    max_iter = 100,
    tolerance = 1e-6,
    learning_rate = 0.01
})

-- Display the solution
solver:display(result)
```

### Solution Components

- **Lagrangian**: The unified GU action L = L_gravity + L_gauge + L_spinor
- **FieldEquations**: Solves Shiab(F_A) + ⋆T = 0
- **EndogenousObserver**: The embedding ι: X^4 ↪ Y^14
- **EinsteinYangMills**: Gravity-gauge unification
- **SpinorFieldEquations**: 128D Dirac equation

### Visualizing the Theory

```lua
-- Display theory info
gu.Solution.info()

-- Visualize the solution structure
gu.Solution.visualize()
```

### The Unification

The key insight: a single connection A on Y^14 encodes both:
- **Gravity**: Einstein's equations (base projection)
- **Gauge forces**: Yang-Mills equations (fiber projection)

```lua
local eym = gu.Solution.EinsteinYangMills.create()
local curvature = eym:computeCurvature()
local einstein = eym:extractEinstein(curvature)
local yang_mills = eym:extractYangMills(curvature)
```

## Mathematical Background

For a detailed understanding of the mathematical foundations, refer to:

1. [Geometric Unity Official Website](https://geometricunity.org/)
2. [2013 Oxford Lecture Transcript](https://geometricunity.org/2013-oxford-lecture/)
3. [The Portal Wiki - Theory of Geometric Unity](https://theportal.wiki/wiki/Theory_of_Geometric_Unity)

## License

This module is part of the torch7u repository and follows its licensing terms.
