# Torch7u - Unified Torch7 Repository

This repository contains all repositories from the [torch](https://github.com/torch) GitHub organization, deeply integrated as a cohesive framework with interconnected features and functions.

## ⚡ Key Features

- **Deep Integration**: All components work together seamlessly through a unified integration layer
- **Event-Driven Architecture**: Cross-module communication via a powerful event system
- **Unified Configuration**: Shared settings across all components
- **Model Registry**: Centralized model management
- **Metrics & Monitoring**: Unified metrics collection across all modules
- **Plugin System**: Extensible architecture for custom functionality
- **Training Framework**: Integrated training interface that works across all modules
- **Node9 OS Framework**: System-level operations with Penrose Lua utilities, kernel functions, and schedulers
- **Tensor Logic**: Neuro-symbolic AI unifying symbolic logic and neural networks through Einstein summation

## 🚀 Quick Start

```lua
-- Load the integrated torch7u system
require 'init'

-- Display integration information
torch7u.info()

-- Configure the system
torch7u.configure({
    default_tensor_type = 'torch.FloatTensor',
    cuda_enabled = true
})

-- Use integrated features
local model = nn.Sequential():add(nn.Linear(10, 5))
torch7u.models.register('my_model', model)

-- Use Tensor Logic for neuro-symbolic AI
local tl = torch7u.tensor_logic
tl.demo()  -- Run logic programming and MLP demos
```

See [INTEGRATION.md](INTEGRATION.md) for complete documentation and examples.

## Contents

This monorepo includes the following torch components:

### Core Libraries
- **torch7** - The main Torch7 scientific computing framework
- **nn** - Neural network modules
- **optim** - Optimization algorithms
- **cutorch** - CUDA backend for Torch
- **cunn** - CUDA neural network modules

### Utilities
- **xlua** - Extended Lua utilities
- **sys** - System utilities
- **paths** - Path manipulation utilities
- **class** - Object-oriented programming for Lua
- **argcheck** - Argument checking utilities
- **cwrap** - C wrapper generator

### Graphics & Visualization
- **gnuplot** - Gnuplot interface
- **image** - Image processing library
- **qttorch** - Qt bindings for Torch
- **qtlua** - Qt bindings for Lua
- **cairo-ffi** - Cairo graphics FFI bindings

### Neural Networks
- **nngraph** - Neural network graph module
- **rnn** - Recurrent neural network modules
- **tds** - Torch data structures

### Build & Distribution
- **distro** - Torch distribution
- **luajit-rocks** - LuaJIT with LuaRocks
- **rocks** - LuaRocks packages
- **luarocks-mirror** - LuaRocks mirror

### Documentation & Examples
- **tutorials** - Torch tutorials
- **demos** - Demo applications
- **dok** - Documentation system
- **torch.github.io** - Website source

### Other Components
- **TH** - Tensor library
- **trepl** - REPL for Torch
- **threads** - Multi-threading support
- **ffi** - FFI utilities
- **graph** - Graph library
- **hash** - Hash utilities
- **vector** - Vector utilities
- **senna** - SENNA NLP system
- **rational** - Rational number support
- **testme** - Testing framework
- **torchunit** - Unit testing
- **socketfile** - Socket file utilities
- **sdl2-ffi** - SDL2 FFI bindings
- **sundown-ffi** - Sundown markdown FFI
- **env** - Environment utilities
- **ezinstall** - Easy installation scripts
- **xt** - Extended tensor operations
- **nimbix-admin** - Nimbix administration
- **node9** - Node9 OS framework (Inferno OS with Lua/LuaJIT)
- **tensor-logic** - Neuro-symbolic AI via Tensor Logic (Pedro Domingos' unified framework)

## Source

All code is sourced from https://github.com/torch

**Node9 integration** sourced from https://github.com/o9nn/node9

## Integration Architecture

Torch7u provides a comprehensive integration layer that connects all components:

- **`init.lua`** - Main integration layer and unified entry point
- **`integrations/`** - Module-specific integration bridges
- **`INTEGRATION.md`** - Complete integration guide and documentation
- **`test_integration.lua`** - Integration test suite
- **`example_integration.lua`** - Example demonstrating deep integration

Key integration features:
- Module registry with dependency management
- Event system for cross-module communication
- Shared configuration across all components
- Unified logging and metrics
- Model registry for centralized management
- Data pipelines that work across all modules
- Plugin system for extensibility

## Usage

### Basic Usage

```lua
require 'init'  -- Load integrated torch7u system

-- Use any torch module - they all work together
local model = nn.Sequential()
    :add(nn.Linear(784, 256))
    :add(nn.ReLU())

-- Register with integrated model registry
torch7u.models.register('classifier', model)

-- Use integrated training interface
local trainer = torch7u.training.create_trainer(model, criterion, config)
```

### Integration Bridges

The repository includes specialized integration bridges:

```lua
-- Load nn-optim integration
local nn_optim = require 'integrations.nn_optim_bridge'
local training_loop = nn_optim.create_training_loop(model, criterion, optimizer)

-- Load image-nn integration  
local image_nn = require 'integrations.image_nn_bridge'
local pipeline = image_nn.create_preprocessing_pipeline(transforms)
```

### Node9 OS Framework

Access system-level operations and Penrose Lua utilities:

```lua
-- Access node9 through torch7u
local node9 = torch7u.node9

-- Or require directly
local node9 = require 'node9'

-- Load Penrose Lua utilities
node9.pl.load()

-- Use advanced collections
local List = node9.pl.List
local mylist = List({1, 2, 3, 4, 5})
local doubled = mylist:map(function(x) return x * 2 end)

-- Use string utilities
local stringx = node9.pl.stringx
local parts = stringx.split("a,b,c", ",")

-- Use table utilities
local tablex = node9.pl.tablex
local merged = tablex.merge({a=1}, {b=2}, true)

-- See node9/README.md for complete documentation
```

### Tensor Logic: Neuro-Symbolic AI

Use tensor logic to unify symbolic AI and neural networks:

```lua
-- Access tensor logic through torch7u
local tl = torch7u.tensor_logic

-- Example: Logic Programming (Symbolic AI)
local Parent = tl.fromMatrix('Parent', {'x', 'y'}, {
    {0, 1, 0, 0},  -- Alice is parent of Bob
    {0, 0, 1, 1}   -- Bob is parent of Charlie and Diana
})

-- Compute transitive closure: Ancestor[x,z] = Σ_y Ancestor[x,y] · Parent[y,z]
local Ancestor = tl.clone(Parent)
local newAncestors = tl.einsum('xy,yz->xz', Ancestor, Parent)
-- Result: Alice is now ancestor of Charlie and Diana!

-- Example: Neural Network (MLP for XOR)
local Input = tl.fromVector('Input', 'i', {1, 0})
local W1 = tl.createTensor('W1', {'h', 'i'}, {2, 2}, {1, 1, 1, 1})
local Hidden = tl.relu(tl.einsum('hi,i->h', W1, Input))
-- Output layer with sigmoid activation
local W2 = tl.createTensor('W2', {'o', 'h'}, {1, 2}, {-2, 2})
local Output = tl.sigmoid(tl.einsum('oh,h->o', W2, Hidden))
-- XOR(1,0) = 0.73 ≈ 1 ✓

-- Run complete demos
tl.demo()  -- Shows both logic programming and neural network examples

-- See tensor-logic/README.md for complete documentation
```

### Running Tests

```lua
-- Run integration tests
th test_integration.lua

-- Run node9 tests
th node9/test.lua

-- Run node9 examples
th node9/example.lua

-- Run integration example
th example_integration.lua

-- Run tensor logic tests
th tensor-logic/test/test.lua

-- Run tensor logic examples
th tensor-logic/example.lua
th tensor_logic_integration_example.lua
```

## Integration Benefits

1. **Seamless Interoperability**: All modules work together without manual coordination
2. **Event-Driven**: Modules communicate via events, enabling loose coupling
3. **Unified Interfaces**: Common patterns for training, data loading, and model management
4. **Centralized Configuration**: Single source of truth for all settings
5. **Enhanced Functionality**: Integration adds features not available in individual modules
6. **Extensible**: Plugin system allows easy addition of new features

## Documentation

- **[INTEGRATION.md](INTEGRATION.md)** - Complete integration guide
- **[example_integration.lua](example_integration.lua)** - Working example
- **[node9/README.md](node9/README.md)** - Node9 OS framework integration guide
- **[node9/example.lua](node9/example.lua)** - Node9 integration examples
- **[tensor-logic/README.md](tensor-logic/README.md)** - Tensor Logic neuro-symbolic AI guide
- **[tensor_logic_integration_example.lua](tensor_logic_integration_example.lua)** - Tensor Logic integration example
- **Component README files** - Individual module documentation in subdirectories

## License

Individual components retain their original licenses. Please refer to each subdirectory for specific licensing information.

---
*Aggregated on: December 26, 2025*
