# Torch7u Integration Guide

## Overview

Torch7u is a unified, deeply integrated monorepo containing all Torch7 components. This guide explains how the various components work together as a cohesive framework.

## Architecture

### Core Integration Layer

The integration layer is provided by the root `init.lua` file, which establishes:

1. **Module Registry System** - Centralized tracking and dependency management
2. **Event System** - Cross-module communication
3. **Shared Configuration** - Unified settings across all components
4. **Utility Functions** - Common functionality available to all modules

### Component Categories

#### Core Libraries
- **torch7** - Main framework with tensor operations
- **nn** - Neural network building blocks
- **optim** - Optimization algorithms
- **TH** - Low-level tensor library

#### GPU Support
- **cutorch** - CUDA backend for torch
- **cunn** - CUDA neural network implementations

#### Utilities
- **xlua** - Extended Lua utilities
- **sys** - System utilities
- **paths** - Path manipulation
- **class** - OOP support
- **argcheck** - Argument validation
- **cwrap** - C wrapper generation

#### Visualization
- **image** - Image processing
- **gnuplot** - Plotting interface
- **qttorch** - Qt GUI bindings
- **qtlua** - Qt Lua bindings

#### Advanced Networks
- **nngraph** - Graph-based neural networks
- **rnn** - Recurrent neural networks
- **tds** - Torch data structures

#### Other Components
- **threads** - Multi-threading
- **ffi** - FFI utilities
- **graph** - Graph library
- **trepl** - Interactive REPL
- **dok** - Documentation system

## Using the Integration Layer

### Basic Usage

```lua
-- Load the integrated torch7u system
require 'init'

-- Display integration information
torch7u.info()

-- Check what modules are loaded
for _, module in ipairs(torch7u.models.list()) do
    print(module)
end
```

### Module Loading

```lua
-- Explicitly load a module
local custom_module = torch7u.load_module('custom_module')

-- Check if CUDA is available
if torch7u.cuda_available then
    print("CUDA support is enabled")
end
```

### Configuration

```lua
-- Configure torch7u
torch7u.configure({
    default_tensor_type = 'torch.FloatTensor',
    logging_level = 'DEBUG',
    cuda_enabled = true,
    auto_gpu = true
})

-- Access configuration
print(torch7u.config.default_tensor_type)
```

### Event System

The event system enables loose coupling between modules:

```lua
-- Subscribe to events
torch7u.events.subscribe('model_registered', function(name, model, metadata)
    print("New model registered:", name)
end, 'my_module')

-- Publish events
torch7u.events.publish('custom_event', arg1, arg2)
```

### Logging

```lua
-- Use unified logging
torch7u.utils.log('INFO', 'Processing started', 'data_loader')
torch7u.utils.log('ERROR', 'Failed to load data', 'data_loader')
torch7u.utils.log('WARNING', 'Low memory', 'trainer')
```

## Deep Integration Features

### 1. Model Registry

Centralized model management across the entire framework:

```lua
-- Register a model
local model = nn.Sequential()
    :add(nn.Linear(784, 256))
    :add(nn.ReLU())
    :add(nn.Linear(256, 10))

torch7u.models.register('mnist_classifier', model, {
    input_size = 784,
    output_size = 10,
    type = 'classifier'
})

-- Retrieve a model anywhere in your code
local model = torch7u.models.get('mnist_classifier')

-- List all registered models
local model_names = torch7u.models.list()
```

### 2. Unified Training Interface

```lua
-- Create a trainer
local trainer = torch7u.training.create_trainer(
    model,
    nn.ClassNLLCriterion(),
    {
        optimizer = 'adam',
        learning_rate = 0.001,
        batch_size = 64,
        epochs = 20
    }
)

-- Add callbacks
trainer:add_callback('epoch_begin', function(epoch, trainer)
    print('Starting epoch', epoch)
end)

trainer:add_callback('epoch_end', function(epoch, trainer)
    print('Completed epoch', epoch)
end)

-- Train
trainer:train(train_data, train_labels)
```

### 3. Data Pipeline

```lua
-- Create a data pipeline
local pipeline = torch7u.data.create_pipeline()

-- Add transforms
pipeline:add_transform('normalize', function(data)
    return (data - data:mean()) / data:std()
end)

pipeline:add_transform('augment', function(data)
    -- Augmentation logic
    return data
end)

-- Process data
local processed = pipeline:process(raw_data, {'normalize', 'augment'})
```

### 4. Metrics and Monitoring

```lua
-- Record metrics
torch7u.metrics.record('train_loss', 0.5, {epoch = 1, batch = 10})
torch7u.metrics.record('val_accuracy', 0.85, {epoch = 1})

-- Subscribe to metric events
torch7u.events.subscribe('metric_recorded', function(name, value, tags)
    print(string.format("Metric %s: %f", name, value))
end)

-- Retrieve metrics
local losses = torch7u.metrics.get('train_loss')
for _, entry in ipairs(losses) do
    print(entry.value, entry.timestamp)
end
```

### 5. Checkpointing

```lua
-- Save checkpoint
torch7u.checkpoint.save('checkpoint_epoch10.t7', {
    model = model,
    optimizer_state = optimizer_state,
    epoch = 10
}, {
    description = 'Model after 10 epochs',
    accuracy = 0.92
})

-- Load checkpoint
local data, metadata = torch7u.checkpoint.load('checkpoint_epoch10.t7')
model = data.model
print('Checkpoint accuracy:', metadata.accuracy)
```

### 6. Plugin System

```lua
-- Create a plugin
local my_plugin = {
    name = 'data_augmentation',
    
    init = function(torch7u)
        print('Initializing data augmentation plugin')
    end,
    
    augment = function(image)
        -- Augmentation logic
        return image
    end
}

-- Register plugin
torch7u.plugins.register('data_augmentation', my_plugin)

-- Use plugin
local plugin = torch7u.plugins.get('data_augmentation')
augmented = plugin.augment(original_image)
```

## Cross-Module Integration Examples

### Example 1: Image Classification with Full Pipeline

```lua
require 'init'

-- Configure for GPU if available
torch7u.configure({ auto_gpu = true })

-- Create model
local model = nn.Sequential()
    :add(nn.SpatialConvolution(3, 32, 5, 5))
    :add(nn.ReLU())
    :add(nn.SpatialMaxPooling(2, 2, 2, 2))
    :add(nn.View(-1):setNumInputDims(3))
    :add(nn.Linear(32*14*14, 10))

-- Auto-transfer to GPU if available
if model.autoGpu then
    model:autoGpu()
end

-- Register model
torch7u.models.register('image_classifier', model)

-- Create data pipeline
local pipeline = torch7u.data.create_pipeline()
pipeline:add_transform('load_image', function(path)
    return image.load(path)
end)
pipeline:add_transform('normalize', function(img)
    return image.rgb2yuv(img)
end)

-- Create trainer with metrics
local trainer = torch7u.training.create_trainer(
    model,
    nn.ClassNLLCriterion(),
    { learning_rate = 0.01, epochs = 10 }
)

trainer:add_callback('epoch_end', function(epoch)
    torch7u.metrics.record('epoch', epoch)
    torch7u.checkpoint.save('checkpoint_e'..epoch..'.t7', { model = model })
end)
```

### Example 2: Multi-threaded Data Loading

```lua
require 'init'

-- Create parallel data loader
if torch7u.cuda_available and threads then
    local loader = torch7u.parallel.create_data_loader(4, function(idx)
        -- Data loading logic per thread
        local data = image.load('data_'..idx..'.jpg')
        return data
    end)
end
```

### Example 3: Custom Integration Plugin

```lua
require 'init'

-- Create a custom plugin that integrates multiple modules
local visualization_plugin = {
    init = function(t7u)
        -- Ensure required modules are loaded
        t7u.load_module('gnuplot')
        t7u.load_module('image')
        
        -- Subscribe to events
        t7u.events.subscribe('metric_recorded', function(name, value)
            if name:match('loss') then
                -- Auto-plot losses
                gnuplot.plot({name, value})
            end
        end)
    end,
    
    visualize_model = function(model, input)
        -- Model visualization logic
    end
}

torch7u.plugins.register('visualization', visualization_plugin)
```

## Module Interconnection Summary

The integration layer creates the following deep interconnections:

1. **torch + nn + optim**: Parameters flow seamlessly between neural network modules and optimizers
2. **nn + image**: Image preprocessing integrates directly with neural network pipelines  
3. **threads + nn**: Parallel data loading for neural network training
4. **cutorch + nn**: Automatic GPU transfer for all neural network components
5. **optim + metrics**: Training statistics automatically flow to the metrics system
6. **All modules + events**: Any module can communicate with any other via the event system
7. **All modules + logging**: Unified logging across the entire framework
8. **All modules + config**: Shared configuration system

## Best Practices

1. **Always use the init.lua entry point** for new projects to get full integration
2. **Register models** in the model registry for centralized management
3. **Use the event system** for loose coupling between components
4. **Leverage the data pipeline** for consistent preprocessing
5. **Record metrics** throughout your code for monitoring
6. **Create plugins** for reusable, cross-cutting functionality
7. **Use checkpointing** for robust training workflows

## Testing Integration

```lua
-- Test that integration is working
require 'init'

-- Verify core modules loaded
assert(torch ~= nil, "torch not loaded")
assert(nn ~= nil, "nn not loaded")
assert(optim ~= nil, "optim not loaded")

-- Test event system
local event_received = false
torch7u.events.subscribe('test_event', function()
    event_received = true
end)
torch7u.events.publish('test_event')
assert(event_received, "Event system not working")

-- Test model registry
local test_model = nn.Linear(10, 10)
torch7u.models.register('test', test_model)
assert(torch7u.models.get('test') == test_model, "Model registry not working")

print("All integration tests passed!")
```

## Troubleshooting

### Module Not Found
```lua
-- Check if module is loaded
local mod = torch7u.load_module('module_name')
if not mod then
    print("Module not available")
end
```

### CUDA Issues
```lua
-- Check CUDA availability
if torch7u.cuda_available then
    print("CUDA is available")
else
    print("CUDA is not available")
end
```

### Event Not Firing
```lua
-- List all event handlers
for event_name, handlers in pairs(torch7u.event_handlers) do
    print(event_name, #handlers)
end
```

## Contributing

When adding new modules or features:

1. Register your module in the module registry
2. Publish events for important operations
3. Subscribe to relevant events from other modules
4. Use the shared configuration system
5. Add logging for debugging
6. Document integration points

## Conclusion

Torch7u provides a deeply integrated environment where all components work together seamlessly. The integration layer eliminates the need for manual module coordination and provides powerful cross-cutting features like events, metrics, and unified configuration.
