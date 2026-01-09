# Torch7u Quick Start Guide

This guide will help you get started with the integrated Torch7u framework.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/e9-o9/torch7u.git
cd torch7u
```

2. The repository includes all torch components pre-integrated. No additional installation needed for the integration layer.

## Basic Usage

### 1. Load the Integration Layer

```lua
-- Load torch7u integration
require 'init'

-- Display system information
torch7u.info()
```

### 2. Configure the System

```lua
-- Set global configuration
torch7u.configure({
    default_tensor_type = 'torch.DoubleTensor',
    logging_level = 'INFO',
    cuda_enabled = false
})
```

### 3. Create a Simple Neural Network

```lua
-- Create a model using nn
local model = nn.Sequential()
    :add(nn.Linear(784, 256))
    :add(nn.ReLU())
    :add(nn.Dropout(0.5))
    :add(nn.Linear(256, 10))
    :add(nn.LogSoftMax())

-- Register in model registry
torch7u.models.register('mnist_classifier', model, {
    task = 'classification',
    dataset = 'mnist'
})
```

### 4. Use Integrated Training

```lua
-- Load nn-optim bridge
local nn_optim = require 'integrations.nn_optim_bridge'

-- Create criterion
local criterion = nn.ClassNLLCriterion()

-- Create training loop
local training_loop = nn_optim.create_training_loop(
    model,
    criterion,
    optim.sgd,
    {
        learningRate = 0.01,
        momentum = 0.9
    }
)

-- Create dummy dataset for example
local train_dataset = {
    data = torch.rand(1000, 784),
    labels = torch.random(1000, 10),
    size = function() return 1000 end
}

-- Train
training_loop:fit(train_dataset, nil, 10, 32)
```

### 5. Use Event System

```lua
-- Subscribe to training events
torch7u.events.subscribe('epoch_end', function(epoch, loss)
    print(string.format('Epoch %d completed, loss: %.4f', epoch, loss))
end, 'my_app')

-- Publish custom events
torch7u.events.publish('model_updated', model)
```

### 6. Track Metrics

```lua
-- Record metrics during training
torch7u.metrics.record('train_loss', 0.5, {epoch = 1})
torch7u.metrics.record('val_accuracy', 0.85, {epoch = 1})

-- Retrieve metrics
local losses = torch7u.metrics.get('train_loss')
for i, entry in ipairs(losses) do
    print(string.format('Epoch %d: %.4f', i, entry.value))
end
```

### 7. Save and Load Checkpoints

```lua
-- Save checkpoint
torch7u.checkpoint.save('model_checkpoint.t7', {
    model = model,
    epoch = 10,
    optimizer_state = {}
}, {
    description = 'Model after 10 epochs',
    accuracy = 0.92
})

-- Load checkpoint
local data, metadata = torch7u.checkpoint.load('model_checkpoint.t7')
model = data.model
print('Loaded model from epoch:', data.epoch)
```

### 8. Create Data Pipeline

```lua
-- Create preprocessing pipeline
local pipeline = torch7u.data.create_pipeline()

-- Add transforms
pipeline:add_transform('normalize', function(x)
    return (x - x:mean()) / x:std()
end)

pipeline:add_transform('augment', function(x)
    -- Your augmentation logic
    return x
end)

-- Process data
local processed = pipeline:process(raw_data, {'normalize', 'augment'})
```

## Advanced Usage

### Image Processing Integration

```lua
-- Load image-nn bridge
local image_nn = require 'integrations.image_nn_bridge'

-- Create image preprocessing pipeline
local img_pipeline = image_nn.create_preprocessing_pipeline({
    {type = 'resize', width = 224, height = 224},
    {type = 'normalize', mean = {0.5, 0.5, 0.5}, std = {0.5, 0.5, 0.5}}
})

-- Process an image
local img = image.load('path/to/image.jpg')
local processed_img = img_pipeline:process(img)

-- Create data augmenter
local augmenter = image_nn.create_augmenter({
    hflip_prob = 0.5,
    rotation_range = {-15, 15}
})

local augmented = augmenter:augment(img)
```

### Parallel Processing

```lua
-- Load threads-nn bridge
local threads_nn = require 'integrations.threads_nn_bridge'

-- Create parallel data loader
local loader = threads_nn.create_data_loader(4, dataset, {
    batch_size = 32,
    shuffle = true
})

-- Use in training
for batch_data, batch_labels in loader:iterator() do
    -- Training step
end

loader:terminate()
```

### Model Ensemble

```lua
local threads_nn = require 'integrations.threads_nn_bridge'

-- Create ensemble of models
local ensemble = threads_nn.create_model_ensemble({
    model1,
    model2,
    model3
}, 'average')

-- Use for prediction
local output = ensemble:forward(input)
```

### Learning Rate Scheduling

```lua
-- Create LR scheduler
local scheduler = model:createLRScheduler(0.1, 'exponential', {
    gamma = 0.95
})

-- Update learning rate each epoch
for epoch = 1, 100 do
    local new_lr = scheduler:step(epoch)
    -- Update optimizer config with new_lr
end
```

### Custom Plugins

```lua
-- Create a custom plugin
local my_plugin = {
    init = function(torch7u)
        print('Plugin initialized')
        
        -- Subscribe to events
        torch7u.events.subscribe('model_registered', function(name)
            print('Model registered:', name)
        end)
    end,
    
    custom_function = function()
        return "Hello from plugin"
    end
}

-- Register plugin
torch7u.plugins.register('my_plugin', my_plugin)

-- Use plugin
local plugin = torch7u.plugins.get('my_plugin')
print(plugin.custom_function())
```

## Complete Example

Here's a complete example putting it all together:

```lua
-- Load torch7u
require 'init'

-- Configure
torch7u.configure({
    default_tensor_type = 'torch.FloatTensor'
})

-- Load integration bridges
local nn_optim = require 'integrations.nn_optim_bridge'

-- Create model
local model = nn.Sequential()
    :add(nn.Linear(784, 256))
    :add(nn.ReLU())
    :add(nn.Linear(256, 10))
    :add(nn.LogSoftMax())

-- Register model
torch7u.models.register('classifier', model)

-- Subscribe to events
torch7u.events.subscribe('epoch_end', function(epoch, loss)
    print(string.format('Epoch %d: loss = %.4f', epoch, loss))
    
    -- Save checkpoint every 10 epochs
    if epoch % 10 == 0 then
        torch7u.checkpoint.save(
            string.format('checkpoint_epoch_%d.t7', epoch),
            {model = model, epoch = epoch}
        )
    end
end)

-- Create training loop
local criterion = nn.ClassNLLCriterion()
local training_loop = nn_optim.create_training_loop(
    model,
    criterion,
    optim.adam,
    {learningRate = 0.001}
)

-- Create dataset (replace with real data)
local train_data = {
    data = torch.rand(1000, 784),
    labels = torch.random(1000, 10),
    size = function() return 1000 end
}

-- Train
local metrics = training_loop:fit(train_data, nil, 20, 32)

-- Display final metrics
print('Training completed!')
print('Final loss:', metrics.train_loss[#metrics.train_loss])

-- Get all recorded metrics
local all_losses = torch7u.metrics.get('epoch_loss')
print('Total epochs:', #all_losses)
```

## Running Examples

The repository includes comprehensive examples:

```bash
# Run integration test suite
th test_integration.lua

# Run full integration example
th example_integration.lua
```

## Tips and Best Practices

1. **Always use `require 'init'`** at the start of your scripts to load the integration layer

2. **Register models** in the model registry for easy access across your application

3. **Use events** for decoupled communication between components

4. **Record metrics** throughout your code for monitoring and debugging

5. **Create data pipelines** for consistent preprocessing

6. **Use integration bridges** for enhanced functionality between modules

7. **Save checkpoints** regularly using the integrated checkpointing system

8. **Subscribe to system events** to monitor training progress

9. **Leverage plugins** for reusable, cross-cutting functionality

10. **Check `torch7u.config`** for global settings

## Documentation

- **[INTEGRATION.md](INTEGRATION.md)** - Complete integration documentation
- **[example_integration.lua](example_integration.lua)** - Comprehensive example
- **Individual module READMEs** - Detailed documentation for each component

## Troubleshooting

### Module not found
```lua
local mod = torch7u.load_module('module_name')
if not mod then
    print('Module not available')
end
```

### Check loaded modules
```lua
for name, _ in pairs(torch7u.loaded_modules) do
    print(name)
end
```

### Enable debug logging
```lua
torch7u.configure({logging_level = 'DEBUG'})
```

### View configuration
```lua
for k, v in pairs(torch7u.config) do
    print(k, v)
end
```

## Next Steps

- Read [INTEGRATION.md](INTEGRATION.md) for detailed documentation
- Explore `example_integration.lua` for a comprehensive example
- Check integration bridges in `integrations/` for advanced features
- Run tests with `test_integration.lua` to verify setup

## Getting Help

- Review the integration documentation
- Check the example files
- Look at individual module documentation
- Examine the integration bridge source code for advanced usage

Happy coding with Torch7u!
