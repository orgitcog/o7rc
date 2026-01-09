-- ============================================================================
-- Torch7u Integration Example
-- ============================================================================
-- This example demonstrates the deep interconnection between all torch7
-- components through the torch7u integration layer.
-- ============================================================================

-- Load the unified torch7u system
require 'init'

print("\n" .. string.rep("=", 70))
print("Torch7u Deep Integration Example")
print(string.rep("=", 70) .. "\n")

-- ============================================================================
-- Step 1: Display System Information
-- ============================================================================

print("Step 1: System Information")
print(string.rep("-", 70))
torch7u.info()

-- ============================================================================
-- Step 2: Configure the System
-- ============================================================================

print("\nStep 2: System Configuration")
print(string.rep("-", 70))

torch7u.configure({
    default_tensor_type = 'torch.DoubleTensor',
    logging_level = 'INFO',
})

print("Configuration set:")
print("  Tensor type: " .. torch7u.config.default_tensor_type)
print("  Logging level: " .. torch7u.config.logging_level)
print("  CUDA available: " .. tostring(torch7u.cuda_available))

-- ============================================================================
-- Step 3: Event System Demonstration
-- ============================================================================

print("\nStep 3: Event System")
print(string.rep("-", 70))

-- Subscribe to events
torch7u.events.subscribe('model_created', function(name, model)
    print("  [EVENT] Model created: " .. name)
end, 'example')

torch7u.events.subscribe('training_milestone', function(epoch, loss)
    print(string.format("  [EVENT] Training milestone - Epoch: %d, Loss: %.4f", epoch, loss))
end, 'example')

-- ============================================================================
-- Step 4: Create and Register Models
-- ============================================================================

print("\nStep 4: Model Creation and Registry")
print(string.rep("-", 70))

-- Create a simple feedforward network
local simple_model = nn.Sequential()
    :add(nn.Linear(784, 256))
    :add(nn.ReLU())
    :add(nn.Dropout(0.5))
    :add(nn.Linear(256, 128))
    :add(nn.ReLU())
    :add(nn.Dropout(0.5))
    :add(nn.Linear(128, 10))
    :add(nn.LogSoftMax())

-- Register the model
torch7u.models.register('mnist_classifier', simple_model, {
    architecture = 'feedforward',
    input_size = 784,
    output_size = 10,
    task = 'classification',
    description = 'Simple MNIST classifier'
})

torch7u.events.publish('model_created', 'mnist_classifier', simple_model)

print("Model registered: mnist_classifier")
print("  Input: 784")
print("  Hidden: 256 -> 128")
print("  Output: 10")

-- Create a convolutional network
local conv_model = nn.Sequential()
    :add(nn.SpatialConvolution(1, 32, 5, 5))
    :add(nn.ReLU())
    :add(nn.SpatialMaxPooling(2, 2, 2, 2))
    :add(nn.SpatialConvolution(32, 64, 5, 5))
    :add(nn.ReLU())
    :add(nn.SpatialMaxPooling(2, 2, 2, 2))
    :add(nn.View(-1):setNumInputDims(3))
    :add(nn.Linear(64 * 4 * 4, 128))
    :add(nn.ReLU())
    :add(nn.Linear(128, 10))
    :add(nn.LogSoftMax())

torch7u.models.register('mnist_conv', conv_model, {
    architecture = 'convolutional',
    input_channels = 1,
    output_size = 10,
    task = 'classification'
})

torch7u.events.publish('model_created', 'mnist_conv', conv_model)

print("\nModels in registry:")
for _, name in ipairs(torch7u.models.list()) do
    print("  - " .. name)
end

-- ============================================================================
-- Step 5: Data Pipeline Integration
-- ============================================================================

print("\nStep 5: Data Pipeline")
print(string.rep("-", 70))

-- Create unified data pipeline
local pipeline = torch7u.data.create_pipeline()

-- Add preprocessing transforms
pipeline:add_transform('flatten', function(data)
    if data:nDimension() == 3 then
        return data:view(-1)
    end
    return data
end)

pipeline:add_transform('normalize', function(data)
    local mean = data:mean()
    local std = data:std()
    return (data - mean) / (std + 1e-7)
end)

pipeline:add_transform('add_noise', function(data)
    return data + torch.randn(data:size()) * 0.01
end)

print("Data pipeline created with transforms:")
print("  - flatten")
print("  - normalize")
print("  - add_noise")

-- Test pipeline
local sample_data = torch.rand(28, 28)
local processed = pipeline:process(sample_data, {'flatten', 'normalize'})
print(string.format("\nSample data: %dx%d -> Processed: %d", 
    sample_data:size(1), sample_data:size(2), processed:size(1)))

-- ============================================================================
-- Step 6: nn-optim Integration Bridge
-- ============================================================================

print("\nStep 6: nn-optim Integration Bridge")
print(string.rep("-", 70))

-- Load integration bridge
local nn_optim_bridge = require 'integrations.nn_optim_bridge'

-- Get a model
local model = torch7u.models.get('mnist_classifier')

-- Create optimizer using integrated method
local optimizer = model:createOptimizer(optim.adam, {
    learningRate = 0.001,
    beta1 = 0.9,
    beta2 = 0.999
})

print("Optimizer created for model:")
print("  Type: Adam")
print("  Learning rate: 0.001")

-- Create learning rate scheduler
local scheduler = model:createLRScheduler(0.001, 'exponential', {gamma = 0.95})

print("\nLearning rate scheduler created:")
print("  Type: exponential")
print("  Initial LR: 0.001")
print("  Decay: 0.95")

-- Simulate scheduler steps
print("\nLR Schedule preview:")
for epoch = 1, 5 do
    local lr = scheduler:step(epoch)
    print(string.format("  Epoch %d: LR = %.6f", epoch, lr))
end
scheduler:reset()

-- ============================================================================
-- Step 7: Metrics and Monitoring
-- ============================================================================

print("\nStep 7: Metrics and Monitoring")
print(string.rep("-", 70))

-- Subscribe to metric events
torch7u.events.subscribe('metric_recorded', function(name, value, tags)
    if name == 'training_loss' then
        -- Could trigger actions based on metrics
    end
end, 'example')

-- Simulate recording metrics
print("Recording simulated training metrics...")
for epoch = 1, 5 do
    local train_loss = 2.0 * math.exp(-0.3 * epoch) + math.random() * 0.1
    local val_loss = 2.0 * math.exp(-0.25 * epoch) + math.random() * 0.1
    
    torch7u.metrics.record('training_loss', train_loss, {epoch = epoch})
    torch7u.metrics.record('validation_loss', val_loss, {epoch = epoch})
    
    torch7u.events.publish('training_milestone', epoch, train_loss)
end

-- Display metrics
local train_losses = torch7u.metrics.get('training_loss')
print(string.format("\nRecorded %d training loss values", #train_losses))
print("Last 3 training losses:")
for i = math.max(1, #train_losses - 2), #train_losses do
    print(string.format("  Epoch %d: %.4f", i, train_losses[i].value))
end

-- ============================================================================
-- Step 8: Unified Training Interface Demo
-- ============================================================================

print("\nStep 8: Unified Training Interface")
print(string.rep("-", 70))

-- Create training loop using integrated bridge
local criterion = nn.ClassNLLCriterion()

local training_loop = nn_optim_bridge.create_training_loop(
    model,
    criterion,
    optim.sgd,
    {learningRate = 0.01, momentum = 0.9}
)

print("Training loop created:")
print("  Model: mnist_classifier")
print("  Criterion: ClassNLLCriterion")
print("  Optimizer: SGD")
print("  Learning rate: 0.01")
print("  Momentum: 0.9")

-- ============================================================================
-- Step 9: Checkpointing Integration
-- ============================================================================

print("\nStep 9: Checkpointing")
print(string.rep("-", 70))

-- Subscribe to checkpoint events
torch7u.events.subscribe('checkpoint_saved', function(filename, checkpoint)
    print("  [EVENT] Checkpoint saved: " .. filename)
end, 'example')

-- Save a checkpoint
local checkpoint_file = '/tmp/torch7u_example_checkpoint.t7'
torch7u.checkpoint.save(checkpoint_file, {
    model = model,
    optimizer_state = optimizer.state,
    epoch = 10,
    best_loss = 0.234
}, {
    description = 'Example checkpoint',
    training_completed = false
})

print("Checkpoint saved:")
print("  File: " .. checkpoint_file)

-- Load checkpoint
local loaded_data, metadata = torch7u.checkpoint.load(checkpoint_file)
print("\nCheckpoint loaded:")
print("  Epoch: " .. loaded_data.epoch)
print("  Best loss: " .. loaded_data.best_loss)
print("  Description: " .. metadata.description)

-- Cleanup
os.remove(checkpoint_file)

-- ============================================================================
-- Step 10: Plugin System
-- ============================================================================

print("\nStep 10: Plugin System")
print(string.rep("-", 70))

-- Create a custom plugin
local visualization_plugin = {
    name = "visualization",
    
    init = function(torch7u)
        print("  Visualization plugin initialized")
        
        -- Subscribe to training events
        torch7u.events.subscribe('training_milestone', function(epoch, loss)
            -- In real scenario, this would create plots
            print(string.format("    [VIZ] Plotting loss at epoch %d: %.4f", epoch, loss))
        end, 'visualization_plugin')
    end,
    
    plot_loss = function(losses)
        print("  Plotting loss curve...")
        return true
    end,
    
    visualize_model = function(model)
        print("  Visualizing model architecture...")
        return true
    end
}

-- Register plugin
torch7u.plugins.register('visualization', visualization_plugin)

print("\nPlugin registered: visualization")
print("Available plugins:")
for _, name in ipairs(torch7u.plugins.list()) do
    print("  - " .. name)
end

-- Use plugin
local viz = torch7u.plugins.get('visualization')
viz.plot_loss({1.0, 0.8, 0.6, 0.4})

-- ============================================================================
-- Step 11: Cross-Module Interconnection Demo
-- ============================================================================

print("\nStep 11: Cross-Module Interconnections")
print(string.rep("-", 70))

print("\nIntegrated features available:")
print("  ✓ nn.Module extended with optimizer creation")
print("  ✓ Automatic parameter extraction for optim")
print("  ✓ Unified training interface across all modules")
print("  ✓ Event-based communication between components")
print("  ✓ Shared configuration across all modules")
print("  ✓ Centralized model registry")
print("  ✓ Unified metrics collection")
print("  ✓ Cross-module data pipelines")
print("  ✓ Plugin system for extensibility")

if torch7u.cuda_available then
    print("  ✓ Automatic GPU transfer support")
end

-- ============================================================================
-- Summary
-- ============================================================================

print("\n" .. string.rep("=", 70))
print("Integration Summary")
print(string.rep("=", 70))

print("\nModules loaded: " .. #torch7u.models.list() .. " models registered")
print("Events published: Multiple training and system events")
print("Metrics recorded: Training and validation losses")
print("Plugins active: " .. #torch7u.plugins.list())

print("\nThe torch7u integration layer provides:")
print("  1. Seamless interoperability between all torch modules")
print("  2. Unified interfaces for common tasks")
print("  3. Event-driven architecture for loose coupling")
print("  4. Centralized configuration and logging")
print("  5. Extensible plugin system")
print("  6. Deep integration bridges between key modules")

print("\n" .. string.rep("=", 70))
print("Example completed successfully!")
print(string.rep("=", 70) .. "\n")
