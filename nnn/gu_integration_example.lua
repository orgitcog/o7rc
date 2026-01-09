-- ============================================================================
-- NNN-GU Integration Examples
-- ============================================================================
-- Comprehensive examples demonstrating how to use nested neural networks
-- with Geometric Unity (ObserverseTensor) structures.
-- ============================================================================

require 'nn'
local nnn = require 'nnn'
local gu = require 'gu'

print("=======================================================")
print("NNN-GU Integration Examples")
print("=======================================================\n")

-- ============================================================================
-- Example 1: Basic GU-Aware Transform
-- ============================================================================
print("Example 1: Basic GU-Aware Transform")
print("-----------------------------------")

-- Create a simple nn module and wrap it for GU support
local linear = nn.Linear(10, 10)
local guLinear = nnn.transformGU(linear, {applyTo = 'fiber'})

-- Create an ObserverseTensor
local obs = gu.randomObserverse(4)  -- batch of 4
print("Input ObserverseTensor:")
print(obs)

-- Forward pass - applies linear only to fiber component
local output = guLinear:forward(obs)
print("\nOutput ObserverseTensor (linear applied to fiber):")
print(output)
print()

-- ============================================================================
-- Example 2: Nested ObserverseTensors
-- ============================================================================
print("\nExample 2: Nested ObserverseTensors")
print("------------------------------------")

-- Create a nested structure of ObserverseTensors
local nested_input = {
    left = gu.randomObserverse(2),
    right = {
        a = gu.randomObserverse(2),
        b = gu.randomObserverse(2)
    }
}

print("Nested structure depth:", nnn.gu.depth(nested_input))
print("Number of ObserverseTensors:", nnn.gu.count(nested_input))

-- Apply ReLU to all fibers in the nested structure
local guRelu = nnn.GUReLU({applyTo = 'fiber'})
local nested_output = guRelu:forward(nested_input)

print("\nNested output structure preserved!")
print("Output left fiber size:", nested_output.left.fiber:size())
print()

-- ============================================================================
-- Example 3: Building a GU Sequential Model
-- ============================================================================
print("\nExample 3: Building a GU Sequential Model")
print("------------------------------------------")

-- Create a sequential model for nested ObserverseTensors
local model = nnn.GUSequential()
model:add(nnn.GULinear(10, 8, {applyTo = 'fiber'}))
model:add(nnn.GUReLU({applyTo = 'fiber'}))
model:add(nnn.GULinear(8, 10, {applyTo = 'fiber'}))
model:add(nnn.GUTanh({applyTo = 'fiber'}))

print("Model architecture:")
print(model)

-- Forward pass with nested input
local model_output = model:forward(nested_input)
print("\nModel output structure preserved!")
print()

-- ============================================================================
-- Example 4: Using nnn.GULayer (Full GU Dynamics)
-- ============================================================================
print("\nExample 4: Using nnn.GULayer (Full GU Dynamics)")
print("------------------------------------------------")

-- Create a nested GU layer with Swerve and Gauge transformations
local guLayer = nnn.GULayer(10, {
    use_swerve = true,
    use_gauge = true,
    gauge_type = 'tilted',
    activation = 'tanh',
    use_residual = true
})

print("GU Layer:", guLayer)

-- Works with nested ObserverseTensors
local gu_output = guLayer:forward(nested_input)
print("GU Layer processed nested structure!")
print()

-- ============================================================================
-- Example 5: GU-Specific Criteria
-- ============================================================================
print("\nExample 5: GU-Specific Criteria")
print("--------------------------------")

-- Create input and target ObserverseTensors
local input_obs = gu.randomObserverse(4)
local target_obs = gu.randomObserverse(4)

-- Standard GU MSE Criterion
local criterion = nnn.GUMSECriterion({applyTo = 'both'})
local loss = criterion:forward(input_obs, target_obs)
print("GU MSE Loss (both components):", loss)

-- Fiber-only loss
local fiber_criterion = nnn.GUMSECriterion({applyTo = 'fiber'})
local fiber_loss = fiber_criterion:forward(input_obs, target_obs)
print("GU MSE Loss (fiber only):", fiber_loss)

-- Weighted Observerse MSE
local weighted_criterion = nnn.ObserverseMSECriterion({
    baseWeight = 0.3,
    fiberWeight = 0.7
})
local weighted_loss = weighted_criterion:forward(input_obs, target_obs)
print("Weighted Observerse MSE Loss:", weighted_loss)
print()

-- ============================================================================
-- Example 6: Nested Criteria
-- ============================================================================
print("\nExample 6: Nested Criteria")
print("--------------------------")

-- Create nested input and target
local nested_in = {
    gu.randomObserverse(2),
    {gu.randomObserverse(2), gu.randomObserverse(2)}
}
local nested_target = {
    gu.randomObserverse(2),
    {gu.randomObserverse(2), gu.randomObserverse(2)}
}

-- Criterion with different aggregation modes
local mean_criterion = nnn.GUMSECriterion({aggregation = 'mean'})
local sum_criterion = nnn.GUMSECriterion({aggregation = 'sum'})

print("Nested MSE (mean aggregation):", mean_criterion:forward(nested_in, nested_target))
print("Nested MSE (sum aggregation):", sum_criterion:forward(nested_in, nested_target))
print()

-- ============================================================================
-- Example 7: Training Loop with NNN-GU
-- ============================================================================
print("\nExample 7: Training Loop with NNN-GU")
print("-------------------------------------")

-- Create a simple model
local train_model = nnn.GUSequential()
train_model:add(nnn.GULinear(10, 10, {applyTo = 'fiber'}))
train_model:add(nnn.GUReLU({applyTo = 'fiber'}))

local train_criterion = nnn.GUMSECriterion({applyTo = 'fiber'})
local learning_rate = 0.01

-- Training data
local train_input = gu.randomObserverse(8)
local train_target = gu.randomObserverse(8)

-- Get parameters
local params, gradParams = train_model:getParameters()

print("Initial loss:", train_criterion:forward(train_model:forward(train_input), train_target))

-- Training loop
for epoch = 1, 100 do
    -- Zero gradients
    gradParams:zero()

    -- Forward pass
    local output = train_model:forward(train_input)
    local loss = train_criterion:forward(output, train_target)

    -- Backward pass
    local gradOutput = train_criterion:backward(output, train_target)
    train_model:backward(train_input, gradOutput)

    -- Update parameters
    params:add(-learning_rate, gradParams)

    if epoch % 25 == 0 then
        print(string.format("Epoch %d, Loss: %.6f", epoch, loss))
    end
end

print()

-- ============================================================================
-- Example 8: Utility Functions
-- ============================================================================
print("\nExample 8: Utility Functions")
print("----------------------------")

-- Create a complex nested structure
local complex_nested = {
    level1 = {
        a = gu.randomObserverse(2),
        b = gu.randomObserverse(2)
    },
    level2 = gu.randomObserverse(2)
}

-- Flatten to list
local flattened = nnn.gu.flatten(complex_nested)
print("Flattened list length:", #flattened)

-- Count ObserverseTensors
print("Total ObserverseTensors:", nnn.gu.count(complex_nested))

-- Get depth
print("Nesting depth:", nnn.gu.depth(complex_nested))

-- Map a function over all ObserverseTensors
local mapped = nnn.gu.map(complex_nested, function(obs)
    -- Scale fiber by 2
    return gu.ObserverseTensor.create(obs.base, obs.fiber * 2)
end)
print("Mapped structure preserved!")

-- Clone the structure
local cloned = nnn.gu.clone(complex_nested)
print("Cloned structure!")

print()

-- ============================================================================
-- Example 9: Random Nested Structure Generation
-- ============================================================================
print("\nExample 9: Random Nested Structure Generation")
print("----------------------------------------------")

-- Define a structure template
local template = {
    branch1 = 1,  -- Single ObserverseTensor
    branch2 = {
        sub1 = 1,
        sub2 = 1
    }
}

-- Generate random nested ObserverseTensors following the template
local random_nested = nnn.gu.randomNested(template, 4)  -- batch size 4
print("Generated nested structure with batch size 4")
print("Branch1 base shape:", random_nested.branch1.base:size())
print("Branch2.sub1 fiber shape:", random_nested.branch2.sub1.fiber:size())
print()

-- ============================================================================
-- Example 10: Complete End-to-End Pipeline
-- ============================================================================
print("\nExample 10: Complete End-to-End Pipeline")
print("-----------------------------------------")

-- Build a complete GU neural network
local gu_network = nnn.GUSequential()

-- Input projection
gu_network:add(nnn.GULinear(10, 20, {applyTo = 'fiber'}))
gu_network:add(nnn.GUReLU({applyTo = 'fiber'}))
gu_network:add(nnn.GUDropout(0.1, {applyTo = 'fiber'}))

-- Hidden layers
gu_network:add(nnn.GULinear(20, 20, {applyTo = 'fiber'}))
gu_network:add(nnn.GUTanh({applyTo = 'fiber'}))

-- Output projection
gu_network:add(nnn.GULinear(20, 10, {applyTo = 'fiber'}))

print("Complete GU Network:")
print(gu_network)

-- Create nested training data
local pipeline_input = {
    sample1 = gu.randomObserverse(4),
    sample2 = {
        a = gu.randomObserverse(4),
        b = gu.randomObserverse(4)
    }
}
local pipeline_target = nnn.gu.clone(pipeline_input)

-- Criterion with weighted components
local pipeline_criterion = nnn.ObserverseMSECriterion({
    baseWeight = 0.2,
    fiberWeight = 0.8
})

-- Forward pass through entire nested structure
gu_network:training()
local pipeline_output = gu_network:forward(pipeline_input)
local pipeline_loss = pipeline_criterion:forward(pipeline_output, pipeline_target)

print(string.format("\nPipeline loss: %.6f", pipeline_loss))
print("Nested structure processed successfully!")

-- ============================================================================
-- Display Integration Info
-- ============================================================================
print("\n")
nnn.gu.info()

print("\n=======================================================")
print("All examples completed successfully!")
print("=======================================================")
