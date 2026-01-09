-- nnn: Nested Neural Nets Functional Operator System
-- Transforms nn.* operations to work with nested tensors (nestors)
-- Similar to how nn.* transforms * with tensor embeddings,
-- nnn.* transforms * with nestor (nested tensor) metagraph embeddings

local nnn = {}

-- Import utilities from nn
nnn.NestedTensor = require('nn.NestedTensor')
nnn.PrimeFactorType = require('nn.PrimeFactorType')

-- Module registry for transformed modules
nnn._registry = {}

-- Core transformation function: wraps any nn module to work with nested tensors
function nnn.transform(module, config)
    config = config or {}
    
    local NestedOperator, parent = torch.class('nnn.NestedOperator', 'nn.Module')
    
    function NestedOperator:__init(wrappedModule)
        parent.__init(self)
        self.module = wrappedModule
        self.maxDepth = config.maxDepth or 10
        self.aggregation = config.aggregation or 'preserve'  -- 'preserve', 'flatten', 'mean'
    end
    
    -- Process nested structure recursively
    function NestedOperator:processNested(input, depth)
        depth = depth or 0
        
        if depth > self.maxDepth then
            error(string.format('Nesting depth %d exceeds maximum depth %d', depth, self.maxDepth))
        end
        
        -- Base case: tensor input
        if torch.isTensor(input) then
            return self.module:forward(input)
        end
        
        -- Recursive case: nested table structure
        if type(input) == 'table' then
            local outputs = {}
            for k, v in pairs(input) do
                outputs[k] = self:processNested(v, depth + 1)
            end
            return outputs
        end
        
        error("input must be a tensor or table")
    end
    
    function NestedOperator:updateOutput(input)
        self.output = self:processNested(input, 0)
        return self.output
    end
    
    -- Backward pass through nested structure
    function NestedOperator:processNestedBackward(input, gradOutput, depth)
        depth = depth or 0
        
        -- Base case: tensor input
        if torch.isTensor(input) then
            return self.module:backward(input, gradOutput)
        end
        
        -- Recursive case: nested table structure
        if type(input) == 'table' then
            local gradInputs = {}
            for k, v in pairs(input) do
                gradInputs[k] = self:processNestedBackward(v, gradOutput[k], depth + 1)
            end
            return gradInputs
        end
        
        error("input must be a tensor or table")
    end
    
    function NestedOperator:updateGradInput(input, gradOutput)
        self.gradInput = self:processNestedBackward(input, gradOutput, 0)
        return self.gradInput
    end
    
    function NestedOperator:accGradParameters(input, gradOutput, scale)
        -- The wrapped module's parameters are updated directly
        -- We just need to ensure backward is called with correct inputs
        local function accumulate(inp, gradOut, depth)
            depth = depth or 0
            
            if torch.isTensor(inp) then
                self.module:accGradParameters(inp, gradOut, scale)
            elseif type(inp) == 'table' then
                for k, v in pairs(inp) do
                    accumulate(v, gradOut[k], depth + 1)
                end
            end
        end
        
        accumulate(input, gradOutput, 0)
    end
    
    function NestedOperator:type(type, tensorCache)
        parent.type(self, type, tensorCache)
        self.module:type(type, tensorCache)
        return self
    end
    
    function NestedOperator:clearState()
        self.module:clearState()
        return parent.clearState(self)
    end
    
    function NestedOperator:reset(stdv)
        if self.module.reset then
            self.module:reset(stdv)
        end
    end
    
    return NestedOperator(module)
end

-- Convenience wrapper for common nn modules
function nnn.wrap(module)
    return nnn.transform(module)
end

-- Create nnn versions of common nn modules
-- These maintain compatibility while adding nested tensor support

-- nnn.Sequential: extends nn.Sequential for nested tensors
function nnn.Sequential()
    local wrapper = nnn.transform(nn.Sequential())
    wrapper.__typename = 'nnn.Sequential'
    return wrapper
end

-- nnn.Linear: extends nn.Linear for nested tensors
function nnn.Linear(inputSize, outputSize, bias)
    local wrapper = nnn.transform(nn.Linear(inputSize, outputSize, bias))
    wrapper.__typename = 'nnn.Linear'
    return wrapper
end

-- nnn.Tanh: extends nn.Tanh for nested tensors
function nnn.Tanh()
    local wrapper = nnn.transform(nn.Tanh())
    wrapper.__typename = 'nnn.Tanh'
    return wrapper
end

-- nnn.ReLU: extends nn.ReLU for nested tensors
function nnn.ReLU(inplace)
    local wrapper = nnn.transform(nn.ReLU(inplace))
    wrapper.__typename = 'nnn.ReLU'
    return wrapper
end

-- nnn.Sigmoid: extends nn.Sigmoid for nested tensors
function nnn.Sigmoid()
    local wrapper = nnn.transform(nn.Sigmoid())
    wrapper.__typename = 'nnn.Sigmoid'
    return wrapper
end

-- nnn.SoftMax: extends nn.SoftMax for nested tensors
function nnn.SoftMax()
    local wrapper = nnn.transform(nn.SoftMax())
    wrapper.__typename = 'nnn.SoftMax'
    return wrapper
end

-- Modal Classifier: extends any criterion to work with nested tensors
local NestedCriterion, criterionParent = torch.class('nnn.NestedCriterion', 'nn.Criterion')

function NestedCriterion:__init(criterion)
    criterionParent.__init(self)
    self.criterion = criterion
    self.maxDepth = 10
end

-- Process nested structure for criterion
function NestedCriterion:processNested(input, target, depth)
    depth = depth or 0
    
    if depth > self.maxDepth then
        error(string.format('Nesting depth %d exceeds maximum depth %d', depth, self.maxDepth))
    end
    
    -- Base case: tensor input
    if torch.isTensor(input) and torch.isTensor(target) then
        return self.criterion:forward(input, target)
    end
    
    -- Recursive case: nested table structure
    if type(input) == 'table' and type(target) == 'table' then
        local totalLoss = 0
        local count = 0
        for k, v in pairs(input) do
            totalLoss = totalLoss + self:processNested(v, target[k], depth + 1)
            count = count + 1
        end
        -- Average loss across branches
        return totalLoss / count
    end
    
    error("input and target must both be tensors or tables")
end

function NestedCriterion:updateOutput(input, target)
    self.output = self:processNested(input, target, 0)
    return self.output
end

-- Backward pass for nested criterion
function NestedCriterion:processNestedGrad(input, target, depth)
    depth = depth or 0
    
    -- Base case: tensor input
    if torch.isTensor(input) and torch.isTensor(target) then
        return self.criterion:backward(input, target)
    end
    
    -- Recursive case: nested table structure
    if type(input) == 'table' and type(target) == 'table' then
        local gradInputs = {}
        local count = 0
        for k in pairs(input) do
            count = count + 1
        end
        
        for k, v in pairs(input) do
            gradInputs[k] = self:processNestedGrad(v, target[k], depth + 1)
            -- Scale gradient by number of branches
            if torch.isTensor(gradInputs[k]) then
                gradInputs[k]:div(count)
            end
        end
        return gradInputs
    end
    
    error("input and target must both be tensors or tables")
end

function NestedCriterion:updateGradInput(input, target)
    self.gradInput = self:processNestedGrad(input, target, 0)
    return self.gradInput
end

-- Factory function for nnn.Criterion
function nnn.Criterion(criterion)
    return nnn.NestedCriterion(criterion)
end

-- Specific criterion wrappers
function nnn.MSECriterion()
    return nnn.NestedCriterion(nn.MSECriterion())
end

function nnn.ClassNLLCriterion(weights)
    return nnn.NestedCriterion(nn.ClassNLLCriterion(weights))
end

function nnn.BCECriterion(weights)
    return nnn.NestedCriterion(nn.BCECriterion(weights))
end

function nnn.CrossEntropyCriterion(weights)
    return nnn.NestedCriterion(nn.CrossEntropyCriterion(weights))
end

-- Embedding modules from nn
nnn.NestedEmbedding = require('nn.NestedEmbedding')
nnn.NestedNeuralNet = require('nn.NestedNeuralNet')

-- Factory function for creating nnn versions of any nn module
function nnn.fromNN(moduleName, ...)
    local nnModule = nn[moduleName]
    if not nnModule then
        error(string.format("nn.%s does not exist", moduleName))
    end
    
    local instance = nnModule(...)
    return nnn.transform(instance)
end

-- Helper to check if input is nested
function nnn.isNested(input)
    return type(input) == 'table' and not torch.isTensor(input)
end

-- Get depth of nested structure
function nnn.depth(input)
    return nnn.NestedTensor.depth(input)
end

-- Flatten nested structure
function nnn.flatten(input)
    return nnn.NestedTensor.flatten(input)
end

-- Clone nested structure
function nnn.clone(input)
    return nnn.NestedTensor.clone(input)
end

-- Apply function to all tensors in nested structure
function nnn.map(input, func)
    return nnn.NestedTensor.map(input, func)
end

-- Module version info
nnn._VERSION = '1.1.0'
nnn._DESCRIPTION = 'Nested Neural Nets (NNN) - Functional operators for nested tensor operations with GU integration'

-- Future extensions placeholder
nnn.operad = {}
nnn.operad._FUTURE = 'Operad gadgets indexed by prime factorizations as orbifold symmetries'

-- ============================================================================
-- GU (Geometric Unity) Integration
-- ============================================================================
-- Extends NNN to natively support ObserverseTensor structures from the GU module.
-- This creates a unified framework for nested geometric neural networks.
-- ============================================================================

nnn.gu = {}

-- Check if GU module is available
local function getGU()
    local ok, gu = pcall(require, 'gu')
    if ok then return gu end
    return nil
end

-- Check if input is an ObserverseTensor
function nnn.gu.isObserverse(input)
    return type(input) == 'table' and input._type == 'ObserverseTensor'
end

-- ============================================================================
-- GU-Aware Transform Function
-- ============================================================================
-- Enhanced transform that handles both nested tensors AND ObserverseTensors.
-- This is the core integration point between NNN and GU.
-- ============================================================================

function nnn.transformGU(module, config)
    config = config or {}
    local gu = getGU()

    local GUNestedOperator, parent = torch.class('nnn.GUNestedOperator', 'nn.Module')

    function GUNestedOperator:__init(wrappedModule)
        parent.__init(self)
        self.module = wrappedModule
        self.maxDepth = config.maxDepth or 10
        self.aggregation = config.aggregation or 'preserve'
        self.applyTo = config.applyTo or 'both'  -- 'base', 'fiber', or 'both'
        self.preserveGU = config.preserveGU ~= false  -- Preserve ObserverseTensor structure
    end

    -- Process nested structure with GU awareness
    function GUNestedOperator:processNested(input, depth)
        depth = depth or 0

        if depth > self.maxDepth then
            error(string.format('Nesting depth %d exceeds maximum depth %d', depth, self.maxDepth))
        end

        -- Case 1: ObserverseTensor (GU structure)
        if nnn.gu.isObserverse(input) then
            local base_out = input.base
            local fiber_out = input.fiber

            if self.applyTo == 'base' or self.applyTo == 'both' then
                base_out = self.module:forward(input.base:clone())
            end
            if self.applyTo == 'fiber' or self.applyTo == 'both' then
                fiber_out = self.module:forward(input.fiber:clone())
            end

            if self.preserveGU and gu then
                return gu.ObserverseTensor.create(base_out, fiber_out)
            else
                return {base = base_out, fiber = fiber_out, _type = 'ObserverseTensor'}
            end
        end

        -- Case 2: Regular tensor
        if torch.isTensor(input) then
            return self.module:forward(input)
        end

        -- Case 3: Nested table structure
        if type(input) == 'table' then
            local outputs = {}
            for k, v in pairs(input) do
                outputs[k] = self:processNested(v, depth + 1)
            end
            return outputs
        end

        error("input must be a tensor, ObserverseTensor, or table")
    end

    function GUNestedOperator:updateOutput(input)
        self.output = self:processNested(input, 0)
        return self.output
    end

    -- Backward pass with GU awareness
    function GUNestedOperator:processNestedBackward(input, gradOutput, depth)
        depth = depth or 0

        -- Case 1: ObserverseTensor
        if nnn.gu.isObserverse(input) then
            local base_grad = gradOutput.base
            local fiber_grad = gradOutput.fiber

            if self.applyTo == 'base' or self.applyTo == 'both' then
                base_grad = self.module:backward(input.base, gradOutput.base)
            end
            if self.applyTo == 'fiber' or self.applyTo == 'both' then
                fiber_grad = self.module:backward(input.fiber, gradOutput.fiber)
            end

            if self.preserveGU and gu then
                return gu.ObserverseTensor.create(base_grad, fiber_grad)
            else
                return {base = base_grad, fiber = fiber_grad, _type = 'ObserverseTensor'}
            end
        end

        -- Case 2: Regular tensor
        if torch.isTensor(input) then
            return self.module:backward(input, gradOutput)
        end

        -- Case 3: Nested table structure
        if type(input) == 'table' then
            local gradInputs = {}
            for k, v in pairs(input) do
                gradInputs[k] = self:processNestedBackward(v, gradOutput[k], depth + 1)
            end
            return gradInputs
        end

        error("input must be a tensor, ObserverseTensor, or table")
    end

    function GUNestedOperator:updateGradInput(input, gradOutput)
        self.gradInput = self:processNestedBackward(input, gradOutput, 0)
        return self.gradInput
    end

    function GUNestedOperator:accGradParameters(input, gradOutput, scale)
        local function accumulate(inp, gradOut, depth)
            depth = depth or 0

            if nnn.gu.isObserverse(inp) then
                if self.applyTo == 'base' or self.applyTo == 'both' then
                    self.module:accGradParameters(inp.base, gradOut.base, scale)
                end
                if self.applyTo == 'fiber' or self.applyTo == 'both' then
                    self.module:accGradParameters(inp.fiber, gradOut.fiber, scale)
                end
            elseif torch.isTensor(inp) then
                self.module:accGradParameters(inp, gradOut, scale)
            elseif type(inp) == 'table' then
                for k, v in pairs(inp) do
                    accumulate(v, gradOut[k], depth + 1)
                end
            end
        end

        accumulate(input, gradOutput, 0)
    end

    function GUNestedOperator:type(type_str, tensorCache)
        parent.type(self, type_str, tensorCache)
        self.module:type(type_str, tensorCache)
        return self
    end

    function GUNestedOperator:clearState()
        self.module:clearState()
        return parent.clearState(self)
    end

    function GUNestedOperator:reset(stdv)
        if self.module.reset then
            self.module:reset(stdv)
        end
    end

    function GUNestedOperator:__tostring()
        return string.format('nnn.GUNestedOperator(%s, applyTo=%s)',
            torch.type(self.module), self.applyTo)
    end

    return GUNestedOperator(module)
end

-- ============================================================================
-- Hybrid GU Modules
-- ============================================================================
-- Pre-built modules that natively work with nested ObserverseTensors.
-- ============================================================================

-- nnn.GULinear: Linear layer for nested ObserverseTensors
function nnn.GULinear(inputSize, outputSize, config)
    config = config or {}
    local applyTo = config.applyTo or 'fiber'  -- Default to fiber for GU

    local wrapper = nnn.transformGU(nn.Linear(inputSize, outputSize), {
        applyTo = applyTo,
        preserveGU = true
    })
    wrapper.__typename = 'nnn.GULinear'
    return wrapper
end

-- nnn.GUReLU: ReLU activation for nested ObserverseTensors
function nnn.GUReLU(config)
    config = config or {}
    local applyTo = config.applyTo or 'fiber'

    local wrapper = nnn.transformGU(nn.ReLU(), {
        applyTo = applyTo,
        preserveGU = true
    })
    wrapper.__typename = 'nnn.GUReLU'
    return wrapper
end

-- nnn.GUTanh: Tanh activation for nested ObserverseTensors
function nnn.GUTanh(config)
    config = config or {}
    local applyTo = config.applyTo or 'fiber'

    local wrapper = nnn.transformGU(nn.Tanh(), {
        applyTo = applyTo,
        preserveGU = true
    })
    wrapper.__typename = 'nnn.GUTanh'
    return wrapper
end

-- nnn.GUSigmoid: Sigmoid activation for nested ObserverseTensors
function nnn.GUSigmoid(config)
    config = config or {}
    local applyTo = config.applyTo or 'fiber'

    local wrapper = nnn.transformGU(nn.Sigmoid(), {
        applyTo = applyTo,
        preserveGU = true
    })
    wrapper.__typename = 'nnn.GUSigmoid'
    return wrapper
end

-- nnn.GUSoftMax: SoftMax for nested ObserverseTensors
function nnn.GUSoftMax(config)
    config = config or {}
    local applyTo = config.applyTo or 'fiber'

    local wrapper = nnn.transformGU(nn.SoftMax(), {
        applyTo = applyTo,
        preserveGU = true
    })
    wrapper.__typename = 'nnn.GUSoftMax'
    return wrapper
end

-- nnn.GUDropout: Dropout for nested ObserverseTensors
function nnn.GUDropout(p, config)
    config = config or {}
    local applyTo = config.applyTo or 'fiber'

    local wrapper = nnn.transformGU(nn.Dropout(p), {
        applyTo = applyTo,
        preserveGU = true
    })
    wrapper.__typename = 'nnn.GUDropout'
    return wrapper
end

-- nnn.GUBatchNormalization: BatchNorm for nested ObserverseTensors
function nnn.GUBatchNormalization(nFeatures, config)
    config = config or {}
    local applyTo = config.applyTo or 'fiber'

    local wrapper = nnn.transformGU(nn.BatchNormalization(nFeatures), {
        applyTo = applyTo,
        preserveGU = true
    })
    wrapper.__typename = 'nnn.GUBatchNormalization'
    return wrapper
end

-- ============================================================================
-- nnn.GULayer: Full GU Layer wrapped for nested tensor support
-- ============================================================================
-- Wraps gu.GULayer to work with nested structures of ObserverseTensors.
-- ============================================================================

function nnn.GULayer(fiber_dim, config)
    local gu = getGU()
    if not gu then
        error("GU module not available. Please require 'gu' first.")
    end

    config = config or {}
    local guLayer = gu.GULayer(fiber_dim, config)

    -- Create nested wrapper
    local NNNGULayer, parent = torch.class('nnn.NNNGULayer', 'nn.Module')

    function NNNGULayer:__init(layer)
        parent.__init(self)
        self.layer = layer
        self.maxDepth = config.maxDepth or 10
    end

    function NNNGULayer:processNested(input, depth)
        depth = depth or 0

        if depth > self.maxDepth then
            error(string.format('Nesting depth %d exceeds maximum depth %d', depth, self.maxDepth))
        end

        -- ObserverseTensor: apply GU layer directly
        if nnn.gu.isObserverse(input) then
            return self.layer:forward(input)
        end

        -- Nested table: recurse
        if type(input) == 'table' then
            local outputs = {}
            for k, v in pairs(input) do
                outputs[k] = self:processNested(v, depth + 1)
            end
            return outputs
        end

        error("nnn.GULayer expects ObserverseTensor or nested table of ObserverseTensors")
    end

    function NNNGULayer:updateOutput(input)
        self.output = self:processNested(input, 0)
        return self.output
    end

    function NNNGULayer:processNestedBackward(input, gradOutput, depth)
        depth = depth or 0

        if nnn.gu.isObserverse(input) then
            return self.layer:backward(input, gradOutput)
        end

        if type(input) == 'table' then
            local gradInputs = {}
            for k, v in pairs(input) do
                gradInputs[k] = self:processNestedBackward(v, gradOutput[k], depth + 1)
            end
            return gradInputs
        end

        error("Expected ObserverseTensor or nested table")
    end

    function NNNGULayer:updateGradInput(input, gradOutput)
        self.gradInput = self:processNestedBackward(input, gradOutput, 0)
        return self.gradInput
    end

    function NNNGULayer:accGradParameters(input, gradOutput, scale)
        local function accumulate(inp, gradOut)
            if nnn.gu.isObserverse(inp) then
                self.layer:accGradParameters(inp, gradOut, scale)
            elseif type(inp) == 'table' then
                for k, v in pairs(inp) do
                    accumulate(v, gradOut[k])
                end
            end
        end
        accumulate(input, gradOutput)
    end

    function NNNGULayer:parameters()
        return self.layer:parameters()
    end

    function NNNGULayer:training()
        self.layer:training()
    end

    function NNNGULayer:evaluate()
        self.layer:evaluate()
    end

    function NNNGULayer:__tostring()
        return string.format('nnn.GULayer(%s)', tostring(self.layer))
    end

    return NNNGULayer(guLayer)
end

-- ============================================================================
-- nnn.GUSequential: Sequential container for nested GU models
-- ============================================================================

function nnn.GUSequential()
    local GUSequential, parent = torch.class('nnn.GUSequential', 'nn.Sequential')

    function GUSequential:__init()
        parent.__init(self)
    end

    -- Override updateOutput to handle nested ObserverseTensors
    function GUSequential:updateOutput(input)
        local currentOutput = input
        for i = 1, #self.modules do
            currentOutput = self.modules[i]:updateOutput(currentOutput)
        end
        self.output = currentOutput
        return self.output
    end

    function GUSequential:__tostring()
        local tab = '  '
        local line = '\n'
        local str = 'nnn.GUSequential {' .. line
        for i = 1, #self.modules do
            str = str .. tab .. '[' .. i .. '] ' .. tostring(self.modules[i]) .. line
        end
        str = str .. '}'
        return str
    end

    return GUSequential()
end

-- ============================================================================
-- GU-Specific Nested Criteria
-- ============================================================================
-- Criteria that work with nested structures of ObserverseTensors.
-- ============================================================================

local GUNestedCriterion, guCriterionParent = torch.class('nnn.GUNestedCriterion', 'nn.Criterion')

function GUNestedCriterion:__init(criterion, config)
    guCriterionParent.__init(self)
    self.criterion = criterion
    config = config or {}
    self.maxDepth = config.maxDepth or 10
    self.applyTo = config.applyTo or 'fiber'  -- 'base', 'fiber', or 'both'
    self.aggregation = config.aggregation or 'mean'  -- 'mean', 'sum', 'max'
end

function GUNestedCriterion:processNested(input, target, depth)
    depth = depth or 0

    if depth > self.maxDepth then
        error(string.format('Nesting depth %d exceeds maximum depth %d', depth, self.maxDepth))
    end

    -- Case 1: ObserverseTensor
    if nnn.gu.isObserverse(input) and nnn.gu.isObserverse(target) then
        local losses = {}
        local count = 0

        if self.applyTo == 'base' or self.applyTo == 'both' then
            table.insert(losses, self.criterion:forward(input.base, target.base))
            count = count + 1
        end
        if self.applyTo == 'fiber' or self.applyTo == 'both' then
            table.insert(losses, self.criterion:forward(input.fiber, target.fiber))
            count = count + 1
        end

        -- Aggregate losses
        local total = 0
        for _, l in ipairs(losses) do total = total + l end
        return total / count
    end

    -- Case 2: Regular tensors
    if torch.isTensor(input) and torch.isTensor(target) then
        return self.criterion:forward(input, target)
    end

    -- Case 3: Nested tables
    if type(input) == 'table' and type(target) == 'table' then
        local losses = {}
        local count = 0
        for k, v in pairs(input) do
            if target[k] then
                table.insert(losses, self:processNested(v, target[k], depth + 1))
                count = count + 1
            end
        end

        if count == 0 then return 0 end

        -- Aggregate based on mode
        if self.aggregation == 'sum' then
            local total = 0
            for _, l in ipairs(losses) do total = total + l end
            return total
        elseif self.aggregation == 'max' then
            local maxLoss = losses[1]
            for i = 2, #losses do
                if losses[i] > maxLoss then maxLoss = losses[i] end
            end
            return maxLoss
        else  -- mean
            local total = 0
            for _, l in ipairs(losses) do total = total + l end
            return total / count
        end
    end

    error("input and target must both be tensors, ObserverseTensors, or tables")
end

function GUNestedCriterion:updateOutput(input, target)
    self.output = self:processNested(input, target, 0)
    return self.output
end

function GUNestedCriterion:processNestedGrad(input, target, depth)
    depth = depth or 0

    -- Case 1: ObserverseTensor
    if nnn.gu.isObserverse(input) and nnn.gu.isObserverse(target) then
        local gu = getGU()
        local base_grad = input.base:clone():zero()
        local fiber_grad = input.fiber:clone():zero()
        local count = 0

        if self.applyTo == 'base' or self.applyTo == 'both' then
            base_grad = self.criterion:backward(input.base, target.base)
            count = count + 1
        end
        if self.applyTo == 'fiber' or self.applyTo == 'both' then
            fiber_grad = self.criterion:backward(input.fiber, target.fiber)
            count = count + 1
        end

        -- Scale gradients for averaging
        if count > 1 then
            base_grad:div(count)
            fiber_grad:div(count)
        end

        if gu then
            return gu.ObserverseTensor.create(base_grad, fiber_grad)
        else
            return {base = base_grad, fiber = fiber_grad, _type = 'ObserverseTensor'}
        end
    end

    -- Case 2: Regular tensors
    if torch.isTensor(input) and torch.isTensor(target) then
        return self.criterion:backward(input, target)
    end

    -- Case 3: Nested tables
    if type(input) == 'table' and type(target) == 'table' then
        local gradInputs = {}
        local count = 0
        for k in pairs(input) do
            if target[k] then count = count + 1 end
        end

        for k, v in pairs(input) do
            if target[k] then
                gradInputs[k] = self:processNestedGrad(v, target[k], depth + 1)
                -- Scale gradient for mean aggregation
                if self.aggregation == 'mean' and count > 1 then
                    local function scaleGrad(g)
                        if torch.isTensor(g) then
                            g:div(count)
                        elseif nnn.gu.isObserverse(g) then
                            g.base:div(count)
                            g.fiber:div(count)
                        elseif type(g) == 'table' then
                            for kk, vv in pairs(g) do
                                scaleGrad(vv)
                            end
                        end
                    end
                    scaleGrad(gradInputs[k])
                end
            end
        end
        return gradInputs
    end

    error("input and target must both be tensors, ObserverseTensors, or tables")
end

function GUNestedCriterion:updateGradInput(input, target)
    self.gradInput = self:processNestedGrad(input, target, 0)
    return self.gradInput
end

-- Factory function for GU criteria
function nnn.GUCriterion(criterion, config)
    return nnn.GUNestedCriterion(criterion, config)
end

-- Specific GU criterion wrappers
function nnn.GUMSECriterion(config)
    return nnn.GUNestedCriterion(nn.MSECriterion(), config)
end

function nnn.GUClassNLLCriterion(weights, config)
    return nnn.GUNestedCriterion(nn.ClassNLLCriterion(weights), config)
end

function nnn.GUBCECriterion(weights, config)
    return nnn.GUNestedCriterion(nn.BCECriterion(weights), config)
end

function nnn.GUCrossEntropyCriterion(weights, config)
    return nnn.GUNestedCriterion(nn.CrossEntropyCriterion(weights), config)
end

-- ============================================================================
-- GU-Specific Nested Criterion: ObserverseMSE
-- ============================================================================
-- MSE criterion specifically designed for ObserverseTensor comparisons,
-- with configurable weighting between base and fiber components.
-- ============================================================================

local ObserverseMSECriterion, obseMSEParent = torch.class('nnn.ObserverseMSECriterion', 'nn.Criterion')

function ObserverseMSECriterion:__init(config)
    obseMSEParent.__init(self)
    config = config or {}
    self.baseWeight = config.baseWeight or 1.0
    self.fiberWeight = config.fiberWeight or 1.0
    self.maxDepth = config.maxDepth or 10
    self.baseCriterion = nn.MSECriterion()
    self.fiberCriterion = nn.MSECriterion()
end

function ObserverseMSECriterion:computeLoss(input, target, depth)
    depth = depth or 0

    if depth > self.maxDepth then
        error("Maximum depth exceeded")
    end

    if nnn.gu.isObserverse(input) and nnn.gu.isObserverse(target) then
        local baseLoss = self.baseCriterion:forward(input.base, target.base)
        local fiberLoss = self.fiberCriterion:forward(input.fiber, target.fiber)
        return self.baseWeight * baseLoss + self.fiberWeight * fiberLoss
    end

    if type(input) == 'table' and type(target) == 'table' then
        local totalLoss = 0
        local count = 0
        for k, v in pairs(input) do
            if target[k] then
                totalLoss = totalLoss + self:computeLoss(v, target[k], depth + 1)
                count = count + 1
            end
        end
        return count > 0 and totalLoss / count or 0
    end

    error("Expected ObserverseTensor or nested table")
end

function ObserverseMSECriterion:updateOutput(input, target)
    self.output = self:computeLoss(input, target, 0)
    return self.output
end

function ObserverseMSECriterion:computeGrad(input, target, depth)
    depth = depth or 0
    local gu = getGU()

    if nnn.gu.isObserverse(input) and nnn.gu.isObserverse(target) then
        local baseGrad = self.baseCriterion:backward(input.base, target.base)
        local fiberGrad = self.fiberCriterion:backward(input.fiber, target.fiber)
        baseGrad:mul(self.baseWeight)
        fiberGrad:mul(self.fiberWeight)

        if gu then
            return gu.ObserverseTensor.create(baseGrad, fiberGrad)
        else
            return {base = baseGrad, fiber = fiberGrad, _type = 'ObserverseTensor'}
        end
    end

    if type(input) == 'table' and type(target) == 'table' then
        local gradInputs = {}
        local count = 0
        for k in pairs(input) do
            if target[k] then count = count + 1 end
        end

        for k, v in pairs(input) do
            if target[k] then
                gradInputs[k] = self:computeGrad(v, target[k], depth + 1)
                -- Scale for mean
                if count > 1 then
                    local function scale(g)
                        if torch.isTensor(g) then g:div(count)
                        elseif nnn.gu.isObserverse(g) then
                            g.base:div(count)
                            g.fiber:div(count)
                        elseif type(g) == 'table' then
                            for kk, vv in pairs(g) do scale(vv) end
                        end
                    end
                    scale(gradInputs[k])
                end
            end
        end
        return gradInputs
    end

    error("Expected ObserverseTensor or nested table")
end

function ObserverseMSECriterion:updateGradInput(input, target)
    self.gradInput = self:computeGrad(input, target, 0)
    return self.gradInput
end

-- ============================================================================
-- Utility Functions for NNN-GU Integration
-- ============================================================================

-- Flatten nested ObserverseTensors to a list
function nnn.gu.flatten(input)
    local result = {}

    local function collect(inp)
        if nnn.gu.isObserverse(inp) then
            table.insert(result, inp)
        elseif type(inp) == 'table' then
            for _, v in pairs(inp) do
                collect(v)
            end
        end
    end

    collect(input)
    return result
end

-- Count ObserverseTensors in nested structure
function nnn.gu.count(input)
    local count = 0

    local function counter(inp)
        if nnn.gu.isObserverse(inp) then
            count = count + 1
        elseif type(inp) == 'table' then
            for _, v in pairs(inp) do
                counter(v)
            end
        end
    end

    counter(input)
    return count
end

-- Map function over nested ObserverseTensors
function nnn.gu.map(input, func)
    if nnn.gu.isObserverse(input) then
        return func(input)
    elseif type(input) == 'table' then
        local result = {}
        for k, v in pairs(input) do
            result[k] = nnn.gu.map(v, func)
        end
        return result
    else
        return input
    end
end

-- Clone nested ObserverseTensor structure
function nnn.gu.clone(input)
    return nnn.gu.map(input, function(obs)
        local gu = getGU()
        if gu then
            return gu.ObserverseTensor.create(obs.base:clone(), obs.fiber:clone())
        else
            return {base = obs.base:clone(), fiber = obs.fiber:clone(), _type = 'ObserverseTensor'}
        end
    end)
end

-- Get depth of nested GU structure
function nnn.gu.depth(input)
    if nnn.gu.isObserverse(input) then
        return 0
    elseif type(input) == 'table' then
        local maxDepth = 0
        for _, v in pairs(input) do
            local d = nnn.gu.depth(v)
            if d + 1 > maxDepth then
                maxDepth = d + 1
            end
        end
        return maxDepth
    else
        return 0
    end
end

-- Create random nested ObserverseTensor structure
function nnn.gu.randomNested(structure, batch_size)
    local gu = getGU()
    if not gu then
        error("GU module required for randomNested")
    end

    batch_size = batch_size or 1

    if type(structure) == 'number' then
        -- Single ObserverseTensor
        return gu.randomObserverse(batch_size)
    elseif type(structure) == 'table' then
        -- Nested structure
        local result = {}
        for k, v in pairs(structure) do
            result[k] = nnn.gu.randomNested(v, batch_size)
        end
        return result
    else
        error("structure must be a number or table")
    end
end

-- ============================================================================
-- Geonestor Neuroglyph
-- ============================================================================
-- A Geonestor Neuroglyph is a geometric nested tensor neural gauge-awareness
-- symmetry structure - the unified formalism for NNN-GU integration.
-- ============================================================================

nnn.Neuroglyph = require 'nnn.Neuroglyph'
nnn.gu.Neuroglyph = nnn.Neuroglyph

-- Create a Neuroglyph from nested structure
function nnn.gu.neuroglyph(nested, config)
    return nnn.Neuroglyph.fromNested(nested, config)
end

-- Create a Neuroglyph from model
function nnn.gu.neuroglyphFromModel(model, config)
    return nnn.Neuroglyph.fromModel(model, config)
end

-- ============================================================================
-- Bounded Learning: The Correspondence Principle
-- ============================================================================
-- Bounded Learning unifies geometric neural networks with generative language
-- models through a structural correspondence.
-- ============================================================================

nnn.BoundedLearning = require 'nnn.BoundedLearning'
nnn.gu.BoundedLearning = nnn.BoundedLearning

-- Visualize the GU ↔ GPT correspondence
function nnn.gu.correspondence()
    nnn.BoundedLearning.visualizeCorrespondence()
end

-- Display info about NNN-GU integration
function nnn.gu.info()
    print("═══════════════════════════════════════════════════════")
    print("  NNN-GU Integration: Geonestor Neuroglyphs")
    print("═══════════════════════════════════════════════════════")
    print("")
    print("  GEONESTOR NEUROGLYPH")
    print("  Geometric Nested Tensor Neural Gauge-Awareness Symmetry")
    print("")
    print("  Components:")
    print("    GEO    - Geometric Unity (Observerse, gauge, fibers)")
    print("    NESTOR - Nested Tensor structures (recursive trees)")
    print("    NEURO  - Neural network operations (learnable)")
    print("    GLYPH  - Symbolic representation (type signatures)")
    print("")
    print("───────────────────────────────────────────────────────")
    print("  Neuroglyph API:")
    print("    nnn.Neuroglyph.create(config)     - Create neuroglyph")
    print("    nnn.gu.neuroglyph(nested)         - From nested structure")
    print("    nnn.gu.neuroglyphFromModel(model) - From model")
    print("    glyph:signature()                 - Get unique signature")
    print("    glyph:primeSignature()            - Prime factorization")
    print("    glyph:createModel(config)         - Generate model")
    print("    glyph:visualize()                 - ASCII visualization")
    print("")
    print("───────────────────────────────────────────────────────")
    print("  Hybrid Modules:")
    print("    nnn.GULinear(in, out, config)   - Linear layer")
    print("    nnn.GUReLU(config)              - ReLU activation")
    print("    nnn.GUTanh(config)              - Tanh activation")
    print("    nnn.GUSigmoid(config)           - Sigmoid activation")
    print("    nnn.GUSoftMax(config)           - SoftMax activation")
    print("    nnn.GUDropout(p, config)        - Dropout")
    print("    nnn.GUBatchNormalization(n)     - Batch normalization")
    print("    nnn.GULayer(fiber_dim, config)  - Full GU layer")
    print("    nnn.GUSequential()              - Sequential container")
    print("")
    print("───────────────────────────────────────────────────────")
    print("  GU-Aware Criteria:")
    print("    nnn.GUMSECriterion(config)      - MSE for nested GU")
    print("    nnn.GUBCECriterion(w, config)   - BCE for nested GU")
    print("    nnn.ObserverseMSECriterion()    - Weighted Observerse MSE")
    print("")
    print("───────────────────────────────────────────────────────")
    print("  Utility Functions:")
    print("    nnn.transformGU(module, config) - Wrap any nn module")
    print("    nnn.gu.isObserverse(input)      - Check ObserverseTensor")
    print("    nnn.gu.flatten(input)           - Flatten to list")
    print("    nnn.gu.map(input, func)         - Map over structure")
    print("    nnn.gu.clone(input)             - Deep clone")
    print("    nnn.gu.depth(input)             - Get nesting depth")
    print("    nnn.gu.count(input)             - Count ObserverseTensors")
    print("    nnn.gu.randomNested(struct, n)  - Random nested structure")
    print("═══════════════════════════════════════════════════════")
end

return nnn
