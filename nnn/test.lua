-- Tests for NNN Functional Operator System

require 'nn'
local nnn = require 'nnn'

local mytester = torch.Tester()
local nnntest = torch.TestSuite()

-- Test basic transformation
function nnntest.transform_basic()
    local linear = nn.Linear(10, 5)
    local nested = nnn.transform(linear)
    
    mytester:assert(nested ~= nil, 'transformation should create module')
    mytester:assert(nested.module ~= nil, 'should have wrapped module')
    mytester:eq(nested.module, linear, 'should wrap the correct module')
end

-- Test transformation with single tensor
function nnntest.transform_single_tensor()
    local linear = nn.Linear(10, 5)
    local nested = nnn.transform(linear)
    
    local input = torch.randn(3, 10)
    local output = nested:forward(input)
    
    mytester:assert(torch.isTensor(output), 'output should be tensor')
    mytester:eq(output:size(1), 3, 'output dim 1')
    mytester:eq(output:size(2), 5, 'output dim 2')
end

-- Test transformation with nested tensor
function nnntest.transform_nested_tensor()
    local linear = nn.Linear(10, 5)
    local nested = nnn.transform(linear)
    
    local input = {
        torch.randn(2, 10),
        torch.randn(4, 10)
    }
    local output = nested:forward(input)
    
    mytester:assert(type(output) == 'table', 'output should be table')
    mytester:eq(#output, 2, 'output should have 2 elements')
    mytester:assert(torch.isTensor(output[1]), 'first element should be tensor')
    mytester:assert(torch.isTensor(output[2]), 'second element should be tensor')
    mytester:eq(output[1]:size(1), 2, 'first output dim 1')
    mytester:eq(output[2]:size(1), 4, 'second output dim 1')
end

-- Test nested structure preservation
function nnntest.structure_preservation()
    local tanh = nnn.Tanh()
    
    local input = {
        {torch.randn(5), torch.randn(3)},
        torch.randn(4)
    }
    
    local output = tanh:forward(input)
    
    mytester:assert(type(output) == 'table', 'output should be table')
    mytester:eq(#output, 2, 'output should have 2 top-level elements')
    mytester:assert(type(output[1]) == 'table', 'first element should be table')
    mytester:eq(#output[1], 2, 'first element should have 2 sub-elements')
    mytester:assert(torch.isTensor(output[2]), 'second element should be tensor')
end

-- Test backward pass with single tensor
function nnntest.backward_single_tensor()
    local linear = nn.Linear(10, 5)
    local nested = nnn.transform(linear)
    
    local input = torch.randn(3, 10)
    local output = nested:forward(input)
    
    local gradOutput = torch.randn(3, 5)
    local gradInput = nested:backward(input, gradOutput)
    
    mytester:assert(torch.isTensor(gradInput), 'gradInput should be tensor')
    mytester:assert(gradInput:isSameSizeAs(input), 'gradInput should match input size')
end

-- Test backward pass with nested tensor
function nnntest.backward_nested_tensor()
    local linear = nn.Linear(10, 5)
    local nested = nnn.transform(linear)
    
    local input = {
        torch.randn(2, 10),
        torch.randn(3, 10)
    }
    local output = nested:forward(input)
    
    local gradOutput = {
        torch.randn(2, 5),
        torch.randn(3, 5)
    }
    local gradInput = nested:backward(input, gradOutput)
    
    mytester:assert(type(gradInput) == 'table', 'gradInput should be table')
    mytester:eq(#gradInput, 2, 'gradInput should have 2 elements')
    mytester:assert(gradInput[1]:isSameSizeAs(input[1]), 'gradInput[1] size')
    mytester:assert(gradInput[2]:isSameSizeAs(input[2]), 'gradInput[2] size')
end

-- Test nnn.Linear
function nnntest.nnn_linear()
    local linear = nnn.Linear(10, 5)
    
    mytester:assert(linear ~= nil, 'nnn.Linear should create module')
    
    local input = {torch.randn(10), torch.randn(10)}
    local output = linear:forward(input)
    
    mytester:assert(type(output) == 'table', 'output should be table')
    mytester:eq(#output, 2, 'output should have 2 elements')
end

-- Test nnn.Sequential
function nnntest.nnn_sequential()
    local model = nnn.Sequential()
        :add(nnn.Linear(10, 5))
        :add(nnn.Tanh())
    
    local input = {torch.randn(10), torch.randn(10)}
    local output = model:forward(input)
    
    mytester:assert(type(output) == 'table', 'output should be table')
    mytester:eq(#output, 2, 'output should have 2 elements')
    mytester:eq(output[1]:size(1), 5, 'output dimension')
end

-- Test nnn.ReLU
function nnntest.nnn_relu()
    local relu = nnn.ReLU()
    
    local input = {
        torch.Tensor({-1, 2, -3, 4}),
        torch.Tensor({-5, 6})
    }
    local output = relu:forward(input)
    
    mytester:assert(output[1][1] == 0, 'negative values should be zero')
    mytester:assert(output[1][2] == 2, 'positive values preserved')
    mytester:assert(output[2][1] == 0, 'negative values should be zero')
    mytester:assert(output[2][2] == 6, 'positive values preserved')
end

-- Test nnn.Sigmoid
function nnntest.nnn_sigmoid()
    local sigmoid = nnn.Sigmoid()
    
    local input = {torch.zeros(5), torch.zeros(3)}
    local output = sigmoid:forward(input)
    
    -- Sigmoid of 0 should be 0.5
    mytester:assert(math.abs(output[1][1] - 0.5) < 0.01, 'sigmoid(0) = 0.5')
    mytester:assert(math.abs(output[2][1] - 0.5) < 0.01, 'sigmoid(0) = 0.5')
end

-- Test nnn.Criterion (MSE)
function nnntest.nnn_mse_criterion()
    local criterion = nnn.MSECriterion()
    
    local predictions = {
        torch.Tensor({1, 2, 3}),
        torch.Tensor({4, 5})
    }
    local targets = {
        torch.Tensor({1, 2, 3}),
        torch.Tensor({4, 5})
    }
    
    local loss = criterion:forward(predictions, targets)
    
    mytester:assert(loss ~= nil, 'loss should not be nil')
    mytester:assert(math.abs(loss) < 0.0001, 'loss should be near zero for identical tensors')
end

-- Test nnn.Criterion backward
function nnntest.nnn_criterion_backward()
    local criterion = nnn.MSECriterion()
    
    local predictions = {
        torch.randn(5),
        torch.randn(3)
    }
    local targets = {
        torch.randn(5),
        torch.randn(3)
    }
    
    local loss = criterion:forward(predictions, targets)
    local gradInput = criterion:backward(predictions, targets)
    
    mytester:assert(type(gradInput) == 'table', 'gradInput should be table')
    mytester:eq(#gradInput, 2, 'gradInput should have 2 elements')
    mytester:assert(gradInput[1]:isSameSizeAs(predictions[1]), 'gradInput[1] size')
    mytester:assert(gradInput[2]:isSameSizeAs(predictions[2]), 'gradInput[2] size')
end

-- Test nnn.wrap
function nnntest.nnn_wrap()
    local linear = nn.Linear(10, 5)
    local wrapped = nnn.wrap(linear)
    
    mytester:assert(wrapped ~= nil, 'wrap should create module')
    mytester:eq(wrapped.module, linear, 'should wrap correct module')
end

-- Test nnn.fromNN
function nnntest.nnn_fromNN()
    local sigmoid = nnn.fromNN('Sigmoid')
    
    mytester:assert(sigmoid ~= nil, 'fromNN should create module')
    
    local input = {torch.randn(5), torch.randn(3)}
    local output = sigmoid:forward(input)
    
    mytester:assert(type(output) == 'table', 'output should be table')
end

-- Test nnn.isNested
function nnntest.utility_isNested()
    local tensor = torch.randn(5)
    local nested = {torch.randn(5)}
    
    mytester:assert(not nnn.isNested(tensor), 'single tensor should not be nested')
    mytester:assert(nnn.isNested(nested), 'table should be nested')
end

-- Test nnn.depth
function nnntest.utility_depth()
    local tensor = torch.randn(5)
    local nested1 = {tensor, tensor}
    local nested2 = {{tensor}, tensor}
    
    mytester:eq(nnn.depth(tensor), 0, 'tensor depth is 0')
    mytester:eq(nnn.depth(nested1), 1, 'single nesting depth is 1')
    mytester:eq(nnn.depth(nested2), 2, 'double nesting depth is 2')
end

-- Test nnn.flatten
function nnntest.utility_flatten()
    local t1 = torch.randn(5)
    local t2 = torch.randn(3)
    local t3 = torch.randn(4)
    
    local nested = {t1, {t2, t3}}
    local flattened = nnn.flatten(nested)
    
    mytester:eq(#flattened, 3, 'flattened should have 3 tensors')
    mytester:eq(flattened[1], t1, 'first tensor')
    mytester:eq(flattened[2], t2, 'second tensor')
    mytester:eq(flattened[3], t3, 'third tensor')
end

-- Test nnn.clone
function nnntest.utility_clone()
    local tensor = torch.randn(5)
    local nested = {tensor, {tensor:clone()}}
    
    local cloned = nnn.clone(nested)
    
    mytester:assert(cloned ~= nested, 'clone should be different object')
    mytester:assert(cloned[1] ~= nested[1], 'cloned tensors should be different')
    mytester:assert(cloned[1]:isSameSizeAs(nested[1]), 'cloned tensor same size')
end

-- Test nnn.map
function nnntest.utility_map()
    local nested = {
        torch.ones(3),
        {torch.ones(2), torch.ones(4)}
    }
    
    local doubled = nnn.map(nested, function(t) return t * 2 end)
    
    mytester:eq(doubled[1][1], 2, 'first tensor should be doubled')
    mytester:eq(doubled[2][1][1], 2, 'nested tensor should be doubled')
end

-- Test deep nesting
function nnntest.deep_nesting()
    local relu = nnn.ReLU()
    
    local deepNested = {
        {
            {torch.randn(5)},
            torch.randn(3)
        },
        torch.randn(4)
    }
    
    local output = relu:forward(deepNested)
    
    mytester:assert(type(output) == 'table', 'output should be table')
    mytester:assert(type(output[1]) == 'table', 'first level nested')
    mytester:assert(type(output[1][1]) == 'table', 'second level nested')
    mytester:assert(torch.isTensor(output[1][1][1]), 'deepest level is tensor')
end

-- Test gradient accumulation
function nnntest.gradient_accumulation()
    local linear = nn.Linear(10, 5)
    local nested = nnn.transform(linear)
    
    local input = {torch.randn(2, 10), torch.randn(3, 10)}
    local output = nested:forward(input)
    
    local gradOutput = {torch.ones(2, 5), torch.ones(3, 5)}
    
    nested:zeroGradParameters()
    nested:backward(input, gradOutput)
    
    -- Check that gradients were accumulated
    mytester:assert(linear.gradWeight:sum() ~= 0, 'gradients should be accumulated')
end

-- Test type conversion
function nnntest.type_conversion()
    local linear = nnn.Linear(10, 5)
    
    -- Convert to double
    linear:double()
    
    local input = {torch.randn(10):double(), torch.randn(10):double()}
    local output = linear:forward(input)
    
    mytester:assert(output[1]:type() == 'torch.DoubleTensor', 'output should be double')
end

-- Integration test
function nnntest.integration_full_model()
    -- Build a complete model
    local model = nnn.Sequential()
        :add(nnn.Linear(20, 15))
        :add(nnn.ReLU())
        :add(nnn.Linear(15, 10))
        :add(nnn.Tanh())
    
    local criterion = nnn.MSECriterion()
    
    -- Create nested input
    local input = {
        {torch.randn(20), torch.randn(20)},
        torch.randn(20)
    }
    
    -- Create matching target
    local target = {
        {torch.randn(10), torch.randn(10)},
        torch.randn(10)
    }
    
    -- Forward
    local output = model:forward(input)
    local loss = criterion:forward(output, target)
    
    mytester:assert(loss ~= nil, 'loss should be computed')
    
    -- Backward
    local gradOutput = criterion:backward(output, target)
    model:zeroGradParameters()
    local gradInput = model:backward(input, gradOutput)
    
    mytester:assert(type(gradInput) == 'table', 'gradInput should be table')
    
    -- Update
    model:updateParameters(0.01)
    
    -- Verify structure preservation
    mytester:assert(type(output[1]) == 'table', 'output structure preserved')
    mytester:eq(#output, 2, 'output top level')
    mytester:eq(#output[1], 2, 'output nested level')
end

-- Run tests
mytester:add(nnntest)
mytester:run()

return nnntest
