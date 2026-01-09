--[[
Test suite for Tensor Logic

Tests core functionality including:
- Tensor creation and manipulation
- Einstein summation
- Activation functions
- Logic programming
- Neural networks (MLP)
--]]

package.path = package.path .. ';../?.lua;../?/init.lua'

local tl = require 'tensor-logic'

local tests_passed = 0
local tests_failed = 0

-- Test helper functions
local function assertEqual(actual, expected, test_name, tolerance)
    tolerance = tolerance or 1e-6
    local passed = false
    
    if type(actual) == 'number' and type(expected) == 'number' then
        passed = math.abs(actual - expected) < tolerance
    elseif type(actual) == 'table' and type(expected) == 'table' then
        if #actual ~= #expected then
            passed = false
        else
            passed = true
            for i = 1, #actual do
                if math.abs(actual[i] - expected[i]) > tolerance then
                    passed = false
                    break
                end
            end
        end
    else
        passed = (actual == expected)
    end
    
    if passed then
        print(string.format('✓ %s', test_name))
        tests_passed = tests_passed + 1
    else
        print(string.format('✗ %s', test_name))
        print(string.format('  Expected: %s', tostring(expected)))
        print(string.format('  Got:      %s', tostring(actual)))
        tests_failed = tests_failed + 1
    end
end

-- Test tensor creation
print('\n=== Testing Tensor Creation ===')

local t1 = tl.createTensor('test', {'x'}, {3}, 'zeros')
assertEqual(#t1.data, 3, 'Create 1D tensor with zeros')
assertEqual(t1.data[1], 0, 'First element is zero')

local t2 = tl.createTensor('test', {'x', 'y'}, {2, 3}, 'ones')
assertEqual(#t2.data, 6, 'Create 2D tensor with ones')
assertEqual(t2.data[1], 1, 'First element is one')

local t3 = tl.fromVector('vec', 'x', {1, 2, 3})
assertEqual(t3.data[2], 2, 'From vector - second element')

local t4 = tl.fromMatrix('mat', {'x', 'y'}, {{1, 2}, {3, 4}})
assertEqual(tl.getElement(t4, 2, 1), 3, 'From matrix - element at (2,1)')

-- Test element access
print('\n=== Testing Element Access ===')

local m = tl.fromMatrix('m', {'i', 'j'}, {{1, 2, 3}, {4, 5, 6}})
assertEqual(tl.getElement(m, 1, 1), 1, 'Get element (1,1)')
assertEqual(tl.getElement(m, 2, 3), 6, 'Get element (2,3)')

tl.setElement(m, 99, 1, 2)
assertEqual(tl.getElement(m, 1, 2), 99, 'Set and get element')

-- Test Einstein summation (matrix multiplication)
print('\n=== Testing Einstein Summation ===')

local A = tl.fromMatrix('A', {'i', 'j'}, {{1, 2}, {3, 4}})
local B = tl.fromMatrix('B', {'j', 'k'}, {{1, 0}, {0, 1}})
local C = tl.einsum('ij,jk->ik', A, B)

assertEqual(tl.getElement(C, 1, 1), 1, 'Matrix mult (1,1)')
assertEqual(tl.getElement(C, 1, 2), 2, 'Matrix mult (1,2)')
assertEqual(tl.getElement(C, 2, 1), 3, 'Matrix mult (2,1)')
assertEqual(tl.getElement(C, 2, 2), 4, 'Matrix mult (2,2)')

-- Test vector dot product
local v1 = tl.fromVector('v1', 'i', {1, 2, 3})
local v2 = tl.fromVector('v2', 'i', {4, 5, 6})
local dot = tl.einsum('i,i->', v1, v2)
assertEqual(dot.data[1], 32, 'Dot product: 1*4 + 2*5 + 3*6 = 32')

-- Test activation functions
print('\n=== Testing Activation Functions ===')

local x = tl.fromVector('x', 'i', {-1, 0, 1, 2})

local relu_result = tl.relu(x)
assertEqual(relu_result.data[1], 0, 'ReLU(-1) = 0')
assertEqual(relu_result.data[2], 0, 'ReLU(0) = 0')
assertEqual(relu_result.data[3], 1, 'ReLU(1) = 1')
assertEqual(relu_result.data[4], 2, 'ReLU(2) = 2')

local threshold_result = tl.threshold(x, 0)
assertEqual(threshold_result.data[1], 0, 'Threshold(-1) = 0')
assertEqual(threshold_result.data[3], 1, 'Threshold(1) = 1')

local sigmoid_result = tl.sigmoid(tl.fromVector('x', 'i', {0}))
assertEqual(sigmoid_result.data[1], 0.5, 'Sigmoid(0) = 0.5', 1e-5)

-- Test tensor operations
print('\n=== Testing Tensor Operations ===')

local a = tl.fromVector('a', 'i', {1, 2, 3})
local b = tl.fromVector('b', 'i', {4, 5, 6})

local sum = tl.add(a, b)
assertEqual(sum.data[1], 5, 'Add vectors (1)')
assertEqual(sum.data[2], 7, 'Add vectors (2)')
assertEqual(sum.data[3], 9, 'Add vectors (3)')

local prod = tl.multiply(a, b)
assertEqual(prod.data[1], 4, 'Multiply vectors (1)')
assertEqual(prod.data[2], 10, 'Multiply vectors (2)')
assertEqual(prod.data[3], 18, 'Multiply vectors (3)')

local scaled = tl.scale(a, 2)
assertEqual(scaled.data[1], 2, 'Scale vector (1)')
assertEqual(scaled.data[2], 4, 'Scale vector (2)')
assertEqual(scaled.data[3], 6, 'Scale vector (3)')

-- Test transpose
local mat = tl.fromMatrix('mat', {'i', 'j'}, {{1, 2, 3}, {4, 5, 6}})
local transposed = tl.transpose(mat)
assertEqual(transposed.shape[1], 3, 'Transposed shape[1]')
assertEqual(transposed.shape[2], 2, 'Transposed shape[2]')
assertEqual(tl.getElement(transposed, 1, 1), 1, 'Transposed (1,1)')
assertEqual(tl.getElement(transposed, 3, 2), 6, 'Transposed (3,2)')

-- Test identity matrix
local I = tl.identity('I', {'i', 'j'}, 3)
assertEqual(tl.getElement(I, 1, 1), 1, 'Identity (1,1)')
assertEqual(tl.getElement(I, 2, 2), 1, 'Identity (2,2)')
assertEqual(tl.getElement(I, 1, 2), 0, 'Identity (1,2)')

-- Test logic programming example
print('\n=== Testing Logic Programming Example ===')

local logic_result = tl.examples.logic.runLogicProgramExample()
assertEqual(logic_result.title, 'Logic Programming: Family Relationships', 'Logic example title')
local final_ancestor = logic_result.steps[#logic_result.steps].tensor
-- Alice(1) should be ancestor of Charlie(3)
assertEqual(tl.getElement(final_ancestor, 1, 3), 1, 'Alice is ancestor of Charlie')
-- Alice(1) should be ancestor of Diana(4)
assertEqual(tl.getElement(final_ancestor, 1, 4), 1, 'Alice is ancestor of Diana')
-- Charlie(3) should NOT be ancestor of anyone
assertEqual(tl.getElement(final_ancestor, 3, 1), 0, 'Charlie is not ancestor of Alice')

-- Test MLP example
print('\n=== Testing MLP Example ===')

local mlp_result = tl.examples.mlp.runMLPExample()
assertEqual(mlp_result.title, 'Multi-Layer Perceptron: XOR Problem', 'MLP example title')
local final_output = mlp_result.steps[#mlp_result.steps].tensor
-- Output should be close to 1 for XOR(1,0)
local xor_output = final_output.data[1]
local is_correct = xor_output > 0.5 and xor_output < 0.9
if is_correct then
    print(string.format('✓ MLP XOR output is correct: %.3f', xor_output))
    tests_passed = tests_passed + 1
else
    print(string.format('✗ MLP XOR output is incorrect: %.3f', xor_output))
    tests_failed = tests_failed + 1
end

-- Test batch XOR
print('\n=== Testing Batch XOR Example ===')

local batch_result = tl.examples.mlp.runFullXORDemo()
local batch_output = batch_result.steps[#batch_result.steps].tensor
-- Check all four XOR outputs
local xor_cases = {
    {input = {0, 0}, expected = 0, name = 'XOR(0,0)'},
    {input = {0, 1}, expected = 1, name = 'XOR(0,1)'},
    {input = {1, 0}, expected = 1, name = 'XOR(1,0)'},
    {input = {1, 1}, expected = 0, name = 'XOR(1,1)'}
}

for i, case in ipairs(xor_cases) do
    local output = batch_output.data[i]
    local is_correct = (case.expected == 1 and output > 0.5) or (case.expected == 0 and output < 0.5)
    if is_correct then
        print(string.format('✓ Batch %s = %.2f (expected %d)', case.name, output, case.expected))
        tests_passed = tests_passed + 1
    else
        print(string.format('✗ Batch %s = %.2f (expected %d)', case.name, output, case.expected))
        tests_failed = tests_failed + 1
    end
end

-- Print summary
print('\n' .. string.rep('=', 60))
print(string.format('Tests passed: %d', tests_passed))
print(string.format('Tests failed: %d', tests_failed))
print(string.rep('=', 60))

if tests_failed == 0 then
    print('✓ All tests passed!')
    os.exit(0)
else
    print(string.format('✗ %d test(s) failed', tests_failed))
    os.exit(1)
end
