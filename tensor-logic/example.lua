#!/usr/bin/env lua
--[[
Simple example demonstrating Tensor Logic in Lua

This script shows the basic usage of the tensor logic module
for both symbolic AI (logic programming) and neural networks.

Run with: lua example.lua
Or with LuaJIT: luajit example.lua
Or with Torch: th example.lua
--]]

-- Add parent directory to path to find the module
package.path = package.path .. ';./?.lua;./?/init.lua'

-- Load the tensor logic module
local tl = require 'tensor-logic'

print('========================================')
print('   Tensor Logic Demo')
print('========================================')
print()

-- Show module info
tl.info()

print()
print('========================================')
print('   Example 1: Basic Tensor Operations')
print('========================================')
print()

-- Create tensors
local v1 = tl.fromVector('v1', 'i', {1, 2, 3})
local v2 = tl.fromVector('v2', 'i', {4, 5, 6})

print('Vector 1:', tl.tensorToString(v1))
print('Vector 2:', tl.tensorToString(v2))

-- Add vectors
local sum = tl.add(v1, v2)
print('Sum:', tl.tensorToString(sum))

-- Dot product using einsum
local dot = tl.einsum('i,i->', v1, v2)
print('Dot product:', tl.tensorToString(dot))

-- Matrix multiplication
print()
local A = tl.fromMatrix('A', {'i', 'j'}, {{1, 2}, {3, 4}})
local B = tl.fromMatrix('B', {'j', 'k'}, {{5, 6}, {7, 8}})
print('Matrix A:')
print(tl.tensorToString(A))
print('Matrix B:')
print(tl.tensorToString(B))

local C = tl.einsum('ij,jk->ik', A, B)
print('Matrix product C = A × B:')
print(tl.tensorToString(C))

print()
print('========================================')
print('   Example 2: Logic Programming')
print('========================================')
print()

-- Run logic programming example
local logic_result = tl.examples.logic.runLogicProgramExample()
print(logic_result.title)
print()
print(logic_result.description)
print()
print('Number of steps:', #logic_result.steps)
print()
print('Final Ancestor relation:')
print(logic_result.steps[#logic_result.steps].tensorString)

print()
print('========================================')
print('   Example 3: Neural Network (MLP)')
print('========================================')
print()

-- Run MLP example
local mlp_result = tl.examples.mlp.runMLPExample()
print(mlp_result.title)
print()
print(mlp_result.description)
print()
print('Number of steps:', #mlp_result.steps)
print()
print('Final output:')
print(mlp_result.steps[#mlp_result.steps].tensorString)
print()
print('For input [1, 0], XOR output is', mlp_result.steps[#mlp_result.steps].tensor.data[1])
print('Since this is > 0.5, the network correctly predicts XOR(1,0) = 1')

print()
print('========================================')
print('   Example 4: Batch Processing')
print('========================================')
print()

-- Run batch XOR example
local batch_result = tl.examples.mlp.runFullXORDemo()
print(batch_result.title)
print()
print('Processing all 4 XOR inputs simultaneously:')
print()
local batch_output = batch_result.steps[#batch_result.steps].tensor
local xor_cases = {
    {input = '(0,0)', expected = 0},
    {input = '(0,1)', expected = 1},
    {input = '(1,0)', expected = 1},
    {input = '(1,1)', expected = 0}
}

for i, case in ipairs(xor_cases) do
    local output = batch_output.data[i]
    local prediction = output > 0.5 and 1 or 0
    local status = prediction == case.expected and '✓' or '✗'
    print(string.format('  XOR%s = %.3f → %d %s (expected %d)',
        case.input, output, prediction, status, case.expected))
end

print()
print('========================================')
print('   Summary')
print('========================================')
print()
print('Tensor Logic successfully demonstrates:')
print('  ✓ Symbolic AI (Logic Programming)')
print('  ✓ Neural Networks (Multi-Layer Perceptron)')
print('  ✓ Unified framework using Einstein summation')
print()
print('Both paradigms use the SAME tensor operations!')
print('The only difference is the data type:')
print('  - Boolean (0/1) for logic')
print('  - Real numbers for neural networks')
print()
print('========================================')
