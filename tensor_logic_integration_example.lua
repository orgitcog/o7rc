#!/usr/bin/env lua
--[[
Tensor Logic Integration Example with Torch7u

This example demonstrates how to use the tensor-logic module
within the Torch7u unified framework.

Based on Pedro Domingos' "Tensor Logic: The Language of AI"
https://arxiv.org/abs/2510.12269

Run with: th tensor_logic_integration_example.lua
--]]

-- Load the integrated torch7u system
require 'init'

print('=' .. string.rep('=', 70))
print('  Tensor Logic Integration with Torch7u')
print('=' .. string.rep('=', 70))
print()

-- Check if tensor-logic is loaded
if torch7u.tensor_logic then
    print('✓ Tensor Logic module successfully loaded into Torch7u!')
    print()
    
    -- Access via torch7u namespace
    local tl = torch7u.tensor_logic
    
    -- Or use the convenience accessor
    -- local tl = torch7u.tl
    
    -- Display module information
    print('-' .. string.rep('-', 70))
    print('  Module Information')
    print('-' .. string.rep('-', 70))
    tl.info()
    
    print()
    print('-' .. string.rep('-', 70))
    print('  Example 1: Symbolic AI - Logic Programming')
    print('-' .. string.rep('-', 70))
    print()
    
    -- Run logic programming example
    local logic_result = tl.examples.logic.runLogicProgramExample()
    print('Title:', logic_result.title)
    print()
    print('This example demonstrates how Datalog-style logic programming')
    print('maps directly to tensor operations through Einstein summation.')
    print()
    print('Logical Rule:')
    print('  Ancestor(x, z) ← Ancestor(x, y), Parent(y, z)')
    print()
    print('Tensor Logic Equation:')
    print('  Ancestor[x, z] = Σ_y Ancestor[x, y] · Parent[y, z]')
    print()
    print('Final Ancestor Relation (after transitive closure):')
    print(logic_result.steps[#logic_result.steps].tensorString)
    
    print()
    print('-' .. string.rep('-', 70))
    print('  Example 2: Neural Networks - Multi-Layer Perceptron')
    print('-' .. string.rep('-', 70))
    print()
    
    -- Run MLP example
    local mlp_result = tl.examples.mlp.runMLPExample()
    print('Title:', mlp_result.title)
    print()
    print('This example shows how neural network layers are tensor operations.')
    print()
    print('Each MLP layer computes:')
    print('  Output[o] = activation(Σ_i Weight[o,i] · Input[i] + Bias[o])')
    print()
    print('Which is identical in structure to a logical rule:')
    print('  Head[o] ← Body[o,i] · Facts[i]')
    print()
    print('XOR(1,0) computed by the network:')
    print(mlp_result.steps[#mlp_result.steps].tensorString)
    local xor_output = mlp_result.steps[#mlp_result.steps].tensor.data[1]
    print(string.format('Output: %.3f → Classification: %d (Correct!)', 
        xor_output, xor_output > 0.5 and 1 or 0))
    
    print()
    print('-' .. string.rep('-', 70))
    print('  Example 3: Batch Processing')
    print('-' .. string.rep('-', 70))
    print()
    
    -- Run batch XOR example
    local batch_result = tl.examples.mlp.runFullXORDemo()
    print('Processing all 4 XOR inputs simultaneously using batch operations:')
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
        print(string.format('  XOR%s = %.3f → %d %s',
            case.input, output, prediction, status))
    end
    
    print()
    print('-' .. string.rep('-', 70))
    print('  Key Insights')
    print('-' .. string.rep('-', 70))
    print()
    print('1. UNIFIED FRAMEWORK: Both symbolic AI and neural networks')
    print('   use the same tensor operations (Einstein summation).')
    print()
    print('2. DATA TYPE DIFFERENCE: The only difference is:')
    print('   • Boolean (0/1) tensors → Symbolic logic')
    print('   • Real-valued tensors → Neural networks')
    print()
    print('3. INTEGRATION: Tensor Logic works seamlessly within Torch7u,')
    print('   enabling hybrid neuro-symbolic AI systems.')
    print()
    print('4. EFFICIENCY: Operations are tensor-based, enabling')
    print('   batch processing and potential GPU acceleration.')
    
    print()
    print('-' .. string.rep('-', 70))
    print('  Integration with Torch7u Features')
    print('-' .. string.rep('-', 70))
    print()
    print('Tensor Logic is now available throughout the Torch7u system:')
    print('  • Access via: torch7u.tensor_logic or torch7u.tl')
    print('  • Compatible with torch tensors (conversion possible)')
    print('  • Can be used with torch7u event system')
    print('  • Integrates with torch7u model registry')
    print('  • Available in torch7u plugin ecosystem')
    
    print()
    print('=' .. string.rep('=', 70))
    print('  Tensor Logic Integration Complete!')
    print('=' .. string.rep('=', 70))
    
else
    print('✗ Tensor Logic module not loaded!')
    print('Please ensure the tensor-logic directory is in the Lua path.')
end
