--[[
Tensor Logic - Neuro-Symbolic AI in Lua

Based on Pedro Domingos' paper "Tensor Logic: The Language of AI"
https://arxiv.org/abs/2510.12269

A Lua implementation of the tensor logic paradigm that unifies symbolic AI
and neural networks through Einstein summation.

Key Components:
- core: Fundamental tensor operations and Einstein summation
- utils: Utility functions for broadcasting and slicing
- examples: Demonstrations of logic programming, neural networks, and more
--]]

local M = {}

-- Version information
M.version = '1.0.0'
M.description = 'Neuro-Symbolic Tensor Logic for Lua'

-- Load core module
M.core = require 'tensor-logic.core'

-- Load utilities
M.utils = require 'tensor-logic.utils'

-- Load examples
M.examples = {
    logic = require 'tensor-logic.examples.logic',
    mlp = require 'tensor-logic.examples.mlp',
}

-- Convenience: expose core functions at top level
M.createTensor = M.core.createTensor
M.fromMatrix = M.core.fromMatrix
M.fromVector = M.core.fromVector
M.einsum = M.core.einsum
M.threshold = M.core.threshold
M.sigmoid = M.core.sigmoid
M.relu = M.core.relu
M.softmax = M.core.softmax
M.add = M.core.add
M.multiply = M.core.multiply
M.scale = M.core.scale
M.transpose = M.core.transpose
M.tensorToString = M.core.tensorToString
M.clone = M.core.clone
M.identity = M.core.identity
M.getElement = M.core.getElement
M.setElement = M.core.setElement

-- Print module information
function M.info()
    print(string.format('Tensor Logic v%s', M.version))
    print(M.description)
    print('\nBased on Pedro Domingos\' paper "Tensor Logic: The Language of AI"')
    print('https://arxiv.org/abs/2510.12269')
    print('\nThe key insight: logical rules and Einstein summation are the same operation.')
    print('- Boolean tensors (0/1) → Symbolic logic')
    print('- Real-valued tensors → Neural networks')
    print('\nAvailable modules:')
    print('  - core: Fundamental tensor operations')
    print('  - utils: Utility functions')
    print('  - examples.logic: Logic programming demonstrations')
    print('  - examples.mlp: Multi-layer perceptron demonstrations')
end

-- Run a quick demo
function M.demo()
    print('Running Tensor Logic Demo\n')
    print('=' .. string.rep('=', 50))
    
    -- Logic programming example
    print('\n1. LOGIC PROGRAMMING: Family Relationships')
    print('-' .. string.rep('-', 50))
    local logicResult = M.examples.logic.runLogicProgramExample()
    print(logicResult.description)
    print('\nFinal Result:')
    print(logicResult.steps[#logicResult.steps].tensorString)
    
    -- MLP example
    print('\n2. NEURAL NETWORK: Multi-Layer Perceptron (XOR)')
    print('-' .. string.rep('-', 50))
    local mlpResult = M.examples.mlp.runMLPExample()
    print(mlpResult.description)
    print('\nFinal Result:')
    print(mlpResult.steps[#mlpResult.steps].tensorString)
    
    print('\n' .. string.rep('=', 52))
    print('Demo complete! Both symbolic AI and neural networks')
    print('unified under the same tensor logic framework.')
end

return M
