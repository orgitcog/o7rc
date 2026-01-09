--[[
MULTI-LAYER PERCEPTRON (MLP) IN TENSOR LOGIC

A Multi-Layer Perceptron is the foundational neural network architecture.
It consists of layers of "neurons" where each neuron computes:
  output = activation(Σ weight_i · input_i + bias)

THE KEY INSIGHT:
Each layer of an MLP is EXACTLY the same operation as a logical rule,
just with real-valued tensors instead of Boolean ones:

Logical rule:   Head[x] ← Body1[x,y] · Body2[y]
Neural layer:   Output[x] = activation(Σ_y Weight[x,y] · Input[y] + Bias[x])

Both are: "project out the intermediate variable y by summing"
--]]

local core = require 'tensor-logic.core'
local utils = require 'tensor-logic.utils'

local M = {}

--[[
Example: XOR Problem

The XOR function cannot be computed by a single layer (it's not linearly
separable). However, a 2-layer MLP CAN compute XOR. The hidden layer learns
to create a new representation where XOR becomes linearly separable.

Input: 2D vectors (x1, x2) ∈ {0,1}²
Output: XOR(x1, x2)

We'll use pre-trained weights that solve XOR.
--]]
function M.runMLPExample()
    local steps = {}
    
    -- Input: XOR test case [1, 0] → expected output: 1
    local Input = core.fromVector('Input', 'i', {1, 0})
    
    table.insert(steps, {
        name = 'Input Vector',
        explanation = 'Input to the network: [1, 0]\n\n' ..
                     'The XOR function returns 1 when inputs differ, 0 when same:\n' ..
                     '  XOR(0,0) = 0\n' ..
                     '  XOR(0,1) = 1\n' ..
                     '  XOR(1,0) = 1  ← our test case\n' ..
                     '  XOR(1,1) = 0\n\n' ..
                     'XOR is NOT linearly separable - you cannot draw a single line that\n' ..
                     'separates the 1s from the 0s in 2D space.',
        tensor = Input,
        tensorString = core.tensorToString(Input, 3)
    })
    
    -- Layer 1: Hidden layer (2 inputs → 2 hidden neurons)
    -- These weights were derived to solve XOR:
    -- - First hidden neuron: computes AND(x1, x2) approximately
    -- - Second hidden neuron: computes OR(x1, x2) approximately
    local W1 = core.createTensor(
        'W1',
        {'h', 'i'},
        {2, 2},
        {1, 1,  -- Hidden unit 0: responds when both inputs are high (AND-like)
         1, 1}  -- Hidden unit 1: responds when either input is high (OR-like)
    )
    
    local B1 = core.fromVector('B1', 'h', {-1.5, -0.5})
    -- Bias -1.5: only activates if both inputs sum > 1.5 (AND)
    -- Bias -0.5: activates if any input sum > 0.5 (OR)
    
    table.insert(steps, {
        name = 'Layer 1 Weights',
        explanation = 'First layer transforms 2D input into 2D hidden representation.\n\n' ..
                     'Weight matrix W1[h,i]:\n' ..
                     '  Row 0 (AND-like): [1, 1] with bias -1.5\n' ..
                     '  Row 1 (OR-like):  [1, 1] with bias -0.5\n\n' ..
                     'Tensor Logic:  PreH[h] = W1[h,i] · Input[i] + B1[h]\n' ..
                     '               Hidden[h] = ReLU(PreH[h])',
        tensor = W1,
        tensorString = core.tensorToString(W1, 3)
    })
    
    -- Forward pass - Layer 1
    -- PreHidden[h] = Σ_i W1[h,i] · Input[i]
    local preHidden = core.einsum('hi,i->h', W1, Input)
    
    table.insert(steps, {
        name = 'Pre-activation (Layer 1)',
        explanation = 'Linear transformation before activation.\n\n' ..
                     'Tensor Logic:  PreH[h] = W1[h,i] · Input[i]\n' ..
                     'Einstein sum:  "hi,i->h" sums over input dimension i\n\n' ..
                     'For input [1, 0]:\n' ..
                     '  PreH[0] = 1·1 + 1·0 = 1\n' ..
                     '  PreH[1] = 1·1 + 1·0 = 1',
        tensor = preHidden,
        tensorString = core.tensorToString(preHidden, 3)
    })
    
    -- Add bias
    local preHiddenWithBias = core.add(preHidden, B1)
    
    table.insert(steps, {
        name = 'After Bias (Layer 1)',
        explanation = 'Add bias to shift the activation threshold.\n\n' ..
                     'Tensor Logic:  PreH[h] += B1[h]\n\n' ..
                     '  PreH[0] = 1 + (-1.5) = -0.5  (below threshold)\n' ..
                     '  PreH[1] = 1 + (-0.5) = 0.5   (above threshold)',
        tensor = preHiddenWithBias,
        tensorString = core.tensorToString(preHiddenWithBias, 3)
    })
    
    -- Apply ReLU activation
    local Hidden = core.relu(preHiddenWithBias)
    
    table.insert(steps, {
        name = 'Hidden Layer (after ReLU)',
        explanation = 'ReLU activation: max(0, x)\n\n' ..
                     'Tensor Logic:  Hidden[h] = ReLU(PreH[h])\n\n' ..
                     '  Hidden[0] = max(0, -0.5) = 0  (AND gate: false)\n' ..
                     '  Hidden[1] = max(0, 0.5) = 0.5 (OR gate: true)\n\n' ..
                     'The hidden layer has learned a new representation!\n' ..
                     'XOR = OR AND NOT(AND)',
        tensor = Hidden,
        tensorString = core.tensorToString(Hidden, 3)
    })
    
    -- Layer 2: Output layer (2 hidden → 1 output)
    -- XOR = OR AND NOT(AND)
    local W2 = core.createTensor(
        'W2',
        {'o', 'h'},
        {1, 2},
        {-2, 2}  -- Positive weight on OR, negative on AND
    )
    
    local B2 = core.fromVector('B2', 'o', {0})
    
    table.insert(steps, {
        name = 'Layer 2 Weights',
        explanation = 'Output layer computes XOR from hidden features.\n\n' ..
                     'Weight matrix W2[o,h]:\n' ..
                     '  [-2, 2] means: +2×(OR result) - 2×(AND result)\n\n' ..
                     'XOR = OR AND NOT(AND)\n' ..
                     'When OR=1 and AND=0: output = -2×0 + 2×1 = 2 (high)\n' ..
                     'When OR=1 and AND=1: output = -2×1 + 2×1 = 0 (low)',
        tensor = W2,
        tensorString = core.tensorToString(W2, 3)
    })
    
    -- Forward pass - Layer 2
    local preOutput = core.einsum('oh,h->o', W2, Hidden)
    local preOutputWithBias = core.add(preOutput, B2)
    
    table.insert(steps, {
        name = 'Pre-activation (Layer 2)',
        explanation = 'Linear transformation for output.\n\n' ..
                     'Tensor Logic:  PreOut[o] = W2[o,h] · Hidden[h] + B2[o]\n' ..
                     'Einstein sum:  "oh,h->o" sums over hidden dimension h\n\n' ..
                     '  PreOut[0] = -2×0 + 2×0.5 + 0 = 1',
        tensor = preOutputWithBias,
        tensorString = core.tensorToString(preOutputWithBias, 3)
    })
    
    -- Apply sigmoid for final output
    local Output = core.sigmoid(preOutputWithBias)
    
    table.insert(steps, {
        name = 'Final Output (after sigmoid)',
        explanation = 'Sigmoid activation: σ(x) = 1/(1 + e^(-x))\n\n' ..
                     'Tensor Logic:  Output[o] = σ(PreOut[o])\n\n' ..
                     '  Output[0] = σ(1) ≈ 0.731\n\n' ..
                     'This is interpreted as probability/confidence.\n' ..
                     'Since 0.731 > 0.5, we classify XOR(1,0) = 1 ✓\n\n' ..
                     'The network correctly computed XOR!',
        tensor = Output,
        tensorString = core.tensorToString(Output, 3)
    })
    
    return {
        title = 'Multi-Layer Perceptron: XOR Problem',
        description = 'This example shows how a Multi-Layer Perceptron maps to Tensor Logic.\n\n' ..
                     'Each layer is an Einstein summation followed by a pointwise activation:\n' ..
                     '  Layer: Output[o] = activation(Weight[o,i] · Input[i] + Bias[o])\n\n' ..
                     'This is identical in structure to a logical rule:\n' ..
                     '  Head[o] ← Body[o,i] · Facts[i]\n\n' ..
                     'The difference is:\n' ..
                     '- Logic uses Boolean tensors (0/1) with threshold activation\n' ..
                     '- Neural nets use real-valued tensors with smooth activations',
        steps = steps
    }
end

--[[
Demonstrates the full MLP computation with all 4 XOR inputs
--]]
function M.runFullXORDemo()
    local steps = {}
    
    -- All XOR inputs as a batch
    local Inputs = core.createTensor(
        'Inputs',
        {'b', 'i'},
        {4, 2},
        {0, 0,  -- Input 0: XOR(0,0) = 0
         0, 1,  -- Input 1: XOR(0,1) = 1
         1, 0,  -- Input 2: XOR(1,0) = 1
         1, 1}  -- Input 3: XOR(1,1) = 0
    )
    
    table.insert(steps, {
        name = 'All XOR Inputs (Batch)',
        explanation = 'Processing all 4 XOR inputs as a batch.\n\n' ..
                     'Input tensor Inputs[batch, input]:\n' ..
                     '  Row 0: [0, 0] → expected 0\n' ..
                     '  Row 1: [0, 1] → expected 1\n' ..
                     '  Row 2: [1, 0] → expected 1\n' ..
                     '  Row 3: [1, 1] → expected 0\n\n' ..
                     'Batch processing is a key efficiency gain - all inputs\n' ..
                     'are processed in parallel using the same matrix operations.',
        tensor = Inputs,
        tensorString = core.tensorToString(Inputs, 0)
    })
    
    -- Layer 1 weights
    local W1 = core.createTensor('W1', {'h', 'i'}, {2, 2}, {1, 1, 1, 1})
    local B1 = core.createTensor('B1', {'h'}, {2}, {-1.5, -0.5})
    
    -- Batch forward pass - Layer 1
    local preHidden = core.einsum('bi,hi->bh', Inputs, W1)
    
    -- Broadcast add bias to each batch element
    local preHiddenWithBias = utils.broadcastAdd(preHidden, B1, 2)
    preHiddenWithBias.name = 'PreH+B'
    
    table.insert(steps, {
        name = 'Hidden Pre-activations (Batch)',
        explanation = 'Batch computation of hidden layer before activation.\n\n' ..
                     'Tensor Logic:  PreH[b,h] = Inputs[b,i] · W1[h,i] + B1[h]\n' ..
                     'Einstein sum:  "bi,hi->bh"\n\n' ..
                     'The batch dimension b is preserved while input dimension i is contracted.',
        tensor = preHiddenWithBias,
        tensorString = core.tensorToString(preHiddenWithBias, 2)
    })
    
    -- Apply ReLU
    local Hidden = core.relu(preHiddenWithBias)
    
    table.insert(steps, {
        name = 'Hidden Layer (Batch, after ReLU)',
        explanation = 'Hidden representations for all inputs.\n\n' ..
                     'Hidden[b,h] after ReLU:\n' ..
                     '  [0,0] → [0.0, 0.0] (neither AND nor OR)\n' ..
                     '  [0,1] → [0.0, 0.5] (not AND, yes OR)\n' ..
                     '  [1,0] → [0.0, 0.5] (not AND, yes OR)\n' ..
                     '  [1,1] → [0.5, 1.5] (yes AND, yes OR)',
        tensor = Hidden,
        tensorString = core.tensorToString(Hidden, 2)
    })
    
    -- Layer 2
    local W2 = core.createTensor('W2', {'o', 'h'}, {1, 2}, {-2, 2})
    local preOutput = core.einsum('bh,oh->bo', Hidden, W2)
    
    local Output = core.sigmoid(preOutput)
    
    table.insert(steps, {
        name = 'Final Outputs (Batch)',
        explanation = 'XOR outputs for all inputs.\n\n' ..
                     'Output[b] after sigmoid:\n' ..
                     '  XOR(0,0) = 0.50 → rounds to 0 ✓\n' ..
                     '  XOR(0,1) = 0.73 → rounds to 1 ✓\n' ..
                     '  XOR(1,0) = 0.73 → rounds to 1 ✓\n' ..
                     '  XOR(1,1) = 0.27 → rounds to 0 ✓\n\n' ..
                     'All four XOR cases computed correctly!',
        tensor = Output,
        tensorString = core.tensorToString(Output, 2)
    })
    
    return {
        title = 'Batch XOR Computation',
        description = 'Demonstrates batch processing - computing all XOR inputs simultaneously.\n' ..
                     'The batch dimension is simply another tensor index that flows through unchanged.',
        steps = steps
    }
end

return M
