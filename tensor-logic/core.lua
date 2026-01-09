--[[
Tensor Logic Core Engine

Based on Pedro Domingos' paper "Tensor Logic: The Language of AI"
https://arxiv.org/abs/2510.12269

The fundamental insight: logical rules and Einstein summation are the same operation.
- In logic: Ancestor(x,z) ← Ancestor(x,y), Parent(y,z) is a JOIN on y then PROJECT to (x,z)
- In tensors: C[x,z] = Σ_y A[x,y] · B[y,z] sums over the shared index y

This unification allows us to express both symbolic AI and neural networks
in the same language, with the only difference being the atomic data types:
- Boolean (0/1) for symbolic logic
- Numeric (floats) for neural networks
--]]

local M = {}

-- Tensor class definition
-- A Tensor is a multi-dimensional array with named indices
-- The indices are the "arguments" of the relation in logic programming terms
local Tensor = {}
Tensor.__index = Tensor

function M.Tensor(name, shape, indices, data)
    local self = setmetatable({}, Tensor)
    self.name = name
    self.shape = shape
    self.indices = indices
    self.data = data or {}
    
    -- Calculate total size
    local size = 1
    for _, dim in ipairs(shape) do
        size = size * dim
    end
    self.size = size
    
    -- Initialize data if not provided
    if #self.data == 0 then
        for i = 1, size do
            self.data[i] = 0
        end
    end
    
    return self
end

-- Create a new tensor with given shape and indices
function M.createTensor(name, indices, shape, initializer)
    initializer = initializer or 'zeros'
    
    local size = 1
    for _, dim in ipairs(shape) do
        size = size * dim
    end
    
    local data = {}
    if type(initializer) == 'table' then
        data = initializer
    elseif initializer == 'zeros' then
        for i = 1, size do
            data[i] = 0
        end
    elseif initializer == 'ones' then
        for i = 1, size do
            data[i] = 1
        end
    elseif initializer == 'random' then
        for i = 1, size do
            data[i] = math.random() * 2 - 1
        end
    end
    
    return M.Tensor(name, shape, indices, data)
end

-- Create a tensor from a 2D matrix
function M.fromMatrix(name, indices, matrix)
    local rows = #matrix
    local cols = #matrix[1]
    local data = {}
    
    for i = 1, rows do
        for j = 1, cols do
            data[(i-1) * cols + j] = matrix[i][j]
        end
    end
    
    return M.Tensor(name, {rows, cols}, indices, data)
end

-- Create a 1D tensor (vector)
function M.fromVector(name, index, values)
    return M.Tensor(name, {#values}, {index}, values)
end

-- Get element at given indices (1-based indexing)
function M.getElement(tensor, ...)
    local indices = {...}
    local flatIndex = 0
    local stride = 1
    
    for i = #tensor.shape, 1, -1 do
        flatIndex = flatIndex + (indices[i] - 1) * stride
        stride = stride * tensor.shape[i]
    end
    
    return tensor.data[flatIndex + 1]
end

-- Set element at given indices (1-based indexing)
function M.setElement(tensor, value, ...)
    local indices = {...}
    local flatIndex = 0
    local stride = 1
    
    for i = #tensor.shape, 1, -1 do
        flatIndex = flatIndex + (indices[i] - 1) * stride
        stride = stride * tensor.shape[i]
    end
    
    tensor.data[flatIndex + 1] = value
end

-- EINSTEIN SUMMATION - The Heart of Tensor Logic
--
-- Implements the einsum operation which performs:
-- 1. Outer product (Cartesian product) of input tensors
-- 2. Element-wise multiplication
-- 3. Summation over indices not in the output (the "contraction")
--
-- This is EXACTLY equivalent to:
-- - JOIN: the outer product aligns matching values
-- - PROJECT: summation removes the contracted indices
--
-- Example: einsum("xy,yz->xz", A, B) computes matrix multiplication
function M.einsum(notation, ...)
    local tensors = {...}
    
    -- Parse the einsum notation: "ij,jk->ik"
    local arrow_pos = notation:find('->')
    local inputPart = notation:sub(1, arrow_pos - 1)
    local outputIndices = notation:sub(arrow_pos + 2)
    
    local inputIndices = {}
    for idx in inputPart:gmatch('[^,]+') do
        table.insert(inputIndices, idx)
    end
    
    -- Build a map of all unique indices to their dimension sizes
    local indexSizes = {}
    for t = 1, #tensors do
        local indices = inputIndices[t]
        for i = 1, #indices do
            local idx = indices:sub(i, i)
            local size = tensors[t].shape[i]
            if indexSizes[idx] then
                if indexSizes[idx] ~= size then
                    error(string.format('Index %s has inconsistent sizes', idx))
                end
            else
                indexSizes[idx] = size
            end
        end
    end
    
    -- Determine output shape
    local outputShape = {}
    for i = 1, #outputIndices do
        local idx = outputIndices:sub(i, i)
        table.insert(outputShape, indexSizes[idx])
    end
    
    local outputSize = 1
    for _, dim in ipairs(outputShape) do
        outputSize = outputSize * dim
    end
    
    local outputData = {}
    for i = 1, outputSize do
        outputData[i] = 0
    end
    
    -- All indices (including contracted ones)
    local allIndices = {}
    local seen = {}
    for _, inp in ipairs(inputIndices) do
        for i = 1, #inp do
            local idx = inp:sub(i, i)
            if not seen[idx] then
                table.insert(allIndices, idx)
                seen[idx] = true
            end
        end
    end
    
    -- Compute strides for output tensor
    local outputStrides = {}
    local stride = 1
    for i = #outputIndices, 1, -1 do
        local idx = outputIndices:sub(i, i)
        outputStrides[idx] = stride
        stride = stride * outputShape[i]
    end
    
    -- Compute strides for input tensors
    local inputStrides = {}
    for t = 1, #tensors do
        local strides = {}
        local s = 1
        for i = #inputIndices[t], 1, -1 do
            local idx = inputIndices[t]:sub(i, i)
            strides[idx] = s
            s = s * tensors[t].shape[i]
        end
        table.insert(inputStrides, strides)
    end
    
    -- Generate all combinations of index values
    local indexRanges = {}
    for _, idx in ipairs(allIndices) do
        table.insert(indexRanges, indexSizes[idx])
    end
    
    -- Iterate over all possible index combinations
    local indexValues = {}
    for i = 1, #allIndices do
        indexValues[i] = 0
    end
    
    local function iterate(depth)
        if depth > #allIndices then
            -- Compute the product of all tensor elements at current indices
            local product = 1
            for t = 1, #tensors do
                local flatIndex = 0
                for i = 1, #inputIndices[t] do
                    local idx = inputIndices[t]:sub(i, i)
                    local idxPos = 0
                    for j = 1, #allIndices do
                        if allIndices[j] == idx then
                            idxPos = j
                            break
                        end
                    end
                    flatIndex = flatIndex + indexValues[idxPos] * (inputStrides[t][idx] or 0)
                end
                product = product * tensors[t].data[flatIndex + 1]
            end
            
            -- Add to output at appropriate position
            local outputFlatIndex = 0
            for i = 1, #outputIndices do
                local idx = outputIndices:sub(i, i)
                local idxPos = 0
                for j = 1, #allIndices do
                    if allIndices[j] == idx then
                        idxPos = j
                        break
                    end
                end
                outputFlatIndex = outputFlatIndex + indexValues[idxPos] * (outputStrides[idx] or 0)
            end
            outputData[outputFlatIndex + 1] = outputData[outputFlatIndex + 1] + product
            return
        end
        
        for v = 0, indexRanges[depth] - 1 do
            indexValues[depth] = v
            iterate(depth + 1)
        end
    end
    
    iterate(1)
    
    local outputIndicesList = {}
    for i = 1, #outputIndices do
        table.insert(outputIndicesList, outputIndices:sub(i, i))
    end
    
    return M.Tensor('result', outputShape, outputIndicesList, outputData)
end

-- LOGICAL OPERATIONS
--
-- In Boolean tensors (0/1 values), einsum becomes logical inference:
-- - Multiplication is AND
-- - Summation followed by threshold (>0) is OR

-- Apply threshold (for Boolean logic) - values > 0 become 1, else 0
function M.threshold(tensor, t)
    t = t or 0
    local data = {}
    for i = 1, #tensor.data do
        data[i] = tensor.data[i] > t and 1 or 0
    end
    return M.Tensor(tensor.name, tensor.shape, tensor.indices, data)
end

-- Sigmoid activation function - smooth version of threshold
-- σ(x, T) = 1 / (1 + e^(-x/T))
function M.sigmoid(tensor, temperature)
    temperature = temperature or 1
    local data = {}
    for i = 1, #tensor.data do
        if temperature == 0 then
            data[i] = tensor.data[i] > 0 and 1 or 0
        else
            data[i] = 1 / (1 + math.exp(-tensor.data[i] / temperature))
        end
    end
    return M.Tensor(tensor.name, tensor.shape, tensor.indices, data)
end

-- ReLU activation: max(0, x)
function M.relu(tensor)
    local data = {}
    for i = 1, #tensor.data do
        data[i] = math.max(0, tensor.data[i])
    end
    return M.Tensor(tensor.name, tensor.shape, tensor.indices, data)
end

-- Softmax: exp(x_i) / Σ exp(x_j)
-- Converts a vector of real numbers into a probability distribution
function M.softmax(tensor, axis)
    axis = axis or -1
    
    if #tensor.shape > 2 then
        error('Softmax currently supports 1D and 2D tensors')
    end
    
    local data = {}
    for i = 1, #tensor.data do
        data[i] = 0
    end
    
    if #tensor.shape == 1 then
        -- 1D tensor
        local max = -math.huge
        for i = 1, #tensor.data do
            max = math.max(max, tensor.data[i])
        end
        
        local sum = 0
        for i = 1, #tensor.data do
            data[i] = math.exp(tensor.data[i] - max)
            sum = sum + data[i]
        end
        
        for i = 1, #data do
            data[i] = data[i] / sum
        end
    else
        -- 2D tensor - apply along axis
        local actualAxis = axis == -1 and 2 or axis
        local rows, cols = tensor.shape[1], tensor.shape[2]
        
        if actualAxis == 2 then
            -- Softmax along columns (each row is independent)
            for i = 1, rows do
                local rowStart = (i - 1) * cols
                local max = -math.huge
                for j = 1, cols do
                    max = math.max(max, tensor.data[rowStart + j])
                end
                
                local sum = 0
                for j = 1, cols do
                    data[rowStart + j] = math.exp(tensor.data[rowStart + j] - max)
                    sum = sum + data[rowStart + j]
                end
                
                for j = 1, cols do
                    data[rowStart + j] = data[rowStart + j] / sum
                end
            end
        else
            -- Softmax along rows (each column is independent)
            for j = 1, cols do
                local max = -math.huge
                for i = 1, rows do
                    max = math.max(max, tensor.data[(i - 1) * cols + j])
                end
                
                local sum = 0
                for i = 1, rows do
                    local idx = (i - 1) * cols + j
                    data[idx] = math.exp(tensor.data[idx] - max)
                    sum = sum + data[idx]
                end
                
                for i = 1, rows do
                    local idx = (i - 1) * cols + j
                    data[idx] = data[idx] / sum
                end
            end
        end
    end
    
    return M.Tensor(tensor.name, tensor.shape, tensor.indices, data)
end

-- Element-wise addition of tensors
function M.add(...)
    local tensors = {...}
    local first = tensors[1]
    local data = {}
    
    for i = 1, #first.data do
        data[i] = 0
        for _, t in ipairs(tensors) do
            data[i] = data[i] + t.data[i]
        end
    end
    
    return M.Tensor(first.name, first.shape, first.indices, data)
end

-- Element-wise multiplication (Hadamard product)
function M.multiply(...)
    local tensors = {...}
    local first = tensors[1]
    local data = {}
    
    for i = 1, #first.data do
        data[i] = 1
        for _, t in ipairs(tensors) do
            data[i] = data[i] * t.data[i]
        end
    end
    
    return M.Tensor(first.name, first.shape, first.indices, data)
end

-- Scalar multiplication
function M.scale(tensor, scalar)
    local data = {}
    for i = 1, #tensor.data do
        data[i] = tensor.data[i] * scalar
    end
    return M.Tensor(tensor.name, tensor.shape, tensor.indices, data)
end

-- Transpose a 2D tensor
function M.transpose(tensor)
    if #tensor.shape ~= 2 then
        error('Transpose only works on 2D tensors')
    end
    
    local rows, cols = tensor.shape[1], tensor.shape[2]
    local data = {}
    
    for i = 1, rows do
        for j = 1, cols do
            data[(j - 1) * rows + i] = tensor.data[(i - 1) * cols + j]
        end
    end
    
    return M.Tensor(tensor.name, {cols, rows}, {tensor.indices[2], tensor.indices[1]}, data)
end

-- Convert tensor to human-readable string
function M.tensorToString(tensor, precision)
    precision = precision or 3
    local format = string.format('%%.%df', precision)
    
    if #tensor.shape == 1 then
        local parts = {}
        for _, v in ipairs(tensor.data) do
            table.insert(parts, string.format(format, v))
        end
        return '[' .. table.concat(parts, ', ') .. ']'
    end
    
    if #tensor.shape == 2 then
        local rows, cols = tensor.shape[1], tensor.shape[2]
        local lines = {}
        for i = 1, rows do
            local row = {}
            for j = 1, cols do
                table.insert(row, string.format(format, tensor.data[(i-1) * cols + j]))
            end
            table.insert(lines, '  [' .. table.concat(row, ', ') .. ']')
        end
        return '[\n' .. table.concat(lines, ',\n') .. '\n]'
    end
    
    -- For higher dimensions, show summary
    return string.format('Tensor(shape=[%s], indices=[%s])', 
        table.concat(tensor.shape, ', '), 
        table.concat(tensor.indices, ', '))
end

-- Clone a tensor
function M.clone(tensor)
    local data = {}
    for i, v in ipairs(tensor.data) do
        data[i] = v
    end
    return M.Tensor(tensor.name, tensor.shape, tensor.indices, data)
end

-- Create an identity matrix
function M.identity(name, indices, size)
    local data = {}
    for i = 1, size * size do
        data[i] = 0
    end
    for i = 1, size do
        data[(i - 1) * size + i] = 1
    end
    return M.Tensor(name, {size, size}, indices, data)
end

return M
