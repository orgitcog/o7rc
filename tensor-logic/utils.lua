--[[
Utility functions for tensor operations

These functions provide common patterns used across examples,
especially for operations that need manual computation when
einsum doesn't handle the exact pattern needed.
--]]

local core = require 'tensor-logic.core'
local M = {}

-- Broadcast-add a 1D tensor to each "row" of a higher-dimensional tensor.
-- 
-- Example: Add bias vector to each batch element
--   result[b, h] = tensor[b, h] + bias[h]
--
-- @param tensor - The tensor to add to (shape [..., dim])
-- @param bias - The 1D tensor to broadcast (shape [dim])
-- @param dimIndex - Which dimension index to broadcast along (default: last)
-- @returns New tensor with bias added
function M.broadcastAdd(tensor, bias, dimIndex)
    dimIndex = dimIndex or #tensor.shape
    
    if #bias.shape ~= 1 then
        error('bias must be 1D tensor')
    end
    if bias.shape[1] ~= tensor.shape[dimIndex] then
        error(string.format('bias dimension %d must match tensor dimension %d', 
            bias.shape[1], tensor.shape[dimIndex]))
    end
    
    local result = core.createTensor(
        tensor.name .. '_plus_bias',
        tensor.indices,
        tensor.shape,
        'zeros'
    )
    
    -- Compute flat index calculation
    local totalSize = 1
    for _, dim in ipairs(tensor.shape) do
        totalSize = totalSize * dim
    end
    local dimSize = tensor.shape[dimIndex]
    
    -- Stride for the dimension we're broadcasting along
    local stride = 1
    for i = #tensor.shape, dimIndex + 1, -1 do
        stride = stride * tensor.shape[i]
    end
    
    for i = 1, totalSize do
        -- Calculate which position in the broadcast dimension
        local dimPos = math.floor(((i - 1) / stride) % dimSize) + 1
        result.data[i] = tensor.data[i] + bias.data[dimPos]
    end
    
    return result
end

-- Element-wise multiply with broadcasting.
--
-- Example: Multiply each element by a 1D weight vector
--   result[n, l, d] = tensor[n, l, d] * weights[d]
--
-- @param tensor - The tensor to multiply
-- @param weights - The 1D tensor to broadcast multiply
-- @param dimIndex - Which dimension index to broadcast along (default: last)
-- @returns New tensor with element-wise multiplication
function M.broadcastMultiply(tensor, weights, dimIndex)
    dimIndex = dimIndex or #tensor.shape
    
    if #weights.shape ~= 1 then
        error('weights must be 1D tensor')
    end
    if weights.shape[1] ~= tensor.shape[dimIndex] then
        error(string.format('weights dimension %d must match tensor dimension %d',
            weights.shape[1], tensor.shape[dimIndex]))
    end
    
    local result = core.createTensor(
        tensor.name .. '_weighted',
        tensor.indices,
        tensor.shape,
        'zeros'
    )
    
    -- Compute flat index calculation
    local totalSize = 1
    for _, dim in ipairs(tensor.shape) do
        totalSize = totalSize * dim
    end
    local dimSize = tensor.shape[dimIndex]
    
    -- Stride for the dimension we're broadcasting along
    local stride = 1
    for i = #tensor.shape, dimIndex + 1, -1 do
        stride = stride * tensor.shape[i]
    end
    
    for i = 1, totalSize do
        -- Calculate which position in the broadcast dimension
        local dimPos = math.floor(((i - 1) / stride) % dimSize) + 1
        result.data[i] = tensor.data[i] * weights.data[dimPos]
    end
    
    return result
end

-- Extract a slice from a tensor along a specific dimension.
--
-- Example: Extract layer L from embeddings
--   result[n, d] = tensor[n, l=L, d]
--
-- @param tensor - The tensor to slice
-- @param dimIndex - Which dimension to slice
-- @param sliceIndex - Which slice to extract (1-based)
-- @returns New tensor with the specified slice
function M.extractSlice(tensor, dimIndex, sliceIndex)
    if sliceIndex < 1 or sliceIndex > tensor.shape[dimIndex] then
        error(string.format('sliceIndex %d out of bounds for dimension %d with size %d',
            sliceIndex, dimIndex, tensor.shape[dimIndex]))
    end
    
    -- New shape without the sliced dimension
    local newShape = {}
    for i, dim in ipairs(tensor.shape) do
        if i ~= dimIndex then
            table.insert(newShape, dim)
        end
    end
    
    -- New indices without the sliced dimension
    local newIndices = {}
    for i, idx in ipairs(tensor.indices) do
        if i ~= dimIndex then
            table.insert(newIndices, idx)
        end
    end
    
    local result = core.createTensor(
        tensor.name .. '_slice',
        newIndices,
        newShape,
        'zeros'
    )
    
    -- Calculate strides for the original tensor
    local strides = {}
    local stride = 1
    for i = #tensor.shape, 1, -1 do
        strides[i] = stride
        stride = stride * tensor.shape[i]
    end
    
    -- Iterate over all positions in the result tensor
    local resultSize = 1
    for _, dim in ipairs(newShape) do
        resultSize = resultSize * dim
    end
    
    for resultIdx = 0, resultSize - 1 do
        -- Convert result index to multi-dimensional coordinates
        local coords = {}
        local temp = resultIdx
        
        for i = #newShape, 1, -1 do
            coords[i] = temp % newShape[i]
            temp = math.floor(temp / newShape[i])
        end
        
        -- Map to original tensor coordinates (insert sliceIndex at dimIndex)
        local origCoords = {}
        local coordIdx = 1
        for i = 1, #tensor.shape do
            if i == dimIndex then
                origCoords[i] = sliceIndex - 1  -- Convert to 0-based
            else
                origCoords[i] = coords[coordIdx]
                coordIdx = coordIdx + 1
            end
        end
        
        -- Calculate flat index in original tensor
        local flatIdx = 0
        for i = 1, #tensor.shape do
            flatIdx = flatIdx + origCoords[i] * strides[i]
        end
        
        result.data[resultIdx + 1] = tensor.data[flatIdx + 1]
    end
    
    return result
end

return M
