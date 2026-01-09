-- ============================================================================
-- ShiabOperator: Ship-in-a-Bottle Operator for Geometric Unity
-- ============================================================================
-- The Shiab operator is a family of differential operators that acts on forms
-- valued in the adjoint bundle. It maps ad-valued k-forms to ad-valued 
-- (k+2)-forms. The key property is that it respects gauge transformations
-- in a way that allows construction of gauge-invariant quantities.
--
-- The name "Shiab" derives from "ship in a bottle" - the operator constructs
-- complex gauge-invariant structures from simpler components.
-- ============================================================================

local ShiabOperator, parent = torch.class('gu.ShiabOperator', 'nn.Module')

function ShiabOperator:__init(input_dim, output_dim, config)
    parent.__init(self)
    
    config = config or {}
    
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.use_bias = config.use_bias ~= false  -- Default true
    
    -- The Shiab operator is implemented as a learnable transformation
    -- In the full theory, this would encode the specific structure of
    -- the operator acting on differential forms
    
    -- Weight matrix for the transformation
    self.weight = torch.Tensor(output_dim, input_dim)
    self.gradWeight = torch.Tensor(output_dim, input_dim)
    
    if self.use_bias then
        self.bias = torch.Tensor(output_dim)
        self.gradBias = torch.Tensor(output_dim)
    end
    
    -- Additional structure for gauge covariance
    -- This encodes how the operator transforms under gauge transformations
    self.gauge_weight = torch.Tensor(output_dim, output_dim)
    self.gradGaugeWeight = torch.Tensor(output_dim, output_dim)
    
    self:reset()
end

function ShiabOperator:reset(stdv)
    if stdv then
        stdv = stdv * math.sqrt(3)
    else
        stdv = 1.0 / math.sqrt(self.input_dim)
    end
    
    self.weight:uniform(-stdv, stdv)
    self.gauge_weight:eye(self.output_dim)  -- Initialize as identity
    
    if self.bias then
        self.bias:zero()
    end
end

function ShiabOperator:updateOutput(input)
    -- Handle both tensor and ObserverseTensor inputs
    local input_tensor
    if type(input) == 'table' and input._type == 'ObserverseTensor' then
        -- Apply to fiber component (where the gauge fields live)
        input_tensor = input.fiber
    else
        input_tensor = input
    end
    
    -- Ensure input is 2D (batch_size x input_dim)
    local nframe = 1
    local input_view = input_tensor
    if input_tensor:dim() == 1 then
        nframe = 1
        input_view = input_tensor:view(1, input_tensor:size(1))
    else
        nframe = input_tensor:size(1)
        input_view = input_tensor
    end
    
    -- Apply the Shiab transformation
    -- Shiab(F) = W * F + b (simplified linear approximation)
    self.output = torch.mm(input_view, self.weight:t())
    
    if self.bias then
        self.output:add(self.bias:view(1, self.output_dim):expandAs(self.output))
    end
    
    -- Apply gauge structure
    self.output = torch.mm(self.output, self.gauge_weight:t())
    
    -- Reshape if input was 1D
    if input_tensor:dim() == 1 then
        self.output = self.output:view(self.output_dim)
    end
    
    -- If input was ObserverseTensor, return ObserverseTensor
    if type(input) == 'table' and input._type == 'ObserverseTensor' then
        local ObserverseTensor = require 'gu.ObserverseTensor'
        -- Keep base unchanged, transform fiber
        self.output = ObserverseTensor.create(input.base:clone(), self.output)
    end
    
    return self.output
end

function ShiabOperator:updateGradInput(input, gradOutput)
    local input_tensor, gradOutput_tensor
    
    if type(input) == 'table' and input._type == 'ObserverseTensor' then
        input_tensor = input.fiber
        gradOutput_tensor = gradOutput.fiber
    else
        input_tensor = input
        gradOutput_tensor = gradOutput
    end
    
    -- Ensure proper dimensions
    local gradOutput_view = gradOutput_tensor
    if gradOutput_tensor:dim() == 1 then
        gradOutput_view = gradOutput_tensor:view(1, gradOutput_tensor:size(1))
    end
    
    -- Backprop through gauge structure
    local grad_after_gauge = torch.mm(gradOutput_view, self.gauge_weight)
    
    -- Backprop through main transformation
    self.gradInput = torch.mm(grad_after_gauge, self.weight)
    
    -- Reshape if needed
    if input_tensor:dim() == 1 then
        self.gradInput = self.gradInput:view(self.input_dim)
    end
    
    -- If input was ObserverseTensor, return ObserverseTensor gradient
    if type(input) == 'table' and input._type == 'ObserverseTensor' then
        local ObserverseTensor = require 'gu.ObserverseTensor'
        -- Gradient for base is zero (not transformed), gradient for fiber is computed
        local base_grad = torch.zeros(input.base:size())
        self.gradInput = ObserverseTensor.create(base_grad, self.gradInput)
    end
    
    return self.gradInput
end

function ShiabOperator:accGradParameters(input, gradOutput, scale)
    scale = scale or 1
    
    local input_tensor, gradOutput_tensor
    
    if type(input) == 'table' and input._type == 'ObserverseTensor' then
        input_tensor = input.fiber
        gradOutput_tensor = gradOutput.fiber
    else
        input_tensor = input
        gradOutput_tensor = gradOutput
    end
    
    -- Ensure proper dimensions
    local input_view = input_tensor
    local gradOutput_view = gradOutput_tensor
    
    if input_tensor:dim() == 1 then
        input_view = input_tensor:view(1, input_tensor:size(1))
    end
    if gradOutput_tensor:dim() == 1 then
        gradOutput_view = gradOutput_tensor:view(1, gradOutput_tensor:size(1))
    end
    
    -- Gradient for gauge_weight
    local grad_after_gauge = torch.mm(gradOutput_view, self.gauge_weight)
    self.gradGaugeWeight:addmm(scale, gradOutput_view:t(), 
        torch.mm(input_view, self.weight:t()))
    
    -- Gradient for weight
    self.gradWeight:addmm(scale, grad_after_gauge:t(), input_view)
    
    -- Gradient for bias
    if self.bias then
        self.gradBias:add(scale, grad_after_gauge:sum(1):view(self.output_dim))
    end
end

function ShiabOperator:parameters()
    if self.bias then
        return {self.weight, self.bias, self.gauge_weight}, 
               {self.gradWeight, self.gradBias, self.gradGaugeWeight}
    else
        return {self.weight, self.gauge_weight}, 
               {self.gradWeight, self.gradGaugeWeight}
    end
end

function ShiabOperator:__tostring()
    return string.format('%s(%d -> %d, gauge_dim=%d)', 
        torch.type(self), self.input_dim, self.output_dim, self.output_dim)
end

return ShiabOperator
