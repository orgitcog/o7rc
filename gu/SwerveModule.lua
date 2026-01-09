-- ============================================================================
-- SwerveModule: The Swerve Operator for Computing Swervature
-- ============================================================================
-- The Swerve is a specific instance of the Shiab operator that computes
-- the Swervature tensor - a central component of the GU field equations.
--
-- The GU field equations take the form:
--   Shiab_{ε,σ}(F_A) + ⋆(Augmented Torsion) = 0
--
-- The Swervature is the exact tensor produced by the Swerve operator.
-- ============================================================================

local SwerveModule, parent = torch.class('gu.SwerveModule', 'nn.Module')

function SwerveModule:__init(dim, config)
    parent.__init(self)
    
    config = config or {}
    
    self.dim = dim
    self.use_torsion = config.use_torsion ~= false  -- Include torsion term
    
    -- The Swerve operator combines the Shiab operator with torsion
    -- Swerve(F) = Shiab(F) + ⋆T (where T is augmented torsion)
    
    -- Main Shiab component
    self.shiab = gu.ShiabOperator(dim, dim, config)
    
    -- Torsion component (learnable adjustment)
    if self.use_torsion then
        self.torsion_weight = torch.Tensor(dim, dim)
        self.gradTorsionWeight = torch.Tensor(dim, dim)
        self.torsion_weight:eye(dim):mul(0.1)  -- Small initial torsion
    end
    
    -- Hodge star operator approximation (for ⋆T term)
    -- In full implementation, this would be the actual Hodge dual
    self.hodge_weight = torch.Tensor(dim, dim)
    self.gradHodgeWeight = torch.Tensor(dim, dim)
    self.hodge_weight:eye(dim)  -- Initialize as identity
end

function SwerveModule:updateOutput(input)
    local input_tensor
    local is_observerse = type(input) == 'table' and input._type == 'ObserverseTensor'
    
    if is_observerse then
        input_tensor = input.fiber
    else
        input_tensor = input
    end
    
    -- Ensure 2D
    local was_1d = input_tensor:dim() == 1
    local input_view = was_1d and input_tensor:view(1, -1) or input_tensor
    
    -- Apply Shiab operator
    local shiab_out
    if is_observerse then
        local shiab_result = self.shiab:forward(input)
        shiab_out = shiab_result.fiber
        if was_1d then
            shiab_out = shiab_out:view(1, -1)
        end
    else
        shiab_out = self.shiab:forward(input_view)
    end
    
    -- Add torsion term: ⋆T
    if self.use_torsion then
        -- Compute torsion contribution
        local torsion_term = torch.mm(input_view, self.torsion_weight:t())
        -- Apply Hodge star
        local hodge_torsion = torch.mm(torsion_term, self.hodge_weight:t())
        -- Combine: Swervature = Shiab(F) + ⋆T
        self.output = shiab_out + hodge_torsion
    else
        self.output = shiab_out
    end
    
    -- Reshape if needed
    if was_1d then
        self.output = self.output:view(self.dim)
    end
    
    -- Return ObserverseTensor if input was ObserverseTensor
    if is_observerse then
        local ObserverseTensor = require 'gu.ObserverseTensor'
        self.output = ObserverseTensor.create(input.base:clone(), self.output)
    end
    
    return self.output
end

function SwerveModule:updateGradInput(input, gradOutput)
    local input_tensor, gradOutput_tensor
    local is_observerse = type(input) == 'table' and input._type == 'ObserverseTensor'
    
    if is_observerse then
        input_tensor = input.fiber
        gradOutput_tensor = gradOutput.fiber
    else
        input_tensor = input
        gradOutput_tensor = gradOutput
    end
    
    -- Ensure 2D
    local was_1d = input_tensor:dim() == 1
    local gradOutput_view = was_1d and gradOutput_tensor:view(1, -1) or gradOutput_tensor
    
    -- Backprop through Shiab
    local shiab_grad
    if is_observerse then
        local shiab_gradInput = self.shiab:backward(input, gradOutput)
        shiab_grad = shiab_gradInput.fiber
        if was_1d then
            shiab_grad = shiab_grad:view(1, -1)
        end
    else
        shiab_grad = self.shiab:backward(input_tensor, gradOutput_tensor)
        if was_1d then
            shiab_grad = shiab_grad:view(1, -1)
        end
    end
    
    -- Backprop through torsion term
    if self.use_torsion then
        local grad_hodge = torch.mm(gradOutput_view, self.hodge_weight)
        local grad_torsion = torch.mm(grad_hodge, self.torsion_weight)
        self.gradInput = shiab_grad + grad_torsion
    else
        self.gradInput = shiab_grad
    end
    
    -- Reshape if needed
    if was_1d then
        self.gradInput = self.gradInput:view(self.dim)
    end
    
    -- Return ObserverseTensor if input was ObserverseTensor
    if is_observerse then
        local ObserverseTensor = require 'gu.ObserverseTensor'
        local base_grad = torch.zeros(input.base:size())
        self.gradInput = ObserverseTensor.create(base_grad, self.gradInput)
    end
    
    return self.gradInput
end

function SwerveModule:accGradParameters(input, gradOutput, scale)
    scale = scale or 1
    
    local input_tensor, gradOutput_tensor
    local is_observerse = type(input) == 'table' and input._type == 'ObserverseTensor'
    
    if is_observerse then
        input_tensor = input.fiber
        gradOutput_tensor = gradOutput.fiber
    else
        input_tensor = input
        gradOutput_tensor = gradOutput
    end
    
    -- Ensure 2D
    local was_1d = input_tensor:dim() == 1
    local input_view = was_1d and input_tensor:view(1, -1) or input_tensor
    local gradOutput_view = was_1d and gradOutput_tensor:view(1, -1) or gradOutput_tensor
    
    -- Accumulate gradients for Shiab
    self.shiab:accGradParameters(input, gradOutput, scale)
    
    -- Accumulate gradients for torsion
    if self.use_torsion then
        -- Gradient for hodge_weight
        local torsion_term = torch.mm(input_view, self.torsion_weight:t())
        self.gradHodgeWeight:addmm(scale, gradOutput_view:t(), torsion_term)
        
        -- Gradient for torsion_weight
        local grad_hodge = torch.mm(gradOutput_view, self.hodge_weight)
        self.gradTorsionWeight:addmm(scale, grad_hodge:t(), input_view)
    end
end

function SwerveModule:parameters()
    local params, gradParams = self.shiab:parameters()
    
    if self.use_torsion then
        table.insert(params, self.torsion_weight)
        table.insert(params, self.hodge_weight)
        table.insert(gradParams, self.gradTorsionWeight)
        table.insert(gradParams, self.gradHodgeWeight)
    end
    
    return params, gradParams
end

function SwerveModule:__tostring()
    return string.format('%s(dim=%d, torsion=%s)', 
        torch.type(self), self.dim, tostring(self.use_torsion))
end

return SwerveModule
