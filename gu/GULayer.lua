-- ============================================================================
-- GULayer: Composite Layer for Geometric Unity Dynamics
-- ============================================================================
-- Combines the Swerve operator and Gauge transformer to implement a single
-- step of the GU field equations. This layer takes an ObserverseTensor as
-- input and produces an updated ObserverseTensor as output.
-- ============================================================================

local GULayer, parent = torch.class('gu.GULayer', 'nn.Module')

function GULayer:__init(fiber_dim, config)
    parent.__init(self)
    
    config = config or {}
    
    self.fiber_dim = fiber_dim
    self.base_dim = config.base_dim or 4
    self.use_swerve = config.use_swerve ~= false
    self.use_gauge = config.use_gauge ~= false
    self.use_residual = config.use_residual or false
    self.dropout_p = config.dropout or 0
    
    -- Swerve operator (computes Swervature)
    if self.use_swerve then
        self.swerve = gu.SwerveModule(fiber_dim, {
            use_torsion = config.use_torsion ~= false
        })
    end
    
    -- Gauge transformer
    if self.use_gauge then
        self.gauge = gu.GaugeTransformer(fiber_dim, {
            base_dim = self.base_dim,
            gauge_type = config.gauge_type or 'tilted',
            learnable = config.learnable_gauge ~= false
        })
    end
    
    -- Optional nonlinearity
    self.activation = config.activation
    if self.activation == 'tanh' then
        self.act_module = nn.Tanh()
    elseif self.activation == 'relu' then
        self.act_module = nn.ReLU()
    elseif self.activation == 'sigmoid' then
        self.act_module = nn.Sigmoid()
    end
    
    -- Optional dropout
    if self.dropout_p > 0 then
        self.dropout = nn.Dropout(self.dropout_p)
    end
    
    -- Layer normalization (optional)
    self.use_layernorm = config.use_layernorm or false
    if self.use_layernorm then
        self.layernorm_fiber = nn.LayerNormalization(fiber_dim)
        self.layernorm_base = nn.LayerNormalization(self.base_dim)
    end
end

function GULayer:updateOutput(input)
    assert(type(input) == 'table' and input._type == 'ObserverseTensor',
        "GULayer expects an ObserverseTensor as input")
    
    local current = input
    
    -- Store input for residual connection
    local residual_base = input.base:clone()
    local residual_fiber = input.fiber:clone()
    
    -- Apply Swerve operator
    if self.use_swerve then
        current = self.swerve:forward(current)
    end
    
    -- Apply activation
    if self.act_module then
        local ObserverseTensor = require 'gu.ObserverseTensor'
        local activated_fiber = self.act_module:forward(current.fiber)
        current = ObserverseTensor.create(current.base, activated_fiber)
    end
    
    -- Apply dropout
    if self.dropout and self.train then
        local ObserverseTensor = require 'gu.ObserverseTensor'
        local dropped_fiber = self.dropout:forward(current.fiber)
        current = ObserverseTensor.create(current.base, dropped_fiber)
    end
    
    -- Apply gauge transformation
    if self.use_gauge then
        current = self.gauge:forward(current)
    end
    
    -- Apply residual connection
    if self.use_residual then
        local ObserverseTensor = require 'gu.ObserverseTensor'
        local new_base = current.base + residual_base
        local new_fiber = current.fiber + residual_fiber
        current = ObserverseTensor.create(new_base, new_fiber)
    end
    
    -- Apply layer normalization
    if self.use_layernorm then
        local ObserverseTensor = require 'gu.ObserverseTensor'
        local normed_base = self.layernorm_base:forward(current.base)
        local normed_fiber = self.layernorm_fiber:forward(current.fiber)
        current = ObserverseTensor.create(normed_base, normed_fiber)
    end
    
    self.output = current
    return self.output
end

function GULayer:updateGradInput(input, gradOutput)
    local current_grad = gradOutput
    
    -- Backprop through layer normalization
    if self.use_layernorm then
        local ObserverseTensor = require 'gu.ObserverseTensor'
        -- Note: This is simplified; full implementation would track intermediate states
        local grad_base = self.layernorm_base:backward(self.output.base, current_grad.base)
        local grad_fiber = self.layernorm_fiber:backward(self.output.fiber, current_grad.fiber)
        current_grad = ObserverseTensor.create(grad_base, grad_fiber)
    end
    
    -- Residual connection gradient passes through
    local residual_grad
    if self.use_residual then
        residual_grad = current_grad
    end
    
    -- Backprop through gauge transformation
    if self.use_gauge then
        current_grad = self.gauge:backward(input, current_grad)
    end
    
    -- Backprop through dropout
    if self.dropout and self.train then
        local ObserverseTensor = require 'gu.ObserverseTensor'
        local grad_fiber = self.dropout:backward(input.fiber, current_grad.fiber)
        current_grad = ObserverseTensor.create(current_grad.base, grad_fiber)
    end
    
    -- Backprop through activation
    if self.act_module then
        local ObserverseTensor = require 'gu.ObserverseTensor'
        local grad_fiber = self.act_module:backward(input.fiber, current_grad.fiber)
        current_grad = ObserverseTensor.create(current_grad.base, grad_fiber)
    end
    
    -- Backprop through Swerve operator
    if self.use_swerve then
        current_grad = self.swerve:backward(input, current_grad)
    end
    
    -- Add residual gradient
    if self.use_residual and residual_grad then
        local ObserverseTensor = require 'gu.ObserverseTensor'
        local new_base = current_grad.base + residual_grad.base
        local new_fiber = current_grad.fiber + residual_grad.fiber
        current_grad = ObserverseTensor.create(new_base, new_fiber)
    end
    
    self.gradInput = current_grad
    return self.gradInput
end

function GULayer:accGradParameters(input, gradOutput, scale)
    scale = scale or 1
    
    -- Accumulate gradients for Swerve
    if self.use_swerve then
        self.swerve:accGradParameters(input, gradOutput, scale)
    end
    
    -- Accumulate gradients for Gauge
    if self.use_gauge then
        self.gauge:accGradParameters(input, gradOutput, scale)
    end
end

function GULayer:parameters()
    local params = {}
    local gradParams = {}
    
    if self.use_swerve then
        local p, gp = self.swerve:parameters()
        for i, v in ipairs(p) do table.insert(params, v) end
        for i, v in ipairs(gp) do table.insert(gradParams, v) end
    end
    
    if self.use_gauge then
        local p, gp = self.gauge:parameters()
        for i, v in ipairs(p) do table.insert(params, v) end
        for i, v in ipairs(gp) do table.insert(gradParams, v) end
    end
    
    return params, gradParams
end

function GULayer:training()
    self.train = true
    if self.swerve then self.swerve:training() end
    if self.gauge then self.gauge:training() end
    if self.dropout then self.dropout:training() end
end

function GULayer:evaluate()
    self.train = false
    if self.swerve then self.swerve:evaluate() end
    if self.gauge then self.gauge:evaluate() end
    if self.dropout then self.dropout:evaluate() end
end

function GULayer:__tostring()
    local str = string.format('%s(fiber_dim=%d, base_dim=%d',
        torch.type(self), self.fiber_dim, self.base_dim)
    if self.use_swerve then str = str .. ', swerve' end
    if self.use_gauge then str = str .. ', gauge' end
    if self.use_residual then str = str .. ', residual' end
    if self.activation then str = str .. ', ' .. self.activation end
    str = str .. ')'
    return str
end

return GULayer
