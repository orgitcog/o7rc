-- ============================================================================
-- ParallelTransportActivation: Geometry-Respecting Fiber-Base Coupling
-- ============================================================================
-- This activation respects the parallel transport geometry of the Observerse,
-- coupling base and fiber components according to connection coefficients.
--
-- Key Features:
--   - Implements parallel transport along base space directions
--   - Couples fiber transformations to base space geometry
--   - Learnable connection coefficients (Christoffel-like symbols)
--   - Preserves geometric structure of the two-space
--   - Supports holonomy-aware non-linear transformations
--
-- Mathematical Formulation:
--   Given ObserverseTensor (base, fiber) and connection Gamma:
--   1. Compute transport direction from base: v = base / ||base||
--   2. Compute infinitesimal transport: delta = Gamma^i_jk * v^j * fiber^k
--   3. Apply non-linear activation: fiber' = activation(fiber + delta)
--   4. Optionally update base based on fiber feedback
-- ============================================================================

local ParallelTransportActivation, parent = torch.class('gu.activations.ParallelTransportActivation', 'nn.Module')

-- ============================================================================
-- Initialization
-- ============================================================================

function ParallelTransportActivation:__init(fiber_dim, config)
    parent.__init(self)

    config = config or {}

    self.fiber_dim = fiber_dim
    self.base_dim = config.base_dim or 4
    self.activation_type = config.activation_type or 'tanh'  -- 'tanh', 'relu', 'elu', 'gelu', 'swish'
    self.transport_strength = config.transport_strength or 1.0
    self.bidirectional = config.bidirectional or false  -- Also update base from fiber
    self.use_holonomy = config.use_holonomy or false  -- Track holonomy effects

    -- Connection coefficients: Gamma^i_jk
    -- i: fiber index (output), j: base index (direction), k: fiber index (input)
    -- Simplified to Gamma[j][i][k] for base direction j
    self.connection = torch.Tensor(self.base_dim, self.fiber_dim, self.fiber_dim)
    self.gradConnection = torch.Tensor(self.base_dim, self.fiber_dim, self.fiber_dim)

    -- Optional: reverse connection for bidirectional transport
    if self.bidirectional then
        self.reverse_connection = torch.Tensor(self.fiber_dim, self.base_dim, self.base_dim)
        self.gradReverseConnection = torch.Tensor(self.fiber_dim, self.base_dim, self.base_dim)
    end

    -- Learnable activation parameters
    self.act_scale = torch.Tensor(self.fiber_dim)
    self.gradActScale = torch.Tensor(self.fiber_dim)

    -- For holonomy tracking
    if self.use_holonomy then
        self.holonomy_matrix = torch.eye(self.fiber_dim)
    end

    -- Initialize
    self:reset()
end

-- ============================================================================
-- Activation Functions
-- ============================================================================

function ParallelTransportActivation:_applyActivation(x)
    local result

    if self.activation_type == 'tanh' then
        result = torch.tanh(x)
    elseif self.activation_type == 'relu' then
        result = torch.cmax(x, 0)
    elseif self.activation_type == 'elu' then
        result = x:clone()
        local mask = x:lt(0)
        if mask:any() then
            result[mask] = torch.exp(x[mask]) - 1
        end
    elseif self.activation_type == 'gelu' then
        -- GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        local sqrt_2_pi = math.sqrt(2 / math.pi)
        local inner = (x + 0.044715 * torch.pow(x, 3)):mul(sqrt_2_pi)
        result = x:clone():cmul((torch.tanh(inner) + 1):mul(0.5))
    elseif self.activation_type == 'swish' then
        result = x:clone():cmul(torch.sigmoid(x))
    elseif self.activation_type == 'softplus' then
        result = torch.log(torch.exp(x) + 1)
    else
        -- Identity
        result = x:clone()
    end

    return result
end

function ParallelTransportActivation:_activationGrad(x, gradOutput)
    local grad

    if self.activation_type == 'tanh' then
        local tanh_x = torch.tanh(x)
        grad = torch.cmul(1 - torch.cmul(tanh_x, tanh_x), gradOutput)
    elseif self.activation_type == 'relu' then
        grad = gradOutput:clone()
        grad[x:lt(0)] = 0
    elseif self.activation_type == 'elu' then
        grad = gradOutput:clone()
        local mask = x:lt(0)
        if mask:any() then
            grad[mask] = torch.cmul(torch.exp(x[mask]), gradOutput[mask])
        end
    elseif self.activation_type == 'gelu' then
        -- Approximate derivative
        local sqrt_2_pi = math.sqrt(2 / math.pi)
        local x3 = torch.pow(x, 3)
        local inner = (x + 0.044715 * x3):mul(sqrt_2_pi)
        local tanh_inner = torch.tanh(inner)
        local sech2 = 1 - torch.cmul(tanh_inner, tanh_inner)
        local d_inner = (1 + 0.134145 * torch.pow(x, 2)):mul(sqrt_2_pi)
        grad = torch.cmul(0.5 * (1 + tanh_inner) + 0.5 * torch.cmul(x, torch.cmul(sech2, d_inner)), gradOutput)
    elseif self.activation_type == 'swish' then
        local sig_x = torch.sigmoid(x)
        local swish = torch.cmul(x, sig_x)
        grad = torch.cmul(swish + torch.cmul(sig_x, 1 - swish), gradOutput)
    elseif self.activation_type == 'softplus' then
        grad = torch.cmul(torch.sigmoid(x), gradOutput)
    else
        grad = gradOutput:clone()
    end

    return grad
end

-- ============================================================================
-- Compute Parallel Transport
-- ============================================================================

function ParallelTransportActivation:_parallelTransport(fiber, base)
    -- Normalize base to get direction
    local base_norm = base:norm() + 1e-8
    local direction = base / base_norm

    -- Compute transport correction: delta^i = Gamma^i_jk * v^j * fiber^k
    local delta = torch.zeros(self.fiber_dim)

    for j = 1, self.base_dim do
        local v_j = direction[j]
        -- Gamma[j] is a fiber_dim x fiber_dim matrix
        local Gamma_j = self.connection[j]
        -- delta += v_j * Gamma_j * fiber
        delta:add(v_j, torch.mv(Gamma_j, fiber))
    end

    return delta * self.transport_strength
end

-- ============================================================================
-- Compute Reverse Transport (Base from Fiber)
-- ============================================================================

function ParallelTransportActivation:_reverseTransport(fiber, base)
    if not self.bidirectional then
        return torch.zeros(self.base_dim)
    end

    local fiber_norm = fiber:norm() + 1e-8
    local direction = fiber / fiber_norm

    local delta = torch.zeros(self.base_dim)

    for j = 1, self.fiber_dim do
        local v_j = direction[j]
        local Gamma_rev_j = self.reverse_connection[j]
        delta:add(v_j, torch.mv(Gamma_rev_j, base))
    end

    return delta * self.transport_strength
end

-- ============================================================================
-- Update Holonomy
-- ============================================================================

function ParallelTransportActivation:_updateHolonomy(base)
    if not self.use_holonomy then
        return
    end

    -- Approximate holonomy update: H' = exp(Gamma_j * v^j) * H
    local base_norm = base:norm() + 1e-8
    local direction = base / base_norm

    -- Build infinitesimal generator
    local A = torch.zeros(self.fiber_dim, self.fiber_dim)
    for j = 1, self.base_dim do
        A:add(direction[j], self.connection[j])
    end

    -- Simple approximation: H' = (I + epsilon * A) * H
    local epsilon = 0.01
    local I = torch.eye(self.fiber_dim)
    local update = I + A:mul(epsilon)
    self.holonomy_matrix = torch.mm(update, self.holonomy_matrix)
end

-- ============================================================================
-- Forward Pass
-- ============================================================================

function ParallelTransportActivation:updateOutput(input)
    assert(type(input) == 'table' and input._type == 'ObserverseTensor',
        "ParallelTransportActivation requires ObserverseTensor input")

    local base_tensor = input.base
    local fiber_tensor = input.fiber

    local base_was_1d = base_tensor:dim() == 1
    local fiber_was_1d = fiber_tensor:dim() == 1

    local base_view = base_was_1d and base_tensor:view(1, -1) or base_tensor
    local fiber_view = fiber_was_1d and fiber_tensor:view(1, -1) or fiber_tensor

    local batch_size = fiber_view:size(1)

    -- Store for backward
    self._base_view = base_view
    self._fiber_view = fiber_view
    self._batch_size = batch_size
    self._base_was_1d = base_was_1d
    self._fiber_was_1d = fiber_was_1d

    -- Output tensors
    local output_fiber = torch.zeros(batch_size, self.fiber_dim)
    local output_base = base_view:clone()

    -- Store intermediate values
    self._transport_deltas = torch.zeros(batch_size, self.fiber_dim)
    self._pre_activation = torch.zeros(batch_size, self.fiber_dim)

    for b = 1, batch_size do
        local base = base_view[b]
        local fiber = fiber_view[b]

        -- Compute parallel transport correction
        local delta = self:_parallelTransport(fiber, base)
        self._transport_deltas[b] = delta

        -- Add transport to fiber
        local transported = fiber + delta
        self._pre_activation[b] = transported

        -- Apply activation with learned scale
        local activated = self:_applyActivation(transported)
        for i = 1, self.fiber_dim do
            activated[i] = activated[i] * self.act_scale[i]
        end
        output_fiber[b] = activated

        -- Bidirectional: update base from fiber
        if self.bidirectional then
            local base_delta = self:_reverseTransport(fiber, base)
            output_base[b] = base + base_delta
        end

        -- Update holonomy tracking
        self:_updateHolonomy(base)
    end

    -- Reshape
    if fiber_was_1d then
        output_fiber = output_fiber:view(self.fiber_dim)
    end
    if base_was_1d then
        output_base = output_base:view(self.base_dim)
    end

    local ObserverseTensor = require 'gu.ObserverseTensor'
    self.output = ObserverseTensor.create(output_base, output_fiber)

    return self.output
end

-- ============================================================================
-- Backward Pass
-- ============================================================================

function ParallelTransportActivation:updateGradInput(input, gradOutput)
    local gradOutput_base = gradOutput.base
    local gradOutput_fiber = gradOutput.fiber

    local gradBase_view = self._base_was_1d and gradOutput_base:view(1, -1) or gradOutput_base
    local gradFiber_view = self._fiber_was_1d and gradOutput_fiber:view(1, -1) or gradOutput_fiber

    local gradInput_fiber = torch.zeros(self._batch_size, self.fiber_dim)
    local gradInput_base = torch.zeros(self._batch_size, self.base_dim)

    for b = 1, self._batch_size do
        -- Gradient through activation and scale
        local grad_scaled = gradFiber_view[b]:clone()
        for i = 1, self.fiber_dim do
            grad_scaled[i] = grad_scaled[i] * self.act_scale[i]
        end

        local grad_act = self:_activationGrad(self._pre_activation[b], grad_scaled)

        -- Gradient through transport: d(fiber + delta)/d_fiber = I + d_delta/d_fiber
        -- d_delta/d_fiber is the connection matrix
        gradInput_fiber[b] = grad_act:clone()

        -- Add gradient through connection
        for j = 1, self.base_dim do
            local base = self._base_view[b]
            local base_norm = base:norm() + 1e-8
            local v_j = base[j] / base_norm
            local Gamma_j = self.connection[j]
            -- d_delta/d_fiber += v_j * Gamma_j^T
            gradInput_fiber[b]:add(v_j * self.transport_strength, torch.mv(Gamma_j:t(), grad_act))
        end

        -- Gradient to base (through direction normalization)
        if self.bidirectional then
            gradInput_base[b] = gradBase_view[b]:clone()
        end
    end

    -- Reshape
    if self._fiber_was_1d then
        gradInput_fiber = gradInput_fiber:view(self.fiber_dim)
    end
    if self._base_was_1d then
        gradInput_base = gradInput_base:view(self.base_dim)
    end

    local ObserverseTensor = require 'gu.ObserverseTensor'
    self.gradInput = ObserverseTensor.create(gradInput_base, gradInput_fiber)

    return self.gradInput
end

-- ============================================================================
-- Accumulate Gradients
-- ============================================================================

function ParallelTransportActivation:accGradParameters(input, gradOutput, scale)
    scale = scale or 1

    local gradOutput_fiber = gradOutput.fiber
    local gradFiber_view = self._fiber_was_1d and gradOutput_fiber:view(1, -1) or gradOutput_fiber

    -- Gradient for connection coefficients
    for b = 1, self._batch_size do
        local base = self._base_view[b]
        local fiber = self._fiber_view[b]
        local base_norm = base:norm() + 1e-8

        -- Gradient through activation
        local grad_scaled = gradFiber_view[b]:clone()
        for i = 1, self.fiber_dim do
            grad_scaled[i] = grad_scaled[i] * self.act_scale[i]
        end
        local grad_act = self:_activationGrad(self._pre_activation[b], grad_scaled)

        -- d_loss/d_Gamma^i_jk = grad_act^i * v^j * fiber^k * transport_strength
        for j = 1, self.base_dim do
            local v_j = base[j] / base_norm
            for i = 1, self.fiber_dim do
                for k = 1, self.fiber_dim do
                    self.gradConnection[j][i][k] = self.gradConnection[j][i][k] +
                        scale * grad_act[i] * v_j * fiber[k] * self.transport_strength
                end
            end
        end

        -- Gradient for activation scale
        local activated = self:_applyActivation(self._pre_activation[b])
        for i = 1, self.fiber_dim do
            self.gradActScale[i] = self.gradActScale[i] +
                scale * gradFiber_view[b][i] * activated[i]
        end
    end
end

-- ============================================================================
-- Parameter Initialization
-- ============================================================================

function ParallelTransportActivation:reset()
    -- Initialize connection coefficients near zero (identity transport)
    local stdv = 0.01 / math.sqrt(self.fiber_dim)
    self.connection:normal(0, stdv)

    if self.bidirectional then
        self.reverse_connection:normal(0, stdv)
    end

    -- Initialize activation scale to 1
    self.act_scale:fill(1.0)

    -- Reset holonomy
    if self.use_holonomy then
        self.holonomy_matrix = torch.eye(self.fiber_dim)
    end

    self:zeroGradParameters()
end

function ParallelTransportActivation:zeroGradParameters()
    self.gradConnection:zero()
    self.gradActScale:zero()

    if self.bidirectional then
        self.gradReverseConnection:zero()
    end
end

-- ============================================================================
-- Parameters
-- ============================================================================

function ParallelTransportActivation:parameters()
    local params = {self.connection, self.act_scale}
    local gradParams = {self.gradConnection, self.gradActScale}

    if self.bidirectional then
        table.insert(params, self.reverse_connection)
        table.insert(gradParams, self.gradReverseConnection)
    end

    return params, gradParams
end

-- ============================================================================
-- Utility Methods
-- ============================================================================

function ParallelTransportActivation:getHolonomy()
    return self.holonomy_matrix
end

function ParallelTransportActivation:resetHolonomy()
    if self.use_holonomy then
        self.holonomy_matrix = torch.eye(self.fiber_dim)
    end
end

function ParallelTransportActivation:getConnection()
    return self.connection
end

function ParallelTransportActivation:__tostring()
    local str = string.format('%s(fiber=%d, base=%d, act=%s',
        torch.type(self), self.fiber_dim, self.base_dim, self.activation_type)
    if self.bidirectional then str = str .. ', bidirectional' end
    if self.use_holonomy then str = str .. ', holonomy' end
    str = str .. ')'
    return str
end

return ParallelTransportActivation
