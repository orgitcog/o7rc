-- ============================================================================
-- LieAlgebraActivation: Lie Group Structure-Preserving Activation
-- ============================================================================
-- This activation function preserves Lie group structure during gauge
-- transformations by applying learned non-linearities in the Lie algebra
-- before exponentiating to the group.
--
-- Key Features:
--   - Applies activation in Lie algebra space (tangent space at identity)
--   - Preserves group structure through exponential map
--   - Supports SO(n), SU(n), Spin(n), U(n) structure groups
--   - Learnable pre-exponential transformation
--   - Smooth interpolation on the group manifold
--
-- Mathematical Formulation:
--   Given input x and Lie algebra parameters theta:
--   1. Map input to Lie algebra: A = sum_a f(theta_a) * T_a
--   2. Apply learned activation: A' = activation(A)
--   3. Exponentiate to group: g = exp(A')
--   4. Transform: y = g * x
-- ============================================================================

local LieAlgebraActivation, parent = torch.class('gu.activations.LieAlgebraActivation', 'nn.Module')

-- ============================================================================
-- Initialization
-- ============================================================================

function LieAlgebraActivation:__init(dim, config)
    parent.__init(self)

    config = config or {}

    self.dim = dim
    self.structure_group = config.structure_group or 'SO'
    self.activation_type = config.activation_type or 'softplus'  -- 'tanh', 'sigmoid', 'softplus', 'elu'
    self.learnable_scale = config.learnable_scale ~= false
    self.use_bias = config.use_bias or false

    -- Compute Lie algebra dimension
    self.lie_dim = self:_computeLieAlgebraDim()

    -- Build Lie algebra generators
    self.generators = self:_buildGenerators()

    -- Learnable parameters for pre-exponential transformation
    -- Scale factors for each generator direction
    self.scale = torch.Tensor(self.lie_dim)
    self.gradScale = torch.Tensor(self.lie_dim)

    -- Optional bias in Lie algebra
    if self.use_bias then
        self.bias = torch.Tensor(self.lie_dim)
        self.gradBias = torch.Tensor(self.lie_dim)
    end

    -- Learnable activation parameters (for parameterized activations)
    self.alpha = torch.Tensor(1)  -- For ELU, LeakyReLU, etc.
    self.gradAlpha = torch.Tensor(1)

    -- Initialize parameters
    self:reset()
end

-- ============================================================================
-- Compute Lie Algebra Dimension
-- ============================================================================

function LieAlgebraActivation:_computeLieAlgebraDim()
    local n = self.dim
    if self.structure_group == 'SO' or self.structure_group == 'Spin' then
        return math.floor(n * (n - 1) / 2)
    elseif self.structure_group == 'SU' then
        return n * n - 1
    elseif self.structure_group == 'U' then
        return n * n
    else  -- GL
        return n * n
    end
end

-- ============================================================================
-- Build Lie Algebra Generators
-- ============================================================================

function LieAlgebraActivation:_buildGenerators()
    local n = self.dim
    local generators = {}

    if self.structure_group == 'SO' or self.structure_group == 'Spin' then
        -- Antisymmetric generators: J_ij[k][l] = delta_ik * delta_jl - delta_il * delta_jk
        local idx = 1
        for i = 1, n do
            for j = i + 1, n do
                local J = torch.zeros(n, n)
                J[i][j] = 1
                J[j][i] = -1
                generators[idx] = J
                idx = idx + 1
            end
        end
    elseif self.structure_group == 'SU' then
        -- Gell-Mann like generators (traceless Hermitian)
        local idx = 1
        -- Off-diagonal symmetric
        for i = 1, n do
            for j = i + 1, n do
                local T = torch.zeros(n, n)
                T[i][j] = 1
                T[j][i] = 1
                generators[idx] = T
                idx = idx + 1
            end
        end
        -- Off-diagonal antisymmetric
        for i = 1, n do
            for j = i + 1, n do
                local T = torch.zeros(n, n)
                T[i][j] = -1
                T[j][i] = 1
                generators[idx] = T
                idx = idx + 1
            end
        end
        -- Diagonal traceless
        for i = 1, n - 1 do
            local T = torch.zeros(n, n)
            local norm = math.sqrt(2.0 / (i * (i + 1)))
            for j = 1, i do
                T[j][j] = norm
            end
            T[i + 1][i + 1] = -i * norm
            generators[idx] = T
            idx = idx + 1
        end
    else  -- GL or U
        local idx = 1
        for i = 1, n do
            for j = 1, n do
                local E = torch.zeros(n, n)
                E[i][j] = 1
                generators[idx] = E
                idx = idx + 1
            end
        end
    end

    return generators
end

-- ============================================================================
-- Activation Functions in Lie Algebra Space
-- ============================================================================

function LieAlgebraActivation:_applyActivation(x)
    local result

    if self.activation_type == 'tanh' then
        result = torch.tanh(x)
    elseif self.activation_type == 'sigmoid' then
        result = torch.sigmoid(x)
    elseif self.activation_type == 'softplus' then
        -- SoftPlus: log(1 + exp(x))
        result = torch.log(torch.exp(x):add(1))
    elseif self.activation_type == 'elu' then
        -- ELU: x if x > 0, alpha * (exp(x) - 1) otherwise
        local alpha = self.alpha[1]
        result = x:clone()
        local mask = x:lt(0)
        if mask:any() then
            local neg_part = torch.exp(x) - 1
            neg_part:mul(alpha)
            result[mask] = neg_part[mask]
        end
    elseif self.activation_type == 'swish' then
        -- Swish: x * sigmoid(x)
        result = x:clone():cmul(torch.sigmoid(x))
    elseif self.activation_type == 'mish' then
        -- Mish: x * tanh(softplus(x))
        local sp = torch.log(torch.exp(x):add(1))
        result = x:clone():cmul(torch.tanh(sp))
    else
        -- Default: identity (linear)
        result = x:clone()
    end

    return result
end

function LieAlgebraActivation:_applyActivationGrad(x, gradOutput)
    local grad

    if self.activation_type == 'tanh' then
        local tanh_x = torch.tanh(x)
        grad = (1 - torch.cmul(tanh_x, tanh_x)):cmul(gradOutput)
    elseif self.activation_type == 'sigmoid' then
        local sig_x = torch.sigmoid(x)
        grad = torch.cmul(sig_x, 1 - sig_x):cmul(gradOutput)
    elseif self.activation_type == 'softplus' then
        grad = torch.sigmoid(x):cmul(gradOutput)
    elseif self.activation_type == 'elu' then
        local alpha = self.alpha[1]
        grad = gradOutput:clone()
        local mask = x:lt(0)
        if mask:any() then
            local deriv = torch.exp(x):mul(alpha)
            grad[mask] = torch.cmul(deriv, gradOutput)[mask]
        end
    elseif self.activation_type == 'swish' then
        local sig_x = torch.sigmoid(x)
        local swish = torch.cmul(x, sig_x)
        grad = torch.cmul(swish + torch.cmul(sig_x, 1 - swish), gradOutput)
    else
        grad = gradOutput:clone()
    end

    return grad
end

-- ============================================================================
-- Matrix Exponential (Pade Approximation)
-- ============================================================================

function LieAlgebraActivation:_matrixExp(A, order)
    order = order or 6
    local n = A:size(1)
    local I = torch.eye(n)

    local norm_A = A:norm()
    if norm_A < 1e-10 then
        return I:clone()
    end

    -- Scale for better convergence
    local s = math.max(0, math.ceil(math.log(norm_A) / math.log(2)))
    local A_scaled = A:clone():div(math.pow(2, s))

    -- Pade coefficients
    local c = {1}
    for k = 1, order do
        c[k + 1] = c[k] * (order - k + 1) / (k * (2 * order - k + 1))
    end

    -- Compute numerator and denominator
    local A_power = I:clone()
    local N = I:clone():mul(c[1])
    local D = I:clone():mul(c[1])

    for k = 1, order do
        A_power = torch.mm(A_power, A_scaled)
        local term = A_power:clone():mul(c[k + 1])
        N:add(term)
        if k % 2 == 0 then
            D:add(term)
        else
            D:add(-1, term)
        end
    end

    -- Solve D * expA = N
    local expA = torch.gesv(N, D)

    -- Square back
    for i = 1, s do
        expA = torch.mm(expA, expA)
    end

    return expA
end

-- ============================================================================
-- Forward Pass
-- ============================================================================

function LieAlgebraActivation:updateOutput(input)
    local is_observerse = type(input) == 'table' and input._type == 'ObserverseTensor'

    local fiber_tensor
    if is_observerse then
        fiber_tensor = input.fiber
    else
        fiber_tensor = input
    end

    -- Handle dimensions
    local was_1d = fiber_tensor:dim() == 1
    local fiber_view = was_1d and fiber_tensor:view(1, -1) or fiber_tensor
    local batch_size = fiber_view:size(1)

    -- Store for backward
    self._input_view = fiber_view
    self._batch_size = batch_size

    -- Compute input magnitude for adaptive scaling
    local input_norms = torch.zeros(batch_size)
    for b = 1, batch_size do
        input_norms[b] = fiber_view[b]:norm()
    end
    self._input_norms = input_norms

    -- Project input to Lie algebra coordinates
    -- theta_a = <x, e_a> where e_a spans the representation space
    local lie_coords = torch.zeros(batch_size, self.lie_dim)

    for b = 1, batch_size do
        local x = fiber_view[b]
        for a = 1, self.lie_dim do
            -- Project onto generator direction
            local T_a = self.generators[a]
            -- Use Frobenius inner product: sum_ij T_a[i][j] * x[i] * x[j] / ||x||
            local coord = 0
            for i = 1, self.dim do
                for j = 1, self.dim do
                    if i <= x:size(1) and j <= x:size(1) then
                        coord = coord + T_a[i][j] * x[i] * x[j]
                    end
                end
            end
            lie_coords[b][a] = coord / (input_norms[b] + 1e-8)
        end
    end

    -- Apply learned scale
    for a = 1, self.lie_dim do
        lie_coords[{{}, a}]:mul(self.scale[a])
    end

    -- Apply bias if used
    if self.use_bias then
        for a = 1, self.lie_dim do
            lie_coords[{{}, a}]:add(self.bias[a])
        end
    end

    -- Apply activation in Lie algebra space
    local activated_coords = self:_applyActivation(lie_coords)
    self._lie_coords = lie_coords
    self._activated_coords = activated_coords

    -- Build output by applying group element
    local output = torch.zeros(batch_size, self.dim)
    self._gauge_elements = {}

    for b = 1, batch_size do
        -- Construct Lie algebra element
        local A = torch.zeros(self.dim, self.dim)
        for a = 1, self.lie_dim do
            A:add(activated_coords[b][a], self.generators[a])
        end

        -- Exponentiate to get group element
        local g = self:_matrixExp(A)
        self._gauge_elements[b] = g

        -- Apply group action: y = g * x
        output[b] = torch.mv(g, fiber_view[b])
    end

    -- Reshape output
    if was_1d then
        output = output:view(self.dim)
    end

    -- Handle ObserverseTensor
    if is_observerse then
        local ObserverseTensor = require 'gu.ObserverseTensor'
        self.output = ObserverseTensor.create(input.base:clone(), output)
    else
        self.output = output
    end

    return self.output
end

-- ============================================================================
-- Backward Pass
-- ============================================================================

function LieAlgebraActivation:updateGradInput(input, gradOutput)
    local is_observerse = type(input) == 'table' and input._type == 'ObserverseTensor'

    local gradOutput_tensor
    if is_observerse then
        gradOutput_tensor = gradOutput.fiber
    else
        gradOutput_tensor = gradOutput
    end

    local was_1d = gradOutput_tensor:dim() == 1
    local gradOutput_view = was_1d and gradOutput_tensor:view(1, -1) or gradOutput_tensor
    local batch_size = self._batch_size

    -- Gradient through group action: dy/dx = g
    local gradInput = torch.zeros(batch_size, self.dim)

    for b = 1, batch_size do
        local g = self._gauge_elements[b]
        gradInput[b] = torch.mv(g:t(), gradOutput_view[b])
    end

    if was_1d then
        gradInput = gradInput:view(self.dim)
    end

    if is_observerse then
        local ObserverseTensor = require 'gu.ObserverseTensor'
        self.gradInput = ObserverseTensor.create(
            torch.zeros(input.base:size()),
            gradInput
        )
    else
        self.gradInput = gradInput
    end

    return self.gradInput
end

-- ============================================================================
-- Accumulate Gradients
-- ============================================================================

function LieAlgebraActivation:accGradParameters(input, gradOutput, scale)
    scale = scale or 1

    local gradOutput_tensor
    if type(gradOutput) == 'table' and gradOutput._type == 'ObserverseTensor' then
        gradOutput_tensor = gradOutput.fiber
    else
        gradOutput_tensor = gradOutput
    end

    local was_1d = gradOutput_tensor:dim() == 1
    local gradOutput_view = was_1d and gradOutput_tensor:view(1, -1) or gradOutput_tensor

    -- Gradient for scale parameters
    -- d_loss/d_scale_a = sum_b d_loss/d_activated[b][a] * d_activated/d_scaled * d_scaled/d_scale
    local grad_activated = self:_applyActivationGrad(self._lie_coords, gradOutput_view)

    for a = 1, self.lie_dim do
        local grad = 0
        for b = 1, self._batch_size do
            -- Approximate gradient contribution
            grad = grad + grad_activated[b][a] * self._lie_coords[b][a] / (self.scale[a] + 1e-8)
        end
        self.gradScale[a] = self.gradScale[a] + scale * grad
    end

    -- Gradient for bias
    if self.use_bias then
        for a = 1, self.lie_dim do
            local grad = 0
            for b = 1, self._batch_size do
                grad = grad + grad_activated[b][a]
            end
            self.gradBias[a] = self.gradBias[a] + scale * grad
        end
    end
end

-- ============================================================================
-- Parameter Initialization
-- ============================================================================

function LieAlgebraActivation:reset()
    -- Initialize scales to 1 (identity-like)
    self.scale:fill(1.0)

    if self.use_bias then
        self.bias:zero()
    end

    -- Alpha for ELU
    self.alpha:fill(1.0)

    self:zeroGradParameters()
end

function LieAlgebraActivation:zeroGradParameters()
    self.gradScale:zero()
    if self.use_bias then
        self.gradBias:zero()
    end
    self.gradAlpha:zero()
end

-- ============================================================================
-- Parameters
-- ============================================================================

function LieAlgebraActivation:parameters()
    local params = {self.scale}
    local gradParams = {self.gradScale}

    if self.use_bias then
        table.insert(params, self.bias)
        table.insert(gradParams, self.gradBias)
    end

    if self.activation_type == 'elu' then
        table.insert(params, self.alpha)
        table.insert(gradParams, self.gradAlpha)
    end

    return params, gradParams
end

-- ============================================================================
-- Utility Methods
-- ============================================================================

function LieAlgebraActivation:getStructureGroup()
    return self.structure_group
end

function LieAlgebraActivation:getGenerators()
    return self.generators
end

function LieAlgebraActivation:getLieAlgebraDim()
    return self.lie_dim
end

function LieAlgebraActivation:__tostring()
    local str = string.format('%s(dim=%d, group=%s, activation=%s',
        torch.type(self), self.dim, self.structure_group, self.activation_type)
    if self.use_bias then str = str .. ', bias' end
    str = str .. ')'
    return str
end

return LieAlgebraActivation
