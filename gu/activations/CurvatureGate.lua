-- ============================================================================
-- CurvatureGate: Curvature-Aware Gating Activation
-- ============================================================================
-- This activation gates features based on the local curvature (field strength)
-- of gauge connections. It amplifies signals where curvature is meaningful
-- and suppresses noise where curvature is small.
--
-- Key Features:
--   - Computes field strength F = [A_mu, A_nu] from connection coefficients
--   - Uses curvature magnitude ||F|| to gate features
--   - Learnable sensitivity parameter for curvature response
--   - Supports both soft and hard gating modes
--   - Can be used standalone or integrated with gauge transformers
--
-- Mathematical Formulation:
--   Given input x and connection coefficients A:
--   1. Compute field strength: F_mu_nu = [A_mu, A_nu]
--   2. Compute curvature magnitude: ||F|| = sqrt(sum Tr(F_mn^2))
--   3. Gate: y = x * sigmoid(alpha * ||F|| + beta)
--   Or in "amplify" mode: y = x * (1 + tanh(alpha * ||F||))
-- ============================================================================

local CurvatureGate, parent = torch.class('gu.activations.CurvatureGate', 'nn.Module')

-- ============================================================================
-- Initialization
-- ============================================================================

function CurvatureGate:__init(dim, config)
    parent.__init(self)

    config = config or {}

    self.dim = dim
    self.base_dim = config.base_dim or 4
    self.lie_dim = config.lie_dim or math.floor(dim * (dim - 1) / 2)
    self.gate_type = config.gate_type or 'sigmoid'  -- 'sigmoid', 'tanh', 'softplus', 'hard'
    self.mode = config.mode or 'gate'  -- 'gate', 'amplify', 'modulate'

    -- Structure group for generator computation
    self.structure_group = config.structure_group or 'SO'

    -- Build generators
    self.generators = self:_buildGenerators()

    -- Learnable parameters
    -- Sensitivity to curvature
    self.alpha = torch.Tensor(1)
    self.gradAlpha = torch.Tensor(1)

    -- Bias term
    self.beta = torch.Tensor(1)
    self.gradBeta = torch.Tensor(1)

    -- Optional: per-feature scaling
    self.use_per_feature = config.use_per_feature or false
    if self.use_per_feature then
        self.feature_scale = torch.Tensor(dim)
        self.gradFeatureScale = torch.Tensor(dim)
    end

    -- Optional: learnable connection coefficients
    self.learnable_connection = config.learnable_connection or false
    if self.learnable_connection then
        self.connection = torch.Tensor(self.base_dim, self.lie_dim)
        self.gradConnection = torch.Tensor(self.base_dim, self.lie_dim)
    end

    -- Temperature for hard gating
    self.temperature = config.temperature or 1.0

    -- Initialize
    self:reset()
end

-- ============================================================================
-- Build Lie Algebra Generators
-- ============================================================================

function CurvatureGate:_buildGenerators()
    local n = self.dim
    local generators = {}

    if self.structure_group == 'SO' or self.structure_group == 'Spin' then
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
        local idx = 1
        for i = 1, n do
            for j = i + 1, n do
                local T = torch.zeros(n, n)
                T[i][j] = 1
                T[j][i] = 1
                generators[idx] = T
                idx = idx + 1
            end
        end
        for i = 1, n do
            for j = i + 1, n do
                local T = torch.zeros(n, n)
                T[i][j] = -1
                T[j][i] = 1
                generators[idx] = T
                idx = idx + 1
            end
        end
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
    else
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
-- Compute Field Strength from Connection
-- ============================================================================

function CurvatureGate:_computeFieldStrength(connection)
    -- connection: [base_dim, lie_dim] tensor
    -- Returns field strength tensor F[mu][nu] as matrix

    local F_mag_sq = 0

    for mu = 1, self.base_dim do
        for nu = mu + 1, self.base_dim do
            -- Build A_mu and A_nu as Lie algebra elements
            local A_mu = torch.zeros(self.dim, self.dim)
            local A_nu = torch.zeros(self.dim, self.dim)

            local actual_lie_dim = math.min(self.lie_dim, connection:size(2))
            local actual_gen_count = math.min(actual_lie_dim, #self.generators)

            for a = 1, actual_gen_count do
                A_mu:add(connection[mu][a], self.generators[a])
                A_nu:add(connection[nu][a], self.generators[a])
            end

            -- Compute commutator [A_mu, A_nu]
            local F_mn = torch.mm(A_mu, A_nu) - torch.mm(A_nu, A_mu)

            -- Add to total magnitude: Tr(F_mn^2)
            F_mag_sq = F_mag_sq + torch.trace(torch.mm(F_mn, F_mn:t()))
        end
    end

    return math.sqrt(F_mag_sq + 1e-10)
end

-- ============================================================================
-- Compute Curvature from Input (adaptive)
-- ============================================================================

function CurvatureGate:_computeAdaptiveCurvature(input)
    -- For inputs without explicit connection, estimate curvature from
    -- the structure of the input itself

    local batch_size = input:size(1)
    local curvatures = torch.zeros(batch_size)

    for b = 1, batch_size do
        local x = input[b]

        -- Estimate local curvature from input variance and structure
        -- Use outer product to estimate connection-like structure
        local outer = torch.ger(x, x)

        -- Compute antisymmetric part (like curvature)
        local antisym = (outer - outer:t()):mul(0.5)

        -- Curvature magnitude is Frobenius norm of antisymmetric part
        curvatures[b] = antisym:norm()
    end

    return curvatures
end

-- ============================================================================
-- Apply Gating Function
-- ============================================================================

function CurvatureGate:_applyGate(x, curvature)
    local alpha = self.alpha[1]
    local beta = self.beta[1]
    local z = alpha * curvature + beta

    local gate
    if self.gate_type == 'sigmoid' then
        gate = 1.0 / (1.0 + math.exp(-z))
    elseif self.gate_type == 'tanh' then
        gate = math.tanh(z)
    elseif self.gate_type == 'softplus' then
        gate = math.log(1 + math.exp(z))
    elseif self.gate_type == 'hard' then
        -- Hard sigmoid with temperature
        gate = math.max(0, math.min(1, (z / self.temperature + 1) / 2))
    else
        gate = 1.0 / (1.0 + math.exp(-z))
    end

    -- Apply based on mode
    local result
    if self.mode == 'gate' then
        result = x * gate
    elseif self.mode == 'amplify' then
        result = x * (1 + gate)
    elseif self.mode == 'modulate' then
        -- Modulate around identity
        result = x * (0.5 + gate)
    else
        result = x * gate
    end

    return result, gate
end

-- ============================================================================
-- Forward Pass
-- ============================================================================

function CurvatureGate:updateOutput(input)
    local is_observerse = type(input) == 'table' and input._type == 'ObserverseTensor'

    local fiber_tensor, base_tensor
    if is_observerse then
        fiber_tensor = input.fiber
        base_tensor = input.base
    else
        fiber_tensor = input
        base_tensor = nil
    end

    local was_1d = fiber_tensor:dim() == 1
    local fiber_view = was_1d and fiber_tensor:view(1, -1) or fiber_tensor
    local batch_size = fiber_view:size(1)

    -- Store for backward
    self._input_view = fiber_view
    self._batch_size = batch_size
    self._was_1d = was_1d

    -- Compute curvature
    local curvatures
    if self.learnable_connection then
        -- Use learnable connection for all batches
        local F_mag = self:_computeFieldStrength(self.connection)
        curvatures = torch.Tensor(batch_size):fill(F_mag)
    else
        -- Estimate curvature from input structure
        curvatures = self:_computeAdaptiveCurvature(fiber_view)
    end

    self._curvatures = curvatures

    -- Apply gating
    local output = torch.zeros(batch_size, self.dim)
    self._gates = torch.zeros(batch_size)

    for b = 1, batch_size do
        local gated, gate = self:_applyGate(fiber_view[b], curvatures[b])
        output[b] = gated
        self._gates[b] = gate
    end

    -- Apply per-feature scaling if enabled
    if self.use_per_feature then
        for i = 1, self.dim do
            output[{{}, i}]:mul(self.feature_scale[i])
        end
    end

    -- Reshape
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

function CurvatureGate:updateGradInput(input, gradOutput)
    local is_observerse = type(input) == 'table' and input._type == 'ObserverseTensor'

    local gradOutput_tensor
    if is_observerse then
        gradOutput_tensor = gradOutput.fiber
    else
        gradOutput_tensor = gradOutput
    end

    local gradOutput_view = self._was_1d and gradOutput_tensor:view(1, -1) or gradOutput_tensor

    -- Gradient through gating
    local gradInput = torch.zeros(self._batch_size, self.dim)

    for b = 1, self._batch_size do
        local gate = self._gates[b]

        -- d(x * gate)/dx = gate (for fixed gate)
        -- In gate mode
        if self.mode == 'gate' then
            gradInput[b] = gradOutput_view[b] * gate
        elseif self.mode == 'amplify' then
            gradInput[b] = gradOutput_view[b] * (1 + gate)
        elseif self.mode == 'modulate' then
            gradInput[b] = gradOutput_view[b] * (0.5 + gate)
        else
            gradInput[b] = gradOutput_view[b] * gate
        end
    end

    -- Handle per-feature scaling gradient
    if self.use_per_feature then
        for i = 1, self.dim do
            gradInput[{{}, i}]:mul(self.feature_scale[i])
        end
    end

    if self._was_1d then
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

function CurvatureGate:accGradParameters(input, gradOutput, scale)
    scale = scale or 1

    local gradOutput_tensor
    if type(gradOutput) == 'table' and gradOutput._type == 'ObserverseTensor' then
        gradOutput_tensor = gradOutput.fiber
    else
        gradOutput_tensor = gradOutput
    end

    local gradOutput_view = self._was_1d and gradOutput_tensor:view(1, -1) or gradOutput_tensor

    -- Gradient for alpha (curvature sensitivity)
    local grad_alpha = 0
    local grad_beta = 0

    for b = 1, self._batch_size do
        local x = self._input_view[b]
        local curvature = self._curvatures[b]
        local gate = self._gates[b]
        local dy = gradOutput_view[b]

        -- d_gate/d_alpha = curvature * gate * (1 - gate) for sigmoid
        local dgate_dalpha, dgate_dbeta
        if self.gate_type == 'sigmoid' then
            dgate_dalpha = curvature * gate * (1 - gate)
            dgate_dbeta = gate * (1 - gate)
        elseif self.gate_type == 'tanh' then
            dgate_dalpha = curvature * (1 - gate * gate)
            dgate_dbeta = 1 - gate * gate
        else
            dgate_dalpha = curvature * gate * (1 - gate)
            dgate_dbeta = gate * (1 - gate)
        end

        -- d_loss/d_alpha = sum dy * x * d_gate/d_alpha
        local factor
        if self.mode == 'gate' then
            factor = torch.dot(dy, x)
        elseif self.mode == 'amplify' then
            factor = torch.dot(dy, x)
        else
            factor = torch.dot(dy, x)
        end

        grad_alpha = grad_alpha + factor * dgate_dalpha
        grad_beta = grad_beta + factor * dgate_dbeta
    end

    self.gradAlpha[1] = self.gradAlpha[1] + scale * grad_alpha
    self.gradBeta[1] = self.gradBeta[1] + scale * grad_beta

    -- Gradient for per-feature scaling
    if self.use_per_feature then
        for i = 1, self.dim do
            local grad = 0
            for b = 1, self._batch_size do
                grad = grad + gradOutput_view[b][i] * self._input_view[b][i] * self._gates[b]
            end
            self.gradFeatureScale[i] = self.gradFeatureScale[i] + scale * grad
        end
    end
end

-- ============================================================================
-- Parameter Initialization
-- ============================================================================

function CurvatureGate:reset()
    -- Initialize alpha to make gate responsive but not saturated
    self.alpha:fill(1.0)
    self.beta:zero()

    if self.use_per_feature then
        self.feature_scale:fill(1.0)
    end

    if self.learnable_connection then
        self.connection:normal(0, 0.01)
    end

    self:zeroGradParameters()
end

function CurvatureGate:zeroGradParameters()
    self.gradAlpha:zero()
    self.gradBeta:zero()

    if self.use_per_feature then
        self.gradFeatureScale:zero()
    end

    if self.learnable_connection then
        self.gradConnection:zero()
    end
end

-- ============================================================================
-- Parameters
-- ============================================================================

function CurvatureGate:parameters()
    local params = {self.alpha, self.beta}
    local gradParams = {self.gradAlpha, self.gradBeta}

    if self.use_per_feature then
        table.insert(params, self.feature_scale)
        table.insert(gradParams, self.gradFeatureScale)
    end

    if self.learnable_connection then
        table.insert(params, self.connection)
        table.insert(gradParams, self.gradConnection)
    end

    return params, gradParams
end

-- ============================================================================
-- Set External Connection
-- ============================================================================

function CurvatureGate:setConnection(connection)
    -- Allow external module to provide connection coefficients
    self._external_connection = connection
end

-- ============================================================================
-- Get Last Curvatures
-- ============================================================================

function CurvatureGate:getLastCurvatures()
    return self._curvatures
end

-- ============================================================================
-- Get Last Gates
-- ============================================================================

function CurvatureGate:getLastGates()
    return self._gates
end

-- ============================================================================
-- String Representation
-- ============================================================================

function CurvatureGate:__tostring()
    local str = string.format('%s(dim=%d, base_dim=%d, gate=%s, mode=%s',
        torch.type(self), self.dim, self.base_dim, self.gate_type, self.mode)
    if self.use_per_feature then str = str .. ', per_feature' end
    if self.learnable_connection then str = str .. ', learnable_conn' end
    str = str .. ')'
    return str
end

return CurvatureGate
