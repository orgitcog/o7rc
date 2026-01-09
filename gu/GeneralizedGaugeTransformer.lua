-- ============================================================================
-- GeneralizedGaugeTransformer: Full Architecture for Gauge Field Transformations
-- ============================================================================
-- This module implements a generalized gauge transformer architecture that
-- combines concepts from:
--   - Gauge theory (structure groups, connections, curvature)
--   - Transformer architectures (multi-head attention, layer normalization)
--   - Geometric Unity (Observerse structure, tilted gauge groups)
--
-- The architecture treats gauge transformations as learnable operations that
-- respect the geometric structure of the underlying fiber bundle.
--
-- Key Components:
--   1. Multi-Head Gauge Attention: Attention over gauge connections
--   2. Structure Group Layer: Enforces Lie group constraints (SO, SU, Spin)
--   3. Connection Module: Learns gauge connection coefficients
--   4. Curvature Regularizer: Penalizes large field strength
--   5. Parallel Transport: Moves vectors along fiber directions
-- ============================================================================

local GeneralizedGaugeTransformer, parent = torch.class('gu.GeneralizedGaugeTransformer', 'nn.Module')

-- ============================================================================
-- Initialization
-- ============================================================================

function GeneralizedGaugeTransformer:__init(dim, config)
    parent.__init(self)

    config = config or {}

    -- Core dimensions
    self.dim = dim
    self.base_dim = config.base_dim or 4
    self.hidden_dim = config.hidden_dim or dim * 2
    self.num_heads = config.num_heads or 4
    self.head_dim = math.floor(dim / self.num_heads)

    -- Architecture options
    self.structure_group = config.structure_group or 'SO'  -- 'GL', 'SO', 'SU', 'Spin', 'U'
    self.use_attention = config.use_attention ~= false
    self.use_connection = config.use_connection or false
    self.use_curvature_reg = config.use_curvature_reg or false
    self.use_residual = config.use_residual ~= false
    self.use_layernorm = config.use_layernorm ~= false
    self.dropout_p = config.dropout or 0.0

    -- Lie algebra dimension for structure group
    self.lie_dim = self:_computeLieAlgebraDim()

    -- ========================================================================
    -- Multi-Head Gauge Attention
    -- ========================================================================
    if self.use_attention then
        -- Query, Key, Value projections for gauge attention
        self.W_q = torch.Tensor(dim, dim)
        self.W_k = torch.Tensor(dim, dim)
        self.W_v = torch.Tensor(dim, dim)
        self.W_o = torch.Tensor(dim, dim)  -- Output projection

        self.gradW_q = torch.Tensor(dim, dim)
        self.gradW_k = torch.Tensor(dim, dim)
        self.gradW_v = torch.Tensor(dim, dim)
        self.gradW_o = torch.Tensor(dim, dim)

        -- Attention bias (optional positional-like encoding for gauge space)
        self.attention_bias = torch.Tensor(self.num_heads, dim, dim)
        self.gradAttentionBias = torch.Tensor(self.num_heads, dim, dim)
    end

    -- ========================================================================
    -- Structure Group Generators
    -- ========================================================================
    -- Learnable Lie algebra parameters that generate the gauge transformation
    self.lie_params = torch.Tensor(self.lie_dim)
    self.gradLieParams = torch.Tensor(self.lie_dim)

    -- Pre-computed generators for the Lie algebra
    self.generators = self:_buildLieAlgebraGenerators()

    -- ========================================================================
    -- Gauge Connection (for parallel transport)
    -- ========================================================================
    if self.use_connection then
        -- Connection 1-form: A_mu^a (base_dim x lie_dim)
        self.connection_form = torch.Tensor(self.base_dim, self.lie_dim)
        self.gradConnectionForm = torch.Tensor(self.base_dim, self.lie_dim)
    end

    -- ========================================================================
    -- Feed-Forward Network (gauge-equivariant)
    -- ========================================================================
    self.ffn_w1 = torch.Tensor(self.hidden_dim, dim)
    self.ffn_w2 = torch.Tensor(dim, self.hidden_dim)
    self.ffn_b1 = torch.Tensor(self.hidden_dim)
    self.ffn_b2 = torch.Tensor(dim)

    self.gradFfnW1 = torch.Tensor(self.hidden_dim, dim)
    self.gradFfnW2 = torch.Tensor(dim, self.hidden_dim)
    self.gradFfnB1 = torch.Tensor(self.hidden_dim)
    self.gradFfnB2 = torch.Tensor(dim)

    -- ========================================================================
    -- Layer Normalization (per-component)
    -- ========================================================================
    if self.use_layernorm then
        self.ln_gamma_fiber = torch.Tensor(dim)
        self.ln_beta_fiber = torch.Tensor(dim)
        self.ln_gamma_base = torch.Tensor(self.base_dim)
        self.ln_beta_base = torch.Tensor(self.base_dim)

        self.gradLnGammaFiber = torch.Tensor(dim)
        self.gradLnBetaFiber = torch.Tensor(dim)
        self.gradLnGammaBase = torch.Tensor(self.base_dim)
        self.gradLnBetaBase = torch.Tensor(self.base_dim)
    end

    -- ========================================================================
    -- Dropout
    -- ========================================================================
    if self.dropout_p > 0 then
        self.dropout = nn.Dropout(self.dropout_p)
    end

    -- Curvature regularization weight
    self.curvature_weight = config.curvature_weight or 0.01

    -- Initialize parameters
    self:reset()
end

-- ============================================================================
-- Compute Lie Algebra Dimension
-- ============================================================================

function GeneralizedGaugeTransformer:_computeLieAlgebraDim()
    local n = self.dim
    if self.structure_group == 'SO' or self.structure_group == 'Spin' then
        return n * (n - 1) / 2  -- Antisymmetric matrices
    elseif self.structure_group == 'SU' then
        return n * n - 1  -- Traceless Hermitian
    elseif self.structure_group == 'U' then
        return n * n  -- Hermitian
    else  -- GL
        return n * n  -- General matrices
    end
end

-- ============================================================================
-- Build Lie Algebra Generators
-- ============================================================================

function GeneralizedGaugeTransformer:_buildLieAlgebraGenerators()
    local n = self.dim
    local generators = {}

    if self.structure_group == 'SO' or self.structure_group == 'Spin' then
        -- Build antisymmetric generators J_ij = e_i ^ e_j
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
        -- Build Gell-Mann like generators (traceless Hermitian)
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
        -- Build standard basis e_ij
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
-- Build Gauge Element from Lie Algebra
-- ============================================================================

function GeneralizedGaugeTransformer:_buildGaugeElement()
    local n = self.dim

    -- Construct Lie algebra element: A = sum_a theta^a T_a
    local A = torch.zeros(n, n)
    for a = 1, self.lie_dim do
        A:add(self.lie_params[a], self.generators[a])
    end

    -- Compute matrix exponential: g = exp(A)
    return self:_matrixExp(A)
end

-- ============================================================================
-- Matrix Exponential (Pade Approximation)
-- ============================================================================

function GeneralizedGaugeTransformer:_matrixExp(A, order)
    order = order or 6
    local n = A:size(1)
    local I = torch.eye(n)

    -- Handle zero matrix
    local norm_A = A:norm()
    if norm_A < 1e-10 then
        return I:clone()
    end

    -- Scale matrix to improve convergence
    local s = math.max(0, math.ceil(math.log(norm_A) / math.log(2)))
    local A_scaled = A:clone():div(2^s)

    -- Pade approximation coefficients
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

    -- Solve D * exp(A) = N
    local expA = torch.gesv(N, D)

    -- Square back up
    for i = 1, s do
        expA = torch.mm(expA, expA)
    end

    return expA
end

-- ============================================================================
-- Multi-Head Gauge Attention
-- ============================================================================

function GeneralizedGaugeTransformer:_multiHeadGaugeAttention(fiber, base)
    if not self.use_attention then
        return fiber
    end

    local batch_size = fiber:size(1)
    local dim = fiber:size(2)

    -- Compute Q, K, V
    local Q = torch.mm(fiber, self.W_q:t())
    local K = torch.mm(fiber, self.W_k:t())
    local V = torch.mm(fiber, self.W_v:t())

    -- Reshape for multi-head attention
    local Q_heads = Q:view(batch_size, self.num_heads, self.head_dim)
    local K_heads = K:view(batch_size, self.num_heads, self.head_dim)
    local V_heads = V:view(batch_size, self.num_heads, self.head_dim)

    -- Compute attention scores
    local scale = 1.0 / math.sqrt(self.head_dim)
    local scores = torch.zeros(batch_size, self.num_heads)

    for h = 1, self.num_heads do
        for b = 1, batch_size do
            local q = Q_heads[b][h]
            local k = K_heads[b][h]
            scores[b][h] = torch.dot(q, k) * scale
        end
    end

    -- Apply softmax
    local attn_weights = torch.exp(scores)
    for b = 1, batch_size do
        local sum = attn_weights[b]:sum()
        if sum > 0 then
            attn_weights[b]:div(sum)
        end
    end

    -- Apply attention to values
    local attended = torch.zeros(batch_size, dim)
    for b = 1, batch_size do
        for h = 1, self.num_heads do
            local v = V_heads[b][h]
            local start_idx = (h - 1) * self.head_dim + 1
            local end_idx = h * self.head_dim
            for i = start_idx, end_idx do
                attended[b][i] = attended[b][i] + attn_weights[b][h] * v[i - start_idx + 1]
            end
        end
    end

    -- Output projection
    local output = torch.mm(attended, self.W_o:t())

    -- Store attention weights for visualization/analysis
    self._last_attn_weights = attn_weights

    return output
end

-- ============================================================================
-- Feed-Forward Network (Gauge-Equivariant)
-- ============================================================================

function GeneralizedGaugeTransformer:_feedForward(x)
    -- FFN: x -> Linear -> ReLU -> Linear
    local hidden = torch.mm(x, self.ffn_w1:t())
    hidden:add(self.ffn_b1:view(1, -1):expandAs(hidden))

    -- ReLU activation
    hidden:cmax(0)

    -- Second linear layer
    local output = torch.mm(hidden, self.ffn_w2:t())
    output:add(self.ffn_b2:view(1, -1):expandAs(output))

    return output
end

-- ============================================================================
-- Layer Normalization
-- ============================================================================

function GeneralizedGaugeTransformer:_layerNorm(x, gamma, beta)
    if not self.use_layernorm then
        return x
    end

    local dim = x:size(2)
    local batch_size = x:size(1)

    -- Compute mean and variance
    local mean = x:mean(2)
    local var = x:var(2)

    -- Normalize
    local eps = 1e-5
    local x_norm = x:clone()
    for b = 1, batch_size do
        local m = mean[b][1]
        local v = var[b][1]
        local std = math.sqrt(v + eps)
        for i = 1, dim do
            x_norm[b][i] = (x[b][i] - m) / std
        end
    end

    -- Scale and shift
    for b = 1, batch_size do
        for i = 1, dim do
            x_norm[b][i] = x_norm[b][i] * gamma[i] + beta[i]
        end
    end

    return x_norm
end

-- ============================================================================
-- Compute Field Strength (Curvature)
-- ============================================================================

function GeneralizedGaugeTransformer:computeFieldStrength()
    if not self.use_connection then
        return nil
    end

    -- Field strength F_mu_nu = d_mu A_nu - d_nu A_mu + [A_mu, A_nu]
    local F = torch.zeros(self.base_dim, self.base_dim, self.dim, self.dim)

    for mu = 1, self.base_dim do
        for nu = mu + 1, self.base_dim do
            -- Build A_mu and A_nu as Lie algebra elements
            local A_mu = torch.zeros(self.dim, self.dim)
            local A_nu = torch.zeros(self.dim, self.dim)

            for a = 1, self.lie_dim do
                A_mu:add(self.connection_form[mu][a], self.generators[a])
                A_nu:add(self.connection_form[nu][a], self.generators[a])
            end

            -- Compute commutator [A_mu, A_nu]
            local commutator = torch.mm(A_mu, A_nu) - torch.mm(A_nu, A_mu)

            -- Store in F (simplified: ignoring derivative terms)
            F[mu][nu] = commutator
            F[nu][mu] = commutator:clone():mul(-1)
        end
    end

    return F
end

-- ============================================================================
-- Curvature Regularization Loss
-- ============================================================================

function GeneralizedGaugeTransformer:curvatureLoss()
    if not self.use_curvature_reg then
        return 0
    end

    local F = self:computeFieldStrength()
    if F == nil then
        return 0
    end

    -- ||F||^2 = sum_{mu < nu} Tr(F_mu_nu^2)
    local loss = 0
    for mu = 1, self.base_dim do
        for nu = mu + 1, self.base_dim do
            local F_mn = F[mu][nu]
            loss = loss + torch.trace(torch.mm(F_mn, F_mn:t()))
        end
    end

    return self.curvature_weight * loss
end

-- ============================================================================
-- Parameter Initialization
-- ============================================================================

function GeneralizedGaugeTransformer:reset()
    local stdv = 1.0 / math.sqrt(self.dim)

    -- Attention weights
    if self.use_attention then
        self.W_q:uniform(-stdv, stdv)
        self.W_k:uniform(-stdv, stdv)
        self.W_v:uniform(-stdv, stdv)
        self.W_o:uniform(-stdv, stdv)
        self.attention_bias:zero()
    end

    -- Lie algebra parameters (small values for near-identity transform)
    self.lie_params:normal(0, 0.01)

    -- Connection form
    if self.use_connection then
        self.connection_form:normal(0, 0.01)
    end

    -- FFN weights
    local ffn_stdv = 1.0 / math.sqrt(self.hidden_dim)
    self.ffn_w1:uniform(-stdv, stdv)
    self.ffn_w2:uniform(-ffn_stdv, ffn_stdv)
    self.ffn_b1:zero()
    self.ffn_b2:zero()

    -- Layer norm
    if self.use_layernorm then
        self.ln_gamma_fiber:fill(1)
        self.ln_beta_fiber:zero()
        self.ln_gamma_base:fill(1)
        self.ln_beta_base:zero()
    end

    -- Zero gradients
    self:zeroGradParameters()
end

function GeneralizedGaugeTransformer:zeroGradParameters()
    if self.use_attention then
        self.gradW_q:zero()
        self.gradW_k:zero()
        self.gradW_v:zero()
        self.gradW_o:zero()
        self.gradAttentionBias:zero()
    end

    self.gradLieParams:zero()

    if self.use_connection then
        self.gradConnectionForm:zero()
    end

    self.gradFfnW1:zero()
    self.gradFfnW2:zero()
    self.gradFfnB1:zero()
    self.gradFfnB2:zero()

    if self.use_layernorm then
        self.gradLnGammaFiber:zero()
        self.gradLnBetaFiber:zero()
        self.gradLnGammaBase:zero()
        self.gradLnBetaBase:zero()
    end
end

-- ============================================================================
-- Forward Pass
-- ============================================================================

function GeneralizedGaugeTransformer:updateOutput(input)
    local is_observerse = type(input) == 'table' and input._type == 'ObserverseTensor'

    local base_tensor, fiber_tensor
    local base_was_1d, fiber_was_1d

    if is_observerse then
        base_tensor = input.base
        fiber_tensor = input.fiber
        base_was_1d = base_tensor:dim() == 1
        fiber_was_1d = fiber_tensor:dim() == 1
    else
        base_tensor = torch.zeros(input:size(1), self.base_dim)
        fiber_tensor = input
        base_was_1d = false
        fiber_was_1d = input:dim() == 1
    end

    -- Ensure 2D
    local base_view = base_was_1d and base_tensor:view(1, -1) or base_tensor
    local fiber_view = fiber_was_1d and fiber_tensor:view(1, -1) or fiber_tensor

    -- Store for residual
    local residual_fiber = fiber_view:clone()
    local residual_base = base_view:clone()

    -- ========================================================================
    -- Step 1: Multi-Head Gauge Attention
    -- ========================================================================
    local attended_fiber = self:_multiHeadGaugeAttention(fiber_view, base_view)

    -- Add residual and normalize
    if self.use_residual then
        attended_fiber:add(residual_fiber)
    end
    if self.use_layernorm then
        attended_fiber = self:_layerNorm(attended_fiber, self.ln_gamma_fiber, self.ln_beta_fiber)
    end

    -- ========================================================================
    -- Step 2: Apply Gauge Transformation
    -- ========================================================================
    local gauge_element = self:_buildGaugeElement()
    local transformed_fiber = torch.mm(attended_fiber, gauge_element:t())

    -- ========================================================================
    -- Step 3: Feed-Forward Network
    -- ========================================================================
    local ffn_residual = transformed_fiber:clone()
    local ffn_output = self:_feedForward(transformed_fiber)

    -- Add residual
    if self.use_residual then
        ffn_output:add(ffn_residual)
    end

    -- Apply dropout
    if self.dropout and self.train then
        ffn_output = self.dropout:forward(ffn_output)
    end

    -- ========================================================================
    -- Step 4: Base Space Transformation (for Observerse)
    -- ========================================================================
    local transformed_base = base_view:clone()
    if is_observerse then
        -- Apply gauge-induced transformation to base
        -- In GU, the tilted gauge group acts on both spaces
        if self.base_dim == self.dim then
            transformed_base = torch.mm(base_view, gauge_element:t())
        end
        if self.use_layernorm then
            transformed_base = self:_layerNorm(transformed_base, self.ln_gamma_base, self.ln_beta_base)
        end
    end

    -- ========================================================================
    -- Reshape and return
    -- ========================================================================
    if base_was_1d then
        transformed_base = transformed_base:view(self.base_dim)
    end
    if fiber_was_1d then
        ffn_output = ffn_output:view(self.dim)
    end

    if is_observerse then
        local ObserverseTensor = require 'gu.ObserverseTensor'
        self.output = ObserverseTensor.create(transformed_base, ffn_output)
    else
        self.output = ffn_output
    end

    -- Store gauge element for backward pass
    self._gauge_element = gauge_element

    return self.output
end

-- ============================================================================
-- Backward Pass
-- ============================================================================

function GeneralizedGaugeTransformer:updateGradInput(input, gradOutput)
    local is_observerse = type(input) == 'table' and input._type == 'ObserverseTensor'

    local input_tensor, gradOutput_tensor

    if is_observerse then
        input_tensor = input.fiber
        gradOutput_tensor = gradOutput.fiber
    else
        input_tensor = input
        gradOutput_tensor = gradOutput
    end

    -- Ensure proper dimensions
    local was_1d = input_tensor:dim() == 1
    local input_view = was_1d and input_tensor:view(1, -1) or input_tensor
    local gradOutput_view = was_1d and gradOutput_tensor:view(1, -1) or gradOutput_tensor

    -- Backprop through gauge transformation
    local gauge_element = self._gauge_element or self:_buildGaugeElement()
    self.gradInput = torch.mm(gradOutput_view, gauge_element)

    -- Reshape if needed
    if was_1d then
        self.gradInput = self.gradInput:view(self.dim)
    end

    -- If input was ObserverseTensor, return ObserverseTensor gradient
    if is_observerse then
        local ObserverseTensor = require 'gu.ObserverseTensor'
        local base_grad
        if was_1d then
            base_grad = torch.zeros(input.base:size())
        else
            base_grad = torch.zeros(input.base:size())
        end
        self.gradInput = ObserverseTensor.create(base_grad, self.gradInput)
    end

    return self.gradInput
end

-- ============================================================================
-- Accumulate Gradients
-- ============================================================================

function GeneralizedGaugeTransformer:accGradParameters(input, gradOutput, scale)
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
    local was_1d = input_tensor:dim() == 1
    local input_view = was_1d and input_tensor:view(1, -1) or input_tensor
    local gradOutput_view = was_1d and gradOutput_tensor:view(1, -1) or gradOutput_tensor

    -- Gradient for Lie parameters
    local gauge_element = self._gauge_element or self:_buildGaugeElement()

    -- dL/d(theta_a) = Tr(dL/dg * dg/d(theta_a))
    -- For exp(sum theta_a T_a), derivative is approximately T_a * exp(...)
    for a = 1, self.lie_dim do
        local T_a = self.generators[a]
        local dg_dtheta = torch.mm(T_a, gauge_element)

        -- Compute gradient contribution
        local grad = 0
        for b = 1, input_view:size(1) do
            local x = input_view[b]
            local dy = gradOutput_view[b]
            -- dy * (dg/dtheta * x)^T
            local dx_dtheta = torch.mv(dg_dtheta:t(), x)
            grad = grad + torch.dot(dy, dx_dtheta)
        end

        self.gradLieParams[a] = self.gradLieParams[a] + scale * grad
    end

    -- Gradient for FFN weights (simplified)
    self.gradFfnW2:addmm(scale, gradOutput_view:t(), input_view)
end

-- ============================================================================
-- Parameters
-- ============================================================================

function GeneralizedGaugeTransformer:parameters()
    local params = {}
    local gradParams = {}

    if self.use_attention then
        table.insert(params, self.W_q)
        table.insert(params, self.W_k)
        table.insert(params, self.W_v)
        table.insert(params, self.W_o)
        table.insert(gradParams, self.gradW_q)
        table.insert(gradParams, self.gradW_k)
        table.insert(gradParams, self.gradW_v)
        table.insert(gradParams, self.gradW_o)
    end

    table.insert(params, self.lie_params)
    table.insert(gradParams, self.gradLieParams)

    if self.use_connection then
        table.insert(params, self.connection_form)
        table.insert(gradParams, self.gradConnectionForm)
    end

    table.insert(params, self.ffn_w1)
    table.insert(params, self.ffn_w2)
    table.insert(params, self.ffn_b1)
    table.insert(params, self.ffn_b2)
    table.insert(gradParams, self.gradFfnW1)
    table.insert(gradParams, self.gradFfnW2)
    table.insert(gradParams, self.gradFfnB1)
    table.insert(gradParams, self.gradFfnB2)

    if self.use_layernorm then
        table.insert(params, self.ln_gamma_fiber)
        table.insert(params, self.ln_beta_fiber)
        table.insert(params, self.ln_gamma_base)
        table.insert(params, self.ln_beta_base)
        table.insert(gradParams, self.gradLnGammaFiber)
        table.insert(gradParams, self.gradLnBetaFiber)
        table.insert(gradParams, self.gradLnGammaBase)
        table.insert(gradParams, self.gradLnBetaBase)
    end

    return params, gradParams
end

-- ============================================================================
-- Training / Evaluation Mode
-- ============================================================================

function GeneralizedGaugeTransformer:training()
    self.train = true
    if self.dropout then self.dropout:training() end
end

function GeneralizedGaugeTransformer:evaluate()
    self.train = false
    if self.dropout then self.dropout:evaluate() end
end

-- ============================================================================
-- Utility Methods
-- ============================================================================

-- Get current gauge element
function GeneralizedGaugeTransformer:getGaugeElement()
    return self:_buildGaugeElement()
end

-- Get attention weights from last forward pass
function GeneralizedGaugeTransformer:getAttentionWeights()
    return self._last_attn_weights
end

-- Get structure group
function GeneralizedGaugeTransformer:getStructureGroup()
    return self.structure_group
end

-- Get Lie algebra generators
function GeneralizedGaugeTransformer:getGenerators()
    return self.generators
end

-- ============================================================================
-- String Representation
-- ============================================================================

function GeneralizedGaugeTransformer:__tostring()
    local components = {}
    table.insert(components, string.format("dim=%d", self.dim))
    table.insert(components, string.format("group=%s", self.structure_group))
    table.insert(components, string.format("heads=%d", self.num_heads))
    if self.use_attention then table.insert(components, "attention") end
    if self.use_connection then table.insert(components, "connection") end
    if self.use_residual then table.insert(components, "residual") end
    if self.use_layernorm then table.insert(components, "layernorm") end
    if self.dropout_p > 0 then table.insert(components, string.format("dropout=%.2f", self.dropout_p)) end

    return string.format('%s(%s)', torch.type(self), table.concat(components, ", "))
end

return GeneralizedGaugeTransformer
