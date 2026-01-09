-- ============================================================================
-- GaugeEquivariantActivation: Gauge Symmetry-Respecting Non-Linearity
-- ============================================================================
-- This activation function preserves gauge symmetry by ensuring that for
-- a gauge transformation g: f(g * x) = g * f(x) (equivariance) or
-- f(g * x) = f(x) (invariance).
--
-- Key Features:
--   - Supports equivariant and invariant modes
--   - Works with SO(n), SU(n), Spin(n), U(n) structure groups
--   - Uses norm-based activations that respect group action
--   - Learnable radial functions for expressivity
--   - Optional angular/phase-preserving transformations
--
-- Mathematical Formulation:
--   For equivariant activation (most common):
--   - Decompose x = r * u where r = ||x|| and u = x/||x||
--   - Apply activation to norm: r' = f(r)
--   - Reconstruct: y = r' * u (preserves direction, transforms magnitude)
--
--   For invariant activation:
--   - Compute invariant features: I(x) = ||x||, Tr(x^T x), etc.
--   - Apply activation to invariants
--   - Use invariants to modulate output
-- ============================================================================

local GaugeEquivariantActivation, parent = torch.class('gu.activations.GaugeEquivariantActivation', 'nn.Module')

-- ============================================================================
-- Initialization
-- ============================================================================

function GaugeEquivariantActivation:__init(dim, config)
    parent.__init(self)

    config = config or {}

    self.dim = dim
    self.structure_group = config.structure_group or 'SO'
    self.mode = config.mode or 'equivariant'  -- 'equivariant', 'invariant', 'hybrid'
    self.radial_activation = config.radial_activation or 'softplus'  -- For norm transformation
    self.use_learnable_radial = config.use_learnable_radial ~= false

    -- For complex representations (SU, U)
    self.is_complex = (self.structure_group == 'SU' or self.structure_group == 'U')

    -- Learnable radial transformation parameters
    -- r' = scale * activation(r * input_scale + bias) + offset
    if self.use_learnable_radial then
        self.input_scale = torch.Tensor(1)
        self.bias = torch.Tensor(1)
        self.output_scale = torch.Tensor(1)
        self.offset = torch.Tensor(1)

        self.gradInputScale = torch.Tensor(1)
        self.gradBias = torch.Tensor(1)
        self.gradOutputScale = torch.Tensor(1)
        self.gradOffset = torch.Tensor(1)
    end

    -- For multi-channel equivariance (different radial functions per representation)
    self.num_channels = config.num_channels or 1
    if self.num_channels > 1 then
        self.channel_dim = math.floor(dim / self.num_channels)
        self.channel_scales = torch.Tensor(self.num_channels)
        self.gradChannelScales = torch.Tensor(self.num_channels)
    end

    -- For hybrid mode: mixing equivariant and invariant
    if self.mode == 'hybrid' then
        self.mix_ratio = torch.Tensor(1)  -- 0 = pure invariant, 1 = pure equivariant
        self.gradMixRatio = torch.Tensor(1)
    end

    -- Angular/phase parameters for complex representations
    if self.is_complex then
        self.phase_shift = torch.Tensor(1)
        self.gradPhaseShift = torch.Tensor(1)
    end

    -- Initialize
    self:reset()
end

-- ============================================================================
-- Radial Activation Functions
-- ============================================================================

function GaugeEquivariantActivation:_radialActivation(r)
    local activation = self.radial_activation
    local result

    if activation == 'softplus' then
        result = math.log(1 + math.exp(r))
    elseif activation == 'relu' then
        result = math.max(0, r)
    elseif activation == 'elu' then
        if r > 0 then
            result = r
        else
            result = math.exp(r) - 1
        end
    elseif activation == 'swish' then
        result = r / (1 + math.exp(-r))
    elseif activation == 'square' then
        result = r * r
    elseif activation == 'sqrt' then
        result = math.sqrt(math.abs(r) + 1e-8)
    elseif activation == 'tanh' then
        result = math.tanh(r)
    elseif activation == 'sigmoid' then
        result = 1 / (1 + math.exp(-r))
    else
        -- Identity
        result = r
    end

    return result
end

function GaugeEquivariantActivation:_radialActivationGrad(r)
    local activation = self.radial_activation
    local grad

    if activation == 'softplus' then
        grad = 1 / (1 + math.exp(-r))
    elseif activation == 'relu' then
        grad = r > 0 and 1 or 0
    elseif activation == 'elu' then
        if r > 0 then
            grad = 1
        else
            grad = math.exp(r)
        end
    elseif activation == 'swish' then
        local sig = 1 / (1 + math.exp(-r))
        local swish = r * sig
        grad = swish + sig * (1 - swish)
    elseif activation == 'square' then
        grad = 2 * r
    elseif activation == 'sqrt' then
        local sign = r >= 0 and 1 or -1
        grad = sign * 0.5 / math.sqrt(math.abs(r) + 1e-8)
    elseif activation == 'tanh' then
        local t = math.tanh(r)
        grad = 1 - t * t
    elseif activation == 'sigmoid' then
        local s = 1 / (1 + math.exp(-r))
        grad = s * (1 - s)
    else
        grad = 1
    end

    return grad
end

-- ============================================================================
-- Compute Norm and Direction
-- ============================================================================

function GaugeEquivariantActivation:_decompose(x)
    local r = x:norm() + 1e-8
    local u = x / r
    return r, u
end

-- ============================================================================
-- Apply Learnable Radial Transform
-- ============================================================================

function GaugeEquivariantActivation:_transformRadius(r)
    if not self.use_learnable_radial then
        return self:_radialActivation(r)
    end

    local input_scale = self.input_scale[1]
    local bias = self.bias[1]
    local output_scale = self.output_scale[1]
    local offset = self.offset[1]

    local z = r * input_scale + bias
    local activated = self:_radialActivation(z)
    local r_prime = output_scale * activated + offset

    return r_prime
end

-- ============================================================================
-- Forward Pass
-- ============================================================================

function GaugeEquivariantActivation:updateOutput(input)
    local is_observerse = type(input) == 'table' and input._type == 'ObserverseTensor'

    local fiber_tensor
    if is_observerse then
        fiber_tensor = input.fiber
    else
        fiber_tensor = input
    end

    local was_1d = fiber_tensor:dim() == 1
    local fiber_view = was_1d and fiber_tensor:view(1, -1) or fiber_tensor
    local batch_size = fiber_view:size(1)

    -- Store for backward
    self._input_view = fiber_view
    self._batch_size = batch_size
    self._was_1d = was_1d

    local output = torch.zeros(batch_size, self.dim)

    -- Store intermediate values
    self._norms = torch.zeros(batch_size)
    self._directions = torch.zeros(batch_size, self.dim)
    self._transformed_norms = torch.zeros(batch_size)

    if self.num_channels > 1 then
        -- Multi-channel equivariant
        for b = 1, batch_size do
            local x = fiber_view[b]
            for c = 1, self.num_channels do
                local start_idx = (c - 1) * self.channel_dim + 1
                local end_idx = math.min(c * self.channel_dim, self.dim)

                local x_c = x[{{start_idx, end_idx}}]
                local r, u = self:_decompose(x_c)

                local r_prime = self:_transformRadius(r) * self.channel_scales[c]

                for i = start_idx, end_idx do
                    output[b][i] = r_prime * u[i - start_idx + 1]
                end
            end
        end
    else
        -- Single equivariant channel
        for b = 1, batch_size do
            local x = fiber_view[b]
            local r, u = self:_decompose(x)

            self._norms[b] = r
            self._directions[b] = u

            if self.mode == 'equivariant' then
                -- Equivariant: transform norm, preserve direction
                local r_prime = self:_transformRadius(r)
                self._transformed_norms[b] = r_prime
                output[b] = u * r_prime

            elseif self.mode == 'invariant' then
                -- Invariant: apply same transformation to all components
                local r_prime = self:_transformRadius(r)
                self._transformed_norms[b] = r_prime
                -- Output is just the transformed norm (scalar invariant, broadcast)
                output[b]:fill(r_prime / math.sqrt(self.dim))

            elseif self.mode == 'hybrid' then
                -- Mix of equivariant and invariant
                local mix = self.mix_ratio[1]
                local r_prime = self:_transformRadius(r)
                self._transformed_norms[b] = r_prime

                local equi_out = u * r_prime
                local inv_out = torch.Tensor(self.dim):fill(r_prime / math.sqrt(self.dim))

                output[b] = equi_out * mix + inv_out * (1 - mix)
            end

            -- Phase shift for complex representations
            if self.is_complex then
                local phase = self.phase_shift[1]
                if phase ~= 0 then
                    -- Apply phase rotation (simplified 2D rotation in pairs)
                    local cos_p = math.cos(phase)
                    local sin_p = math.sin(phase)
                    for i = 1, self.dim - 1, 2 do
                        local real = output[b][i]
                        local imag = output[b][i + 1]
                        output[b][i] = cos_p * real - sin_p * imag
                        output[b][i + 1] = sin_p * real + cos_p * imag
                    end
                end
            end
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

function GaugeEquivariantActivation:updateGradInput(input, gradOutput)
    local is_observerse = type(input) == 'table' and input._type == 'ObserverseTensor'

    local gradOutput_tensor
    if is_observerse then
        gradOutput_tensor = gradOutput.fiber
    else
        gradOutput_tensor = gradOutput
    end

    local gradOutput_view = self._was_1d and gradOutput_tensor:view(1, -1) or gradOutput_tensor

    local gradInput = torch.zeros(self._batch_size, self.dim)

    for b = 1, self._batch_size do
        local r = self._norms[b]
        local u = self._directions[b]
        local r_prime = self._transformed_norms[b]
        local dy = gradOutput_view[b]

        if self.mode == 'equivariant' then
            -- Gradient of y = r' * u
            -- dy/dx = (dr'/dr) * (dr/dx) * u + r' * (du/dx)
            -- dr/dx = x / ||x|| = u
            -- du/dx = (I - u u^T) / ||x||

            local dr_prime_dr = self:_getRadialGradient(r)

            -- Gradient through norm
            local grad_r = torch.dot(dy, u)
            local grad_from_norm = u * (grad_r * dr_prime_dr)

            -- Gradient through direction
            local proj = torch.dot(dy, u)
            local grad_from_dir = (dy - u * proj) * (r_prime / r)

            gradInput[b] = grad_from_norm + grad_from_dir

        elseif self.mode == 'invariant' then
            local dr_prime_dr = self:_getRadialGradient(r)
            local grad_sum = dy:sum()
            gradInput[b] = u * (grad_sum * dr_prime_dr / math.sqrt(self.dim))

        elseif self.mode == 'hybrid' then
            local mix = self.mix_ratio[1]
            local dr_prime_dr = self:_getRadialGradient(r)

            -- Equivariant gradient
            local grad_r = torch.dot(dy, u)
            local grad_from_norm = u * (grad_r * dr_prime_dr)
            local proj = torch.dot(dy, u)
            local grad_from_dir = (dy - u * proj) * (r_prime / r)
            local equi_grad = grad_from_norm + grad_from_dir

            -- Invariant gradient
            local grad_sum = dy:sum()
            local inv_grad = u * (grad_sum * dr_prime_dr / math.sqrt(self.dim))

            gradInput[b] = equi_grad * mix + inv_grad * (1 - mix)
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
-- Get Radial Gradient
-- ============================================================================

function GaugeEquivariantActivation:_getRadialGradient(r)
    if not self.use_learnable_radial then
        return self:_radialActivationGrad(r)
    end

    local input_scale = self.input_scale[1]
    local output_scale = self.output_scale[1]
    local z = r * input_scale + self.bias[1]

    return output_scale * self:_radialActivationGrad(z) * input_scale
end

-- ============================================================================
-- Accumulate Gradients
-- ============================================================================

function GaugeEquivariantActivation:accGradParameters(input, gradOutput, scale)
    scale = scale or 1

    if not self.use_learnable_radial then
        return
    end

    local gradOutput_tensor
    if type(gradOutput) == 'table' and gradOutput._type == 'ObserverseTensor' then
        gradOutput_tensor = gradOutput.fiber
    else
        gradOutput_tensor = gradOutput
    end

    local gradOutput_view = self._was_1d and gradOutput_tensor:view(1, -1) or gradOutput_tensor

    local grad_input_scale = 0
    local grad_bias = 0
    local grad_output_scale = 0
    local grad_offset = 0

    for b = 1, self._batch_size do
        local r = self._norms[b]
        local u = self._directions[b]
        local dy = gradOutput_view[b]

        local input_scale = self.input_scale[1]
        local output_scale = self.output_scale[1]
        local bias = self.bias[1]

        local z = r * input_scale + bias
        local activated = self:_radialActivation(z)
        local d_act = self:_radialActivationGrad(z)

        -- dy/d_output_scale = activated * u (for equivariant)
        local grad_to_r_prime
        if self.mode == 'equivariant' then
            grad_to_r_prime = torch.dot(dy, u)
        else
            grad_to_r_prime = dy:sum() / math.sqrt(self.dim)
        end

        grad_output_scale = grad_output_scale + grad_to_r_prime * activated
        grad_offset = grad_offset + grad_to_r_prime
        grad_input_scale = grad_input_scale + grad_to_r_prime * output_scale * d_act * r
        grad_bias = grad_bias + grad_to_r_prime * output_scale * d_act
    end

    self.gradInputScale[1] = self.gradInputScale[1] + scale * grad_input_scale
    self.gradBias[1] = self.gradBias[1] + scale * grad_bias
    self.gradOutputScale[1] = self.gradOutputScale[1] + scale * grad_output_scale
    self.gradOffset[1] = self.gradOffset[1] + scale * grad_offset

    -- Gradient for mix ratio in hybrid mode
    if self.mode == 'hybrid' then
        local grad_mix = 0
        for b = 1, self._batch_size do
            local r_prime = self._transformed_norms[b]
            local u = self._directions[b]
            local dy = gradOutput_view[b]

            local equi_out = u * r_prime
            local inv_out = torch.Tensor(self.dim):fill(r_prime / math.sqrt(self.dim))
            local diff = equi_out - inv_out

            grad_mix = grad_mix + torch.dot(dy, diff)
        end
        self.gradMixRatio[1] = self.gradMixRatio[1] + scale * grad_mix
    end
end

-- ============================================================================
-- Parameter Initialization
-- ============================================================================

function GaugeEquivariantActivation:reset()
    if self.use_learnable_radial then
        self.input_scale:fill(1.0)
        self.bias:zero()
        self.output_scale:fill(1.0)
        self.offset:zero()
    end

    if self.num_channels > 1 then
        self.channel_scales:fill(1.0)
    end

    if self.mode == 'hybrid' then
        self.mix_ratio:fill(0.5)
    end

    if self.is_complex then
        self.phase_shift:zero()
    end

    self:zeroGradParameters()
end

function GaugeEquivariantActivation:zeroGradParameters()
    if self.use_learnable_radial then
        self.gradInputScale:zero()
        self.gradBias:zero()
        self.gradOutputScale:zero()
        self.gradOffset:zero()
    end

    if self.num_channels > 1 then
        self.gradChannelScales:zero()
    end

    if self.mode == 'hybrid' then
        self.gradMixRatio:zero()
    end

    if self.is_complex then
        self.gradPhaseShift:zero()
    end
end

-- ============================================================================
-- Parameters
-- ============================================================================

function GaugeEquivariantActivation:parameters()
    local params = {}
    local gradParams = {}

    if self.use_learnable_radial then
        table.insert(params, self.input_scale)
        table.insert(params, self.bias)
        table.insert(params, self.output_scale)
        table.insert(params, self.offset)
        table.insert(gradParams, self.gradInputScale)
        table.insert(gradParams, self.gradBias)
        table.insert(gradParams, self.gradOutputScale)
        table.insert(gradParams, self.gradOffset)
    end

    if self.num_channels > 1 then
        table.insert(params, self.channel_scales)
        table.insert(gradParams, self.gradChannelScales)
    end

    if self.mode == 'hybrid' then
        table.insert(params, self.mix_ratio)
        table.insert(gradParams, self.gradMixRatio)
    end

    if self.is_complex then
        table.insert(params, self.phase_shift)
        table.insert(gradParams, self.gradPhaseShift)
    end

    return params, gradParams
end

-- ============================================================================
-- Utility Methods
-- ============================================================================

function GaugeEquivariantActivation:getStructureGroup()
    return self.structure_group
end

function GaugeEquivariantActivation:getMode()
    return self.mode
end

function GaugeEquivariantActivation:__tostring()
    local str = string.format('%s(dim=%d, group=%s, mode=%s, radial=%s',
        torch.type(self), self.dim, self.structure_group, self.mode, self.radial_activation)
    if self.num_channels > 1 then
        str = str .. string.format(', channels=%d', self.num_channels)
    end
    if self.is_complex then str = str .. ', complex' end
    str = str .. ')'
    return str
end

return GaugeEquivariantActivation
