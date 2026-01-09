-- ============================================================================
-- SpinorActivation: Clifford Algebra-Respecting Spinor Non-Linearity
-- ============================================================================
-- This activation is designed for the 128-dimensional GU spinor space,
-- respecting Clifford algebra structure and spin group properties.
--
-- Key Features:
--   - Preserves spinor norm (inner product structure)
--   - Respects chiral decomposition (left/right-handed spinors)
--   - Maintains Clifford product structure
--   - Learnable transformations within Spin(14) representation
--   - Optional chirality-dependent processing
--
-- Mathematical Background:
--   GU spinors are 128-dimensional (2^7 from Cl(14) Clifford algebra)
--   The spinor space decomposes into chiral components: S = S_L + S_R
--   Each chiral component is 64-dimensional
--
--   Activation respects:
--   - Spinor inner product: <psi, phi> invariant
--   - Chiral structure: separate processing for left/right
--   - Clifford multiplication compatibility
-- ============================================================================

local SpinorActivation, parent = torch.class('gu.activations.SpinorActivation', 'nn.Module')

-- ============================================================================
-- Initialization
-- ============================================================================

function SpinorActivation:__init(dim, config)
    parent.__init(self)

    config = config or {}

    self.dim = dim or 128  -- GU spinor dimension
    self.chiral_dim = math.floor(dim / 2)  -- Each chiral component
    self.activation_type = config.activation_type or 'norm_preserving'
    -- Options: 'norm_preserving', 'chiral_gate', 'clifford_mix', 'spinor_relu'

    -- Whether to process chiral components separately
    self.use_chiral_decomposition = config.use_chiral_decomposition ~= false

    -- Learnable spinor transformation parameters
    -- For norm-preserving: parameters on the Spin group
    self.use_learnable = config.use_learnable ~= false

    if self.use_learnable then
        -- Rotation parameters (bivector components for Spin group)
        -- For Spin(14), there are 14*13/2 = 91 bivector generators
        -- We use a subset for computational tractability
        self.num_generators = config.num_generators or 16

        -- Rotation angles for each generator
        self.rotation_angles = torch.Tensor(self.num_generators)
        self.gradRotationAngles = torch.Tensor(self.num_generators)

        -- Chiral mixing parameter
        if self.use_chiral_decomposition then
            self.chiral_mix = torch.Tensor(1)
            self.gradChiralMix = torch.Tensor(1)
        end

        -- Per-component scale for spinor_relu mode
        if self.activation_type == 'spinor_relu' then
            self.leak_factor = torch.Tensor(1)
            self.gradLeakFactor = torch.Tensor(1)
        end
    end

    -- Build bivector generators (rotation generators in spinor space)
    self.generators = self:_buildBivectorGenerators()

    -- Initialize
    self:reset()
end

-- ============================================================================
-- Build Bivector Generators
-- ============================================================================

function SpinorActivation:_buildBivectorGenerators()
    -- Build a subset of bivector generators for spinor rotations
    -- These are block-diagonal transformations that respect spinor structure

    local generators = {}
    local block_size = 8  -- Process in blocks of 8

    for g = 1, self.num_generators do
        local J = torch.zeros(self.dim, self.dim)

        -- Create rotation in a 2D subspace
        local i = ((g - 1) * 2) % self.dim + 1
        local j = (i % self.dim) + 1

        -- Antisymmetric generator
        J[i][j] = 1
        J[j][i] = -1

        -- For chiral structure, ensure generator respects chirality
        if self.use_chiral_decomposition and g <= self.num_generators / 2 then
            -- First half: left-chiral generators (first half of dimensions)
            local offset = 0
            J:zero()
            local ii = offset + ((g - 1) * 2) % self.chiral_dim + 1
            local jj = offset + (ii % self.chiral_dim) + 1
            J[ii][jj] = 1
            J[jj][ii] = -1
        elseif self.use_chiral_decomposition then
            -- Second half: right-chiral generators (second half of dimensions)
            local offset = self.chiral_dim
            J:zero()
            local idx = g - math.floor(self.num_generators / 2)
            local ii = offset + ((idx - 1) * 2) % self.chiral_dim + 1
            local jj = offset + ((ii - offset - 1) % self.chiral_dim) + offset + 1
            if jj > self.dim then jj = offset + 1 end
            J[ii][jj] = 1
            J[jj][ii] = -1
        end

        generators[g] = J
    end

    return generators
end

-- ============================================================================
-- Build Spinor Rotation
-- ============================================================================

function SpinorActivation:_buildSpinorRotation()
    -- Build rotation matrix from bivector generators
    -- R = exp(sum_a theta_a * J_a)

    local A = torch.zeros(self.dim, self.dim)

    for a = 1, self.num_generators do
        A:add(self.rotation_angles[a], self.generators[a])
    end

    -- Matrix exponential using Rodrigues-like formula for small angles
    -- For efficiency, use truncated Taylor series
    local I = torch.eye(self.dim)
    local norm_A = A:norm()

    if norm_A < 1e-8 then
        return I:clone()
    end

    -- exp(A) ≈ I + A + A^2/2 + A^3/6 for small A
    local A2 = torch.mm(A, A)
    local A3 = torch.mm(A2, A)

    local expA = I + A + A2:mul(0.5) + A3:mul(1.0/6.0)

    -- Orthogonalize to ensure it's a proper rotation
    expA = self:_orthogonalize(expA)

    return expA
end

-- ============================================================================
-- Orthogonalize Matrix (Gram-Schmidt)
-- ============================================================================

function SpinorActivation:_orthogonalize(M)
    -- Simple orthogonalization using iterative projection
    local Q = M:clone()
    local n = M:size(1)

    for i = 1, n do
        local v = Q[{{}, i}]:clone()
        for j = 1, i - 1 do
            local u = Q[{{}, j}]
            local proj = torch.dot(v, u) / (torch.dot(u, u) + 1e-10)
            v:add(-proj, u)
        end
        local norm = v:norm() + 1e-10
        Q[{{}, i}] = v / norm
    end

    return Q
end

-- ============================================================================
-- Norm-Preserving Activation
-- ============================================================================

function SpinorActivation:_normPreserving(spinor)
    -- Apply rotation that preserves spinor norm

    if not self.use_learnable then
        return spinor:clone()
    end

    local R = self:_buildSpinorRotation()
    return torch.mv(R, spinor)
end

-- ============================================================================
-- Chiral Gate Activation
-- ============================================================================

function SpinorActivation:_chiralGate(spinor)
    -- Gate based on chiral component magnitudes

    local left = spinor[{{1, self.chiral_dim}}]
    local right = spinor[{{self.chiral_dim + 1, self.dim}}]

    local left_norm = left:norm() + 1e-8
    local right_norm = right:norm() + 1e-8

    -- Compute chirality measure
    local chirality = (left_norm - right_norm) / (left_norm + right_norm)

    -- Gate function based on chirality
    local gate = torch.sigmoid(torch.Tensor({chirality * 5}))[1]

    local output = spinor:clone()

    -- Apply different activations to left and right
    -- Left: amplify if left-dominant
    output[{{1, self.chiral_dim}}]:mul(1 + gate * 0.5)
    -- Right: amplify if right-dominant
    output[{{self.chiral_dim + 1, self.dim}}]:mul(1 + (1 - gate) * 0.5)

    -- Apply chiral mixing if enabled
    if self.use_learnable and self.use_chiral_decomposition then
        local mix = self.chiral_mix[1]
        local left_out = output[{{1, self.chiral_dim}}]:clone()
        local right_out = output[{{self.chiral_dim + 1, self.dim}}]:clone()

        -- Mix chiral components
        output[{{1, self.chiral_dim}}] = left_out * (1 - mix) + right_out * mix
        output[{{self.chiral_dim + 1, self.dim}}] = right_out * (1 - mix) + left_out * mix
    end

    return output
end

-- ============================================================================
-- Clifford Mix Activation
-- ============================================================================

function SpinorActivation:_cliffordMix(spinor)
    -- Mix components according to Clifford algebra structure
    -- Pairs of components interact through gamma matrix action

    local output = spinor:clone()

    -- Apply pairwise mixing (simulating Clifford multiplication structure)
    local block_size = 4  -- Cl(2) blocks

    for i = 1, self.dim, block_size do
        local block_end = math.min(i + block_size - 1, self.dim)
        local block = output[{{i, block_end}}]

        if block:size(1) == block_size then
            -- Apply non-linear mixing within block
            local norm = block:norm() + 1e-8
            local normalized = block / norm

            -- Soft quaternion-like rotation
            local angle = math.tanh(norm * 0.1)
            local cos_a = math.cos(angle)
            local sin_a = math.sin(angle)

            -- Simple 4D rotation in pairs
            local new_block = block:clone()
            new_block[1] = cos_a * block[1] - sin_a * block[2]
            new_block[2] = sin_a * block[1] + cos_a * block[2]
            new_block[3] = cos_a * block[3] - sin_a * block[4]
            new_block[4] = sin_a * block[3] + cos_a * block[4]

            output[{{i, block_end}}] = new_block
        end
    end

    -- Apply learned rotation
    if self.use_learnable then
        local R = self:_buildSpinorRotation()
        output = torch.mv(R, output)
    end

    return output
end

-- ============================================================================
-- Spinor ReLU Activation
-- ============================================================================

function SpinorActivation:_spinorReLU(spinor)
    -- Spinor-aware ReLU that preserves some structure

    local output = spinor:clone()
    local leak = self.use_learnable and self.leak_factor[1] or 0.01

    -- Apply norm-aware gating
    local norm = spinor:norm()
    local threshold = norm * 0.1  -- Adaptive threshold

    for i = 1, self.dim do
        if spinor[i] < -threshold then
            output[i] = leak * spinor[i]
        elseif spinor[i] < threshold then
            -- Soft transition region
            local t = (spinor[i] + threshold) / (2 * threshold)
            output[i] = spinor[i] * (leak + (1 - leak) * t)
        end
        -- Positive values pass through
    end

    return output
end

-- ============================================================================
-- Forward Pass
-- ============================================================================

function SpinorActivation:updateOutput(input)
    local is_observerse = type(input) == 'table' and input._type == 'ObserverseTensor'

    local spinor_tensor
    if is_observerse then
        spinor_tensor = input.fiber
    else
        spinor_tensor = input
    end

    local was_1d = spinor_tensor:dim() == 1
    local spinor_view = was_1d and spinor_tensor:view(1, -1) or spinor_tensor
    local batch_size = spinor_view:size(1)

    -- Store for backward
    self._input_view = spinor_view
    self._batch_size = batch_size
    self._was_1d = was_1d

    local output = torch.zeros(batch_size, self.dim)

    for b = 1, batch_size do
        local spinor = spinor_view[b]

        if self.activation_type == 'norm_preserving' then
            output[b] = self:_normPreserving(spinor)
        elseif self.activation_type == 'chiral_gate' then
            output[b] = self:_chiralGate(spinor)
        elseif self.activation_type == 'clifford_mix' then
            output[b] = self:_cliffordMix(spinor)
        elseif self.activation_type == 'spinor_relu' then
            output[b] = self:_spinorReLU(spinor)
        else
            output[b] = self:_normPreserving(spinor)
        end
    end

    -- Store rotation for backward
    if self.use_learnable then
        self._rotation = self:_buildSpinorRotation()
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

function SpinorActivation:updateGradInput(input, gradOutput)
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
        local dy = gradOutput_view[b]
        local x = self._input_view[b]

        if self.activation_type == 'norm_preserving' then
            -- Gradient through rotation: R^T * dy
            if self.use_learnable and self._rotation then
                gradInput[b] = torch.mv(self._rotation:t(), dy)
            else
                gradInput[b] = dy:clone()
            end

        elseif self.activation_type == 'spinor_relu' then
            local leak = self.use_learnable and self.leak_factor[1] or 0.01
            local norm = x:norm()
            local threshold = norm * 0.1

            for i = 1, self.dim do
                if x[i] < -threshold then
                    gradInput[b][i] = leak * dy[i]
                elseif x[i] < threshold then
                    local t = (x[i] + threshold) / (2 * threshold)
                    local dt = 1 / (2 * threshold)
                    gradInput[b][i] = dy[i] * (leak + (1 - leak) * t + x[i] * (1 - leak) * dt)
                else
                    gradInput[b][i] = dy[i]
                end
            end

        else
            -- Default: pass through rotation transpose
            if self.use_learnable and self._rotation then
                gradInput[b] = torch.mv(self._rotation:t(), dy)
            else
                gradInput[b] = dy:clone()
            end
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

function SpinorActivation:accGradParameters(input, gradOutput, scale)
    scale = scale or 1

    if not self.use_learnable then
        return
    end

    local gradOutput_tensor
    if type(gradOutput) == 'table' and gradOutput._type == 'ObserverseTensor' then
        gradOutput_tensor = gradOutput.fiber
    else
        gradOutput_tensor = gradOutput
    end

    local gradOutput_view = self._was_1d and gradOutput_tensor:view(1, -1) or gradOutput_tensor

    -- Gradient for rotation angles
    local R = self._rotation or self:_buildSpinorRotation()

    for a = 1, self.num_generators do
        local J_a = self.generators[a]
        -- dR/d_theta_a ≈ J_a * R for small angles
        local dR = torch.mm(J_a, R)

        local grad = 0
        for b = 1, self._batch_size do
            local x = self._input_view[b]
            local dy = gradOutput_view[b]
            -- d_loss/d_theta_a = dy^T * dR * x
            local dx = torch.mv(dR, x)
            grad = grad + torch.dot(dy, dx)
        end

        self.gradRotationAngles[a] = self.gradRotationAngles[a] + scale * grad
    end

    -- Gradient for chiral mix
    if self.use_chiral_decomposition and self.chiral_mix then
        local grad_mix = 0
        for b = 1, self._batch_size do
            local x = self._input_view[b]
            local dy = gradOutput_view[b]

            local left_x = x[{{1, self.chiral_dim}}]
            local right_x = x[{{self.chiral_dim + 1, self.dim}}]
            local left_dy = dy[{{1, self.chiral_dim}}]
            local right_dy = dy[{{self.chiral_dim + 1, self.dim}}]

            -- d_out_L/d_mix = -left + right, d_out_R/d_mix = -right + left
            grad_mix = grad_mix + torch.dot(left_dy, right_x - left_x)
            grad_mix = grad_mix + torch.dot(right_dy, left_x - right_x)
        end
        self.gradChiralMix[1] = self.gradChiralMix[1] + scale * grad_mix
    end

    -- Gradient for leak factor
    if self.activation_type == 'spinor_relu' and self.leak_factor then
        local grad_leak = 0
        for b = 1, self._batch_size do
            local x = self._input_view[b]
            local dy = gradOutput_view[b]
            local norm = x:norm()
            local threshold = norm * 0.1

            for i = 1, self.dim do
                if x[i] < -threshold then
                    grad_leak = grad_leak + dy[i] * x[i]
                elseif x[i] < threshold then
                    local t = (x[i] + threshold) / (2 * threshold)
                    grad_leak = grad_leak + dy[i] * x[i] * (1 - t)
                end
            end
        end
        self.gradLeakFactor[1] = self.gradLeakFactor[1] + scale * grad_leak
    end
end

-- ============================================================================
-- Parameter Initialization
-- ============================================================================

function SpinorActivation:reset()
    if self.use_learnable then
        -- Small random rotations
        self.rotation_angles:normal(0, 0.01)

        if self.use_chiral_decomposition and self.chiral_mix then
            self.chiral_mix:fill(0.1)  -- Small mixing
        end

        if self.activation_type == 'spinor_relu' and self.leak_factor then
            self.leak_factor:fill(0.01)
        end
    end

    self:zeroGradParameters()
end

function SpinorActivation:zeroGradParameters()
    if self.use_learnable then
        self.gradRotationAngles:zero()

        if self.use_chiral_decomposition and self.gradChiralMix then
            self.gradChiralMix:zero()
        end

        if self.activation_type == 'spinor_relu' and self.gradLeakFactor then
            self.gradLeakFactor:zero()
        end
    end
end

-- ============================================================================
-- Parameters
-- ============================================================================

function SpinorActivation:parameters()
    if not self.use_learnable then
        return {}, {}
    end

    local params = {self.rotation_angles}
    local gradParams = {self.gradRotationAngles}

    if self.use_chiral_decomposition and self.chiral_mix then
        table.insert(params, self.chiral_mix)
        table.insert(gradParams, self.gradChiralMix)
    end

    if self.activation_type == 'spinor_relu' and self.leak_factor then
        table.insert(params, self.leak_factor)
        table.insert(gradParams, self.gradLeakFactor)
    end

    return params, gradParams
end

-- ============================================================================
-- Utility Methods
-- ============================================================================

function SpinorActivation:getChiralComponents(spinor)
    local left = spinor[{{1, self.chiral_dim}}]
    local right = spinor[{{self.chiral_dim + 1, self.dim}}]
    return left, right
end

function SpinorActivation:getChirality(spinor)
    local left, right = self:getChiralComponents(spinor)
    local left_norm = left:norm() + 1e-8
    local right_norm = right:norm() + 1e-8
    return (left_norm - right_norm) / (left_norm + right_norm)
end

function SpinorActivation:__tostring()
    local str = string.format('%s(dim=%d, type=%s',
        torch.type(self), self.dim, self.activation_type)
    if self.use_chiral_decomposition then str = str .. ', chiral' end
    if self.use_learnable then
        str = str .. string.format(', generators=%d', self.num_generators)
    end
    str = str .. ')'
    return str
end

return SpinorActivation
