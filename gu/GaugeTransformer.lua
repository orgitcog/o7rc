-- ============================================================================
-- GaugeTransformer: Gauge Transformations for Geometric Unity
-- ============================================================================
-- Implements the action of the "tilted gauge group" on GU fields.
-- In GU, the gauge group acts on both the base and fiber components
-- of the Observerse, with the specific structure determined by the
-- geometry of the Chimeric Bundle.
--
-- Gauge Types:
--   - 'standard': Classical gauge transformation (fiber only)
--   - 'tilted': GU's tilted gauge group (coupled base-fiber action)
--   - 'inhomogeneous': Affine gauge transformations with translation
--   - 'lie_algebra': Parameterized by Lie algebra elements (exp map)
--   - 'parallel_transport': Gauge via parallel transport connection
-- ============================================================================

local GaugeTransformer, parent = torch.class('gu.GaugeTransformer', 'nn.Module')

function GaugeTransformer:__init(dim, config)
    parent.__init(self)

    config = config or {}

    self.dim = dim
    self.gauge_type = config.gauge_type or 'tilted'  -- 'tilted', 'standard', 'inhomogeneous', 'lie_algebra', 'parallel_transport'
    self.learnable = config.learnable ~= false  -- Default true
    self.structure_group = config.structure_group or 'GL'  -- 'GL', 'SO', 'SU', 'Spin'

    -- Gauge transformation matrix
    -- For tilted gauge group, this is not a standard Lie group element
    self.gauge_matrix = torch.Tensor(dim, dim)
    self.gradGaugeMatrix = torch.Tensor(dim, dim)

    -- For Lie algebra parameterization, we store the algebra elements
    if self.gauge_type == 'lie_algebra' then
        -- Antisymmetric generators for SO(n)
        local num_generators = dim * (dim - 1) / 2
        self.lie_params = torch.Tensor(num_generators)
        self.gradLieParams = torch.Tensor(num_generators)
    end

    -- For parallel transport, we store connection coefficients
    if self.gauge_type == 'parallel_transport' then
        self.connection = torch.Tensor(dim, dim, dim)  -- Christoffel-like symbols
        self.gradConnection = torch.Tensor(dim, dim, dim)
    end

    -- For inhomogeneous gauge group, we also have a translation component
    if self.gauge_type == 'inhomogeneous' then
        self.translation = torch.Tensor(dim)
        self.gradTranslation = torch.Tensor(dim)
    end

    -- Base space transformation (for full Observerse transformation)
    self.base_dim = config.base_dim or 4
    self.base_matrix = torch.Tensor(self.base_dim, self.base_dim)
    self.gradBaseMatrix = torch.Tensor(self.base_dim, self.base_dim)

    -- Coupling strength between base and fiber (for tilted gauge)
    self.coupling_strength = config.coupling_strength or 0.1

    self:reset()
end

function GaugeTransformer:reset()
    -- Initialize gauge matrix close to identity
    self.gauge_matrix:eye(self.dim)

    -- Add small perturbation for learnable case
    if self.learnable then
        self.gauge_matrix:add(torch.randn(self.dim, self.dim):mul(0.01))
    end

    -- Initialize base matrix as identity
    self.base_matrix:eye(self.base_dim)

    if self.gauge_type == 'inhomogeneous' then
        self.translation:zero()
    end

    -- Initialize Lie algebra parameters
    if self.gauge_type == 'lie_algebra' then
        self.lie_params:zero()
        if self.learnable then
            self.lie_params:normal(0, 0.01)
        end
        self:_updateGaugeFromLie()
    end

    -- Initialize connection coefficients
    if self.gauge_type == 'parallel_transport' then
        self.connection:zero()
        if self.learnable then
            self.connection:normal(0, 0.01)
        end
    end
end

-- Compute gauge matrix from Lie algebra parameters (exponential map)
function GaugeTransformer:_updateGaugeFromLie()
    if self.gauge_type ~= 'lie_algebra' then return end

    -- Build antisymmetric matrix from parameters
    local A = torch.zeros(self.dim, self.dim)
    local idx = 1
    for i = 1, self.dim do
        for j = i + 1, self.dim do
            A[i][j] = self.lie_params[idx]
            A[j][i] = -self.lie_params[idx]
            idx = idx + 1
        end
    end

    -- Compute matrix exponential via Pade approximation (order 6)
    self.gauge_matrix = self:_matrixExp(A)
end

-- Matrix exponential via Pade approximation
function GaugeTransformer:_matrixExp(A, order)
    order = order or 6
    local n = A:size(1)
    local I = torch.eye(n)

    -- Scale matrix to improve convergence
    local norm_A = A:norm()
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

-- Compute parallel transport transformation
function GaugeTransformer:_parallelTransport(fiber, base)
    if self.gauge_type ~= 'parallel_transport' then
        return torch.mm(fiber, self.gauge_matrix:t())
    end

    local batch_size = fiber:size(1)
    local result = fiber:clone()

    -- Apply connection-based transport: v' = v + Gamma^i_jk * base^j * v^k
    for b = 1, batch_size do
        for i = 1, self.dim do
            local correction = 0
            for j = 1, math.min(self.base_dim, self.dim) do
                for k = 1, self.dim do
                    correction = correction + self.connection[i][j][k] * base[b][j] * fiber[b][k]
                end
            end
            result[b][i] = result[b][i] + correction
        end
    end

    return result
end

function GaugeTransformer:updateOutput(input)
    local is_observerse = type(input) == 'table' and input._type == 'ObserverseTensor'

    -- Update gauge matrix from Lie algebra if needed
    if self.gauge_type == 'lie_algebra' then
        self:_updateGaugeFromLie()
    end

    if is_observerse then
        -- Transform both base and fiber components
        local base_tensor = input.base
        local fiber_tensor = input.fiber

        -- Ensure 2D
        local base_was_1d = base_tensor:dim() == 1
        local fiber_was_1d = fiber_tensor:dim() == 1

        local base_view = base_was_1d and base_tensor:view(1, -1) or base_tensor
        local fiber_view = fiber_was_1d and fiber_tensor:view(1, -1) or fiber_tensor

        local transformed_fiber
        local transformed_base

        if self.gauge_type == 'parallel_transport' then
            -- Use connection-based parallel transport
            transformed_fiber = self:_parallelTransport(fiber_view, base_view)
            transformed_base = torch.mm(base_view, self.base_matrix:t())
        elseif self.gauge_type == 'tilted' then
            -- Tilted gauge: coupled transformation of base and fiber
            -- The gauge acts on both spaces with a coupling term
            transformed_fiber = torch.mm(fiber_view, self.gauge_matrix:t())
            transformed_base = torch.mm(base_view, self.base_matrix:t())
            -- Add coupling term: fiber influences base transformation
            local coupling_term = torch.mm(fiber_view:narrow(2, 1, self.base_dim),
                                           torch.eye(self.base_dim):mul(self.coupling_strength))
            transformed_base:add(coupling_term)
        else
            -- Standard or Lie algebra gauge transformation
            transformed_fiber = torch.mm(fiber_view, self.gauge_matrix:t())
            transformed_base = torch.mm(base_view, self.base_matrix:t())
        end

        -- Add translation for inhomogeneous gauge
        if self.gauge_type == 'inhomogeneous' then
            transformed_fiber:add(self.translation:view(1, -1):expandAs(transformed_fiber))
        end

        -- Reshape if needed
        if base_was_1d then
            transformed_base = transformed_base:view(self.base_dim)
        end
        if fiber_was_1d then
            transformed_fiber = transformed_fiber:view(self.dim)
        end

        local ObserverseTensor = require 'gu.ObserverseTensor'
        self.output = ObserverseTensor.create(transformed_base, transformed_fiber)
    else
        -- Transform single tensor
        local was_1d = input:dim() == 1
        local input_view = was_1d and input:view(1, -1) or input

        self.output = torch.mm(input_view, self.gauge_matrix:t())

        if self.gauge_type == 'inhomogeneous' then
            self.output:add(self.translation:view(1, -1):expandAs(self.output))
        end

        if was_1d then
            self.output = self.output:view(self.dim)
        end
    end

    return self.output
end

function GaugeTransformer:updateGradInput(input, gradOutput)
    local is_observerse = type(input) == 'table' and input._type == 'ObserverseTensor'
    
    if is_observerse then
        local base_tensor = input.base
        local fiber_tensor = input.fiber
        local gradBase = gradOutput.base
        local gradFiber = gradOutput.fiber
        
        -- Ensure 2D
        local base_was_1d = base_tensor:dim() == 1
        local fiber_was_1d = fiber_tensor:dim() == 1
        
        local gradBase_view = base_was_1d and gradBase:view(1, -1) or gradBase
        local gradFiber_view = fiber_was_1d and gradFiber:view(1, -1) or gradFiber
        
        -- Backprop through gauge transformation
        local grad_fiber_input = torch.mm(gradFiber_view, self.gauge_matrix)
        local grad_base_input = torch.mm(gradBase_view, self.base_matrix)
        
        -- Reshape if needed
        if base_was_1d then
            grad_base_input = grad_base_input:view(self.base_dim)
        end
        if fiber_was_1d then
            grad_fiber_input = grad_fiber_input:view(self.dim)
        end
        
        local ObserverseTensor = require 'gu.ObserverseTensor'
        self.gradInput = ObserverseTensor.create(grad_base_input, grad_fiber_input)
    else
        local was_1d = input:dim() == 1
        local gradOutput_view = was_1d and gradOutput:view(1, -1) or gradOutput
        
        self.gradInput = torch.mm(gradOutput_view, self.gauge_matrix)
        
        if was_1d then
            self.gradInput = self.gradInput:view(self.dim)
        end
    end
    
    return self.gradInput
end

function GaugeTransformer:accGradParameters(input, gradOutput, scale)
    scale = scale or 1
    
    if not self.learnable then
        return
    end
    
    local is_observerse = type(input) == 'table' and input._type == 'ObserverseTensor'
    
    if is_observerse then
        local fiber_tensor = input.fiber
        local gradFiber = gradOutput.fiber
        local base_tensor = input.base
        local gradBase = gradOutput.base
        
        -- Ensure 2D
        local fiber_was_1d = fiber_tensor:dim() == 1
        local base_was_1d = base_tensor:dim() == 1
        
        local fiber_view = fiber_was_1d and fiber_tensor:view(1, -1) or fiber_tensor
        local gradFiber_view = fiber_was_1d and gradFiber:view(1, -1) or gradFiber
        local base_view = base_was_1d and base_tensor:view(1, -1) or base_tensor
        local gradBase_view = base_was_1d and gradBase:view(1, -1) or gradBase
        
        -- Gradient for gauge_matrix
        self.gradGaugeMatrix:addmm(scale, gradFiber_view:t(), fiber_view)
        
        -- Gradient for base_matrix
        self.gradBaseMatrix:addmm(scale, gradBase_view:t(), base_view)
        
        -- Gradient for translation
        if self.gauge_type == 'inhomogeneous' then
            self.gradTranslation:add(scale, gradFiber_view:sum(1):view(self.dim))
        end
    else
        local was_1d = input:dim() == 1
        local input_view = was_1d and input:view(1, -1) or input
        local gradOutput_view = was_1d and gradOutput:view(1, -1) or gradOutput
        
        -- Gradient for gauge_matrix
        self.gradGaugeMatrix:addmm(scale, gradOutput_view:t(), input_view)
        
        -- Gradient for translation
        if self.gauge_type == 'inhomogeneous' then
            self.gradTranslation:add(scale, gradOutput_view:sum(1):view(self.dim))
        end
    end
end

function GaugeTransformer:parameters()
    local params = {self.gauge_matrix, self.base_matrix}
    local gradParams = {self.gradGaugeMatrix, self.gradBaseMatrix}

    if self.gauge_type == 'inhomogeneous' then
        table.insert(params, self.translation)
        table.insert(gradParams, self.gradTranslation)
    end

    if self.gauge_type == 'lie_algebra' then
        table.insert(params, self.lie_params)
        table.insert(gradParams, self.gradLieParams)
    end

    if self.gauge_type == 'parallel_transport' then
        table.insert(params, self.connection)
        table.insert(gradParams, self.gradConnection)
    end

    return params, gradParams
end

-- Get the current gauge group element
function GaugeTransformer:getGaugeElement()
    if self.gauge_type == 'lie_algebra' then
        self:_updateGaugeFromLie()
    end
    return self.gauge_matrix:clone()
end

-- Set gauge from matrix (will project to Lie algebra if needed)
function GaugeTransformer:setGaugeElement(matrix)
    self.gauge_matrix:copy(matrix)
    if self.gauge_type == 'lie_algebra' then
        -- Project to SO(n) via polar decomposition and extract Lie params
        local U, S, V = torch.svd(matrix)
        local R = torch.mm(U, V:t())
        self.gauge_matrix:copy(R)
        -- Extract Lie algebra parameters (approximate)
        local A = torch.log(R)  -- matrix logarithm (simplified)
        local idx = 1
        for i = 1, self.dim do
            for j = i + 1, self.dim do
                self.lie_params[idx] = (A[i][j] - A[j][i]) / 2
                idx = idx + 1
            end
        end
    end
end

-- Compute curvature of the gauge connection
function GaugeTransformer:curvature()
    if self.gauge_type == 'parallel_transport' then
        -- Compute Riemann curvature from connection
        -- R^i_jkl = d_k Gamma^i_jl - d_l Gamma^i_jk + Gamma^i_mk Gamma^m_jl - Gamma^i_ml Gamma^m_jk
        local R = torch.zeros(self.dim, self.dim, self.dim, self.dim)
        for i = 1, self.dim do
            for j = 1, self.dim do
                for k = 1, self.dim do
                    for l = 1, self.dim do
                        -- Simplified: just the non-derivative terms
                        for m = 1, self.dim do
                            R[i][j][k][l] = R[i][j][k][l]
                                + self.connection[i][m][k] * self.connection[m][j][l]
                                - self.connection[i][m][l] * self.connection[m][j][k]
                        end
                    end
                end
            end
        end
        return R
    else
        -- For matrix gauge, curvature is related to commutator
        -- F = dA + A ^ A (simplified to A^2 - A^T * A for illustration)
        local A = self.gauge_matrix
        local F = torch.mm(A, A) - torch.mm(A:t(), A)
        return F
    end
end

function GaugeTransformer:__tostring()
    return string.format('%s(dim=%d, base_dim=%d, type=%s, group=%s, learnable=%s)',
        torch.type(self), self.dim, self.base_dim, self.gauge_type,
        self.structure_group, tostring(self.learnable))
end

return GaugeTransformer
