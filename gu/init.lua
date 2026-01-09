-- ============================================================================
-- GU: Geometric Unity Extension for Torch7u
-- ============================================================================
-- This module implements key mathematical structures from Eric Weinstein's
-- Geometric Unity theory, leveraging torch7u's nested tensor (nnn) system.
-- ============================================================================

local gu = {}

gu._VERSION = '1.0.0'
gu._DESCRIPTION = 'Geometric Unity Extension - Computational implementation of GU structures'

-- ============================================================================
-- Load Components
-- ============================================================================

gu.ObserverseTensor = require 'gu.ObserverseTensor'
gu.ChimericBundle = require 'gu.ChimericBundle'
gu.ShiabOperator = require 'gu.ShiabOperator'
gu.SwerveModule = require 'gu.SwerveModule'
gu.GaugeTransformer = require 'gu.GaugeTransformer'
gu.GeneralizedGaugeTransformer = require 'gu.GeneralizedGaugeTransformer'
gu.GULayer = require 'gu.GULayer'

-- Load Activations Module
gu.activations = require 'gu.activations'

-- Load Solution Module (Complete GU Solution)
gu.Solution = require 'gu.Solution'

-- Convenience: export Lagrangian to gu namespace
gu.Lagrangian = gu.Solution.Lagrangian

-- ============================================================================
-- Constants
-- ============================================================================

gu.BASE_DIM = 4       -- Dimension of base spacetime (X^4)
gu.FIBER_DIM = 10     -- Dimension of metric fiber
gu.CHIMERIC_DIM = 14  -- Total dimension of Chimeric Bundle (4 + 10)
gu.SPINOR_DIM = 128   -- Dimension of GU spinors (2^7 from 14D)

-- ============================================================================
-- Plugin Registration
-- ============================================================================

function gu.install()
    if torch7u and torch7u.plugins then
        torch7u.plugins.register('gu', gu)
        torch7u.utils.log("INFO", "Geometric Unity (gu) module installed as a plugin.", "gu")
    end
end

-- ============================================================================
-- Utility Functions
-- ============================================================================

-- Check if input is an ObserverseTensor
function gu.isObserverse(input)
    return type(input) == 'table' and input.base ~= nil and input.fiber ~= nil
end

-- Create a random ObserverseTensor for testing
function gu.randomObserverse(batch_size)
    batch_size = batch_size or 1
    local base = torch.randn(batch_size, gu.BASE_DIM)
    local fiber = torch.randn(batch_size, gu.FIBER_DIM)
    return gu.ObserverseTensor.create(base, fiber)
end

-- Create a random Chimeric Bundle vector
function gu.randomChimeric(batch_size)
    batch_size = batch_size or 1
    return torch.randn(batch_size, gu.CHIMERIC_DIM)
end

-- Create a random GU spinor
function gu.randomSpinor(batch_size)
    batch_size = batch_size or 1
    return torch.randn(batch_size, gu.SPINOR_DIM)
end

-- Convert Chimeric Bundle vector to ObserverseTensor
function gu.chimericToObserverse(chimeric)
    local base = chimeric:narrow(2, 1, gu.BASE_DIM)
    local fiber = chimeric:narrow(2, gu.BASE_DIM + 1, gu.FIBER_DIM)
    return gu.ObserverseTensor.create(base, fiber)
end

-- Convert ObserverseTensor to Chimeric Bundle vector
function gu.observerseToChimeric(observerse)
    return torch.cat({observerse.base, observerse.fiber}, 2)
end

-- ============================================================================
-- NNN Integration
-- ============================================================================

-- Transform any nn module to work with ObserverseTensors
function gu.transform(module, config)
    config = config or {}
    local nnn = require 'nnn'
    
    local GUOperator, parent = torch.class('gu.GUOperator', 'nn.Module')
    
    function GUOperator:__init(wrappedModule)
        parent.__init(self)
        self.module = wrappedModule
        self.applyTo = config.applyTo or 'both'  -- 'base', 'fiber', or 'both'
    end
    
    function GUOperator:updateOutput(input)
        if gu.isObserverse(input) then
            local base_out = input.base
            local fiber_out = input.fiber
            
            if self.applyTo == 'base' or self.applyTo == 'both' then
                base_out = self.module:forward(input.base)
            end
            if self.applyTo == 'fiber' or self.applyTo == 'both' then
                fiber_out = self.module:forward(input.fiber)
            end
            
            self.output = gu.ObserverseTensor.create(base_out, fiber_out)
        else
            self.output = self.module:forward(input)
        end
        return self.output
    end
    
    function GUOperator:updateGradInput(input, gradOutput)
        if gu.isObserverse(input) then
            local base_grad = gradOutput.base
            local fiber_grad = gradOutput.fiber
            
            if self.applyTo == 'base' or self.applyTo == 'both' then
                base_grad = self.module:backward(input.base, gradOutput.base)
            end
            if self.applyTo == 'fiber' or self.applyTo == 'both' then
                fiber_grad = self.module:backward(input.fiber, gradOutput.fiber)
            end
            
            self.gradInput = gu.ObserverseTensor.create(base_grad, fiber_grad)
        else
            self.gradInput = self.module:backward(input, gradOutput)
        end
        return self.gradInput
    end
    
    return GUOperator(module)
end

-- ============================================================================
-- Model Building Helpers
-- ============================================================================

-- Create a simple GU model
function gu.createModel(config)
    config = config or {}
    local num_layers = config.num_layers or 3
    local hidden_dim = config.hidden_dim or gu.FIBER_DIM

    local model = nn.Sequential()

    for i = 1, num_layers do
        model:add(gu.GULayer(hidden_dim))
        if config.activation then
            model:add(gu.transform(nn.ReLU(), {applyTo = 'fiber'}))
        end
    end

    return model
end

-- Create a Generalized Gauge Transformer model
function gu.createGaugeTransformerModel(config)
    config = config or {}
    local num_layers = config.num_layers or 3
    local fiber_dim = config.fiber_dim or gu.FIBER_DIM
    local base_dim = config.base_dim or gu.BASE_DIM
    local num_heads = config.num_heads or 4
    local structure_group = config.structure_group or 'SO'
    local use_attention = config.use_attention ~= false
    local use_residual = config.use_residual ~= false
    local use_layernorm = config.use_layernorm ~= false
    local dropout = config.dropout or 0.1

    local model = nn.Sequential()

    for i = 1, num_layers do
        model:add(gu.GeneralizedGaugeTransformer(fiber_dim, {
            base_dim = base_dim,
            num_heads = num_heads,
            structure_group = structure_group,
            use_attention = use_attention,
            use_residual = use_residual,
            use_layernorm = use_layernorm,
            dropout = dropout,
            use_connection = config.use_connection or false,
            use_curvature_reg = config.use_curvature_reg or false
        }))
    end

    return model
end

-- Create a hybrid model with both GU layers and Generalized Gauge Transformers
function gu.createHybridModel(config)
    config = config or {}
    local num_gu_layers = config.num_gu_layers or 2
    local num_transformer_layers = config.num_transformer_layers or 2
    local fiber_dim = config.fiber_dim or gu.FIBER_DIM
    local base_dim = config.base_dim or gu.BASE_DIM

    local model = nn.Sequential()

    -- GU layers for initial processing
    for i = 1, num_gu_layers do
        model:add(gu.GULayer(fiber_dim, {
            base_dim = base_dim,
            use_swerve = true,
            use_gauge = true,
            activation = 'tanh',
            use_residual = true
        }))
    end

    -- Generalized Gauge Transformer layers
    for i = 1, num_transformer_layers do
        model:add(gu.GeneralizedGaugeTransformer(fiber_dim, {
            base_dim = base_dim,
            num_heads = config.num_heads or 4,
            structure_group = config.structure_group or 'SO',
            use_attention = true,
            use_residual = true,
            use_layernorm = true,
            dropout = config.dropout or 0.1
        }))
    end

    return model
end

-- ============================================================================
-- Information Display
-- ============================================================================

function gu.info()
    print("=======================================================")
    print("GU: Geometric Unity Extension for Torch7u")
    print("Version: " .. gu._VERSION)
    print("=======================================================")
    print("\nDimensions:")
    print(string.format("  Base Space (X^4):     %d", gu.BASE_DIM))
    print(string.format("  Fiber Space:          %d", gu.FIBER_DIM))
    print(string.format("  Chimeric Bundle:      %d", gu.CHIMERIC_DIM))
    print(string.format("  GU Spinors:           %d", gu.SPINOR_DIM))
    print("\nCore Components:")
    print("  - ObserverseTensor: Two-space structure (base + fiber)")
    print("  - ChimericBundle: 14D bundle vectors")
    print("  - ShiabOperator: Ship-in-a-bottle operator")
    print("  - SwerveModule: Swervature computation")
    print("  - GaugeTransformer: Gauge transformations (tilted, Lie algebra, parallel transport)")
    print("  - GeneralizedGaugeTransformer: Full transformer architecture with multi-head attention")
    print("  - GULayer: Composite GU dynamics layer")
    print("\nThe Solution (gu.Solution):")
    print("  - Lagrangian: Unified GU action principle")
    print("  - FieldEquations: Shiab(F_A) + *T = 0")
    print("  - EndogenousObserver: Emergent spacetime embedding")
    print("  - EinsteinYangMills: Gravity-gauge unification")
    print("  - SpinorFieldEquations: 128D fermion dynamics")
    print("  - CompleteSolution: Full GU solver")
    print("\nSpecialized Activations (gu.activations):")
    print("  - LieAlgebraActivation: Lie group structure-preserving")
    print("  - CurvatureGate: Curvature-aware gating")
    print("  - ParallelTransportActivation: Fiber-base coupled transport")
    print("  - GaugeEquivariantActivation: Gauge symmetry-respecting")
    print("  - SpinorActivation: Clifford algebra-aware spinor activation")
    print("\nGauge Transformer Features:")
    print("  - Structure groups: GL, SO, SU, Spin, U")
    print("  - Multi-head gauge attention")
    print("  - Lie algebra parameterization with exponential map")
    print("  - Connection-based parallel transport")
    print("  - Curvature regularization")
    print("=======================================================")
    print("\nTo solve Geometric Unity:")
    print("  local solver = gu.Solution.create()")
    print("  local result = solver:solve(initial_conditions)")
    print("  solver:display(result)")
    print("\nFor theory visualization:")
    print("  gu.Solution.visualize()")
    print("=======================================================")
end

-- ============================================================================
-- Auto-install if torch7u is available
-- ============================================================================

if torch7u then
    gu.install()
end

return gu
