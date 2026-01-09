-- ============================================================================
-- GU Activations: Specialized Activation Functions for Geometric Unity
-- ============================================================================
-- This module provides geometry-respecting activation functions designed
-- specifically for the Geometric Unity framework.
--
-- Available Activations:
--   1. LieAlgebraActivation    - Lie group structure-preserving activation
--   2. CurvatureGate           - Curvature-aware gating activation
--   3. ParallelTransportActivation - Fiber-base coupled transport activation
--   4. GaugeEquivariantActivation  - Gauge symmetry-respecting activation
--   5. SpinorActivation        - Clifford algebra-aware spinor activation
--
-- Usage:
--   local activations = require 'gu.activations'
--   local act = activations.LieAlgebraActivation(10, {structure_group = 'SO'})
--   local output = act:forward(input)
--
-- Or through the main GU module:
--   local gu = require 'gu'
--   local act = gu.activations.CurvatureGate(10)
-- ============================================================================

local activations = {}

activations._VERSION = '1.0.0'
activations._DESCRIPTION = 'Geometry-respecting activation functions for GU'

-- ============================================================================
-- Load Activation Modules
-- ============================================================================

activations.LieAlgebraActivation = require 'gu.activations.LieAlgebraActivation'
activations.CurvatureGate = require 'gu.activations.CurvatureGate'
activations.ParallelTransportActivation = require 'gu.activations.ParallelTransportActivation'
activations.GaugeEquivariantActivation = require 'gu.activations.GaugeEquivariantActivation'
activations.SpinorActivation = require 'gu.activations.SpinorActivation'

-- ============================================================================
-- Factory Functions for Easy Construction
-- ============================================================================

-- Create a Lie algebra activation with default settings
function activations.lieAlgebra(dim, structure_group, activation_type)
    return activations.LieAlgebraActivation(dim, {
        structure_group = structure_group or 'SO',
        activation_type = activation_type or 'softplus'
    })
end

-- Create a curvature gate with default settings
function activations.curvatureGate(dim, gate_type, mode)
    return activations.CurvatureGate(dim, {
        gate_type = gate_type or 'sigmoid',
        mode = mode or 'gate'
    })
end

-- Create a parallel transport activation with default settings
function activations.parallelTransport(fiber_dim, base_dim, activation_type)
    return activations.ParallelTransportActivation(fiber_dim, {
        base_dim = base_dim or 4,
        activation_type = activation_type or 'tanh'
    })
end

-- Create a gauge equivariant activation with default settings
function activations.gaugeEquivariant(dim, structure_group, mode)
    return activations.GaugeEquivariantActivation(dim, {
        structure_group = structure_group or 'SO',
        mode = mode or 'equivariant'
    })
end

-- Create a spinor activation with default settings
function activations.spinor(dim, activation_type)
    return activations.SpinorActivation(dim or 128, {
        activation_type = activation_type or 'norm_preserving'
    })
end

-- ============================================================================
-- Preset Configurations
-- ============================================================================

-- Standard GU fiber activation (10D with SO structure)
function activations.guFiber(config)
    config = config or {}
    local dim = config.dim or 10
    local activation_type = config.type or 'gauge_equivariant'

    if activation_type == 'lie_algebra' then
        return activations.lieAlgebra(dim, 'SO', 'softplus')
    elseif activation_type == 'curvature_gate' then
        return activations.curvatureGate(dim, 'sigmoid', 'gate')
    elseif activation_type == 'gauge_equivariant' then
        return activations.gaugeEquivariant(dim, 'SO', 'equivariant')
    else
        return activations.gaugeEquivariant(dim, 'SO', 'equivariant')
    end
end

-- Standard GU base activation (4D)
function activations.guBase(config)
    config = config or {}
    local dim = config.dim or 4

    return activations.gaugeEquivariant(dim, 'SO', 'equivariant')
end

-- Chimeric bundle activation (14D)
function activations.guChimeric(config)
    config = config or {}

    return activations.parallelTransport(10, 4, 'gelu')
end

-- Full spinor activation (128D)
function activations.guSpinor(config)
    config = config or {}
    local activation_type = config.type or 'chiral_gate'

    return activations.spinor(128, activation_type)
end

-- ============================================================================
-- Composite Activation Builder
-- ============================================================================

-- Build a sequential activation chain
function activations.chain(...)
    local modules = {...}
    local chain = nn.Sequential()

    for _, mod in ipairs(modules) do
        chain:add(mod)
    end

    return chain
end

-- Build a parallel activation (applies multiple and combines)
function activations.parallel(combine_mode, ...)
    local modules = {...}
    local parallel = nn.ConcatTable()

    for _, mod in ipairs(modules) do
        parallel:add(mod)
    end

    local combined = nn.Sequential()
    combined:add(parallel)

    if combine_mode == 'add' then
        combined:add(nn.CAddTable())
    elseif combine_mode == 'mul' then
        combined:add(nn.CMulTable())
    elseif combine_mode == 'cat' then
        combined:add(nn.JoinTable(2))
    else
        combined:add(nn.CAddTable())
    end

    return combined
end

-- ============================================================================
-- Information Display
-- ============================================================================

function activations.info()
    print("=======================================================")
    print("GU Activations: Geometry-Respecting Activation Functions")
    print("Version: " .. activations._VERSION)
    print("=======================================================")
    print("\nAvailable Activations:")
    print("  1. LieAlgebraActivation")
    print("     - Preserves Lie group structure")
    print("     - Applies activation in Lie algebra space")
    print("     - Supports SO, SU, Spin, U, GL groups")
    print("")
    print("  2. CurvatureGate")
    print("     - Gates features based on field strength")
    print("     - Curvature-aware signal modulation")
    print("     - Modes: gate, amplify, modulate")
    print("")
    print("  3. ParallelTransportActivation")
    print("     - Couples fiber and base transformations")
    print("     - Learnable connection coefficients")
    print("     - Optional holonomy tracking")
    print("")
    print("  4. GaugeEquivariantActivation")
    print("     - Preserves gauge symmetry: f(g*x) = g*f(x)")
    print("     - Norm-based radial transformations")
    print("     - Modes: equivariant, invariant, hybrid")
    print("")
    print("  5. SpinorActivation")
    print("     - For 128D GU spinor representations")
    print("     - Respects Clifford algebra structure")
    print("     - Chiral decomposition support")
    print("=======================================================")
    print("\nPresets:")
    print("  activations.guFiber()    - For 10D fiber space")
    print("  activations.guBase()     - For 4D base space")
    print("  activations.guChimeric() - For 14D chimeric bundle")
    print("  activations.guSpinor()   - For 128D spinor space")
    print("=======================================================")
end

return activations
