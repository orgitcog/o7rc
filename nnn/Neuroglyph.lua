-- ============================================================================
-- Geonestor Neuroglyph
-- ============================================================================
-- A Geonestor Neuroglyph is a geometric nested tensor neural gauge-awareness
-- symmetry structure. It represents the unified formalism combining:
--
--   GEO    - Geometric Unity (Observerse, gauge transformations, fiber bundles)
--   NESTOR - Nested Tensor structures (recursive tree-shaped data)
--   NEURO  - Neural network operations (learnable transformations)
--   GLYPH  - Symbolic representation (type signatures, symmetry invariants)
--
-- The Neuroglyph captures the full symmetry structure of a nested geometric
-- neural computation, including:
--   - The tree topology of nested ObserverseTensors
--   - Gauge symmetry groups acting on fibers
--   - Type signatures from prime factorization
--   - Learnable parameters respecting geometric constraints
--
-- ============================================================================

local Neuroglyph = {}
Neuroglyph.__index = Neuroglyph

-- ============================================================================
-- Constants
-- ============================================================================

Neuroglyph.VERSION = '1.0.0'
Neuroglyph.DESCRIPTION = 'Geometric Nested Tensor Neural Gauge-Awareness Symmetry'

-- Symmetry group types
Neuroglyph.GAUGE_GROUPS = {
    GL = 'General Linear',
    SO = 'Special Orthogonal',
    SU = 'Special Unitary',
    Spin = 'Spin Group',
    U = 'Unitary'
}

-- Glyph type categories
Neuroglyph.GLYPH_TYPES = {
    LEAF = 'leaf',           -- Single ObserverseTensor
    BRANCH = 'branch',       -- Nested structure
    COMPOSITE = 'composite', -- Mixed structure
    SYMMETRIC = 'symmetric'  -- Gauge-symmetric structure
}

-- ============================================================================
-- Constructor
-- ============================================================================

function Neuroglyph.create(config)
    config = config or {}

    local self = setmetatable({}, Neuroglyph)

    -- Core properties
    self.name = config.name or 'unnamed'
    self.glyphType = config.glyphType or Neuroglyph.GLYPH_TYPES.LEAF

    -- Geometric properties
    self.baseDim = config.baseDim or 4
    self.fiberDim = config.fiberDim or 10
    self.gaugeGroup = config.gaugeGroup or 'SO'

    -- Nested structure properties
    self.depth = config.depth or 0
    self.topology = config.topology or {}  -- Tree structure signature

    -- Neural properties
    self.layers = config.layers or {}
    self.parameters = {}
    self.gradParameters = {}

    -- Symmetry invariants
    self.invariants = config.invariants or {}

    -- Type signature (from prime factorization)
    self.typeSignature = config.typeSignature or nil

    -- Internal state
    self._type = 'Neuroglyph'
    self._initialized = false

    return self
end

-- ============================================================================
-- Static Methods
-- ============================================================================

-- Check if object is a Neuroglyph
function Neuroglyph.isNeuroglyph(obj)
    return type(obj) == 'table' and obj._type == 'Neuroglyph'
end

-- Create from nested ObserverseTensor structure
function Neuroglyph.fromNested(nested, config)
    config = config or {}
    local nnn = require 'nnn'

    -- Analyze structure
    local depth = nnn.gu.depth(nested)
    local count = nnn.gu.count(nested)
    local topology = Neuroglyph._extractTopology(nested)

    -- Determine glyph type
    local glyphType
    if depth == 0 then
        glyphType = Neuroglyph.GLYPH_TYPES.LEAF
    elseif depth == 1 then
        glyphType = Neuroglyph.GLYPH_TYPES.BRANCH
    else
        glyphType = Neuroglyph.GLYPH_TYPES.COMPOSITE
    end

    -- Extract dimensions from first ObserverseTensor
    local first = nnn.gu.flatten(nested)[1]
    local baseDim = first and first.base:size(first.base:dim()) or 4
    local fiberDim = first and first.fiber:size(first.fiber:dim()) or 10

    return Neuroglyph.create({
        name = config.name or 'derived',
        glyphType = glyphType,
        baseDim = baseDim,
        fiberDim = fiberDim,
        depth = depth,
        topology = topology,
        gaugeGroup = config.gaugeGroup or 'SO',
        invariants = Neuroglyph._computeInvariants(nested)
    })
end

-- Create from a model
function Neuroglyph.fromModel(model, config)
    config = config or {}

    local layers = {}
    if model.modules then
        for i, m in ipairs(model.modules) do
            table.insert(layers, {
                index = i,
                type = torch.type(m),
                params = m:parameters() and #m:parameters() or 0
            })
        end
    end

    return Neuroglyph.create({
        name = config.name or 'model',
        glyphType = Neuroglyph.GLYPH_TYPES.COMPOSITE,
        baseDim = config.baseDim or 4,
        fiberDim = config.fiberDim or 10,
        layers = layers,
        gaugeGroup = config.gaugeGroup or 'SO'
    })
end

-- Extract topology signature from nested structure
function Neuroglyph._extractTopology(nested)
    local nnn = require 'nnn'

    if nnn.gu.isObserverse(nested) then
        return {type = 'leaf', dim = {nested.base:dim(), nested.fiber:dim()}}
    elseif type(nested) == 'table' then
        local children = {}
        for k, v in pairs(nested) do
            children[k] = Neuroglyph._extractTopology(v)
        end
        return {type = 'branch', children = children}
    else
        return {type = 'unknown'}
    end
end

-- Compute symmetry invariants
function Neuroglyph._computeInvariants(nested)
    local nnn = require 'nnn'
    local invariants = {}

    -- Count invariant
    invariants.count = nnn.gu.count(nested)

    -- Depth invariant
    invariants.depth = nnn.gu.depth(nested)

    -- Norm invariants (gauge-invariant quantities)
    local totalBaseNorm = 0
    local totalFiberNorm = 0
    local list = nnn.gu.flatten(nested)

    for _, obs in ipairs(list) do
        totalBaseNorm = totalBaseNorm + obs.base:norm()
        totalFiberNorm = totalFiberNorm + obs.fiber:norm()
    end

    invariants.baseNorm = totalBaseNorm
    invariants.fiberNorm = totalFiberNorm
    invariants.totalNorm = math.sqrt(totalBaseNorm^2 + totalFiberNorm^2)

    return invariants
end

-- ============================================================================
-- Instance Methods
-- ============================================================================

-- Get glyph signature (unique identifier based on structure)
function Neuroglyph:signature()
    local sig = string.format(
        "G[%s:%d|%d|%s|d%d]",
        self.name,
        self.baseDim,
        self.fiberDim,
        self.gaugeGroup,
        self.depth
    )
    return sig
end

-- Get prime factorization type signature
function Neuroglyph:primeSignature()
    if self.typeSignature then
        return self.typeSignature
    end

    -- Compute from dimensions
    local function primeFactors(n)
        local factors = {}
        local d = 2
        while d * d <= n do
            while n % d == 0 do
                table.insert(factors, d)
                n = n / d
            end
            d = d + 1
        end
        if n > 1 then
            table.insert(factors, n)
        end
        return factors
    end

    local baseFactors = primeFactors(self.baseDim)
    local fiberFactors = primeFactors(self.fiberDim)

    self.typeSignature = {
        base = baseFactors,
        fiber = fiberFactors,
        combined = {}
    }

    -- Combined signature
    for _, f in ipairs(baseFactors) do
        table.insert(self.typeSignature.combined, f)
    end
    for _, f in ipairs(fiberFactors) do
        table.insert(self.typeSignature.combined, f)
    end

    return self.typeSignature
end

-- Check gauge symmetry compatibility
function Neuroglyph:isGaugeCompatible(other)
    if not Neuroglyph.isNeuroglyph(other) then
        return false
    end
    return self.gaugeGroup == other.gaugeGroup
end

-- Check structural compatibility
function Neuroglyph:isStructurallyCompatible(other)
    if not Neuroglyph.isNeuroglyph(other) then
        return false
    end
    return self.baseDim == other.baseDim and
           self.fiberDim == other.fiberDim and
           self.depth == other.depth
end

-- Compose two neuroglyphs (tensor product of structures)
function Neuroglyph:compose(other, config)
    config = config or {}

    if not Neuroglyph.isNeuroglyph(other) then
        error("Can only compose with another Neuroglyph")
    end

    return Neuroglyph.create({
        name = config.name or (self.name .. '⊗' .. other.name),
        glyphType = Neuroglyph.GLYPH_TYPES.COMPOSITE,
        baseDim = self.baseDim,  -- Preserve base dimension
        fiberDim = self.fiberDim + other.fiberDim,  -- Fiber direct sum
        gaugeGroup = self.gaugeGroup,
        depth = math.max(self.depth, other.depth) + 1,
        topology = {
            left = self.topology,
            right = other.topology
        },
        invariants = {
            count = (self.invariants.count or 1) * (other.invariants.count or 1),
            depth = math.max(self.depth, other.depth) + 1
        }
    })
end

-- Create a neural layer respecting this glyph's symmetry
function Neuroglyph:createLayer(layerType, config)
    local nnn = require 'nnn'
    config = config or {}

    layerType = layerType or 'linear'

    if layerType == 'linear' then
        return nnn.GULinear(self.fiberDim, config.outputDim or self.fiberDim, {
            applyTo = config.applyTo or 'fiber'
        })
    elseif layerType == 'relu' then
        return nnn.GUReLU({applyTo = config.applyTo or 'fiber'})
    elseif layerType == 'tanh' then
        return nnn.GUTanh({applyTo = config.applyTo or 'fiber'})
    elseif layerType == 'full' then
        local gu = require 'gu'
        return nnn.GULayer(self.fiberDim, {
            base_dim = self.baseDim,
            use_swerve = config.use_swerve ~= false,
            use_gauge = config.use_gauge ~= false,
            gauge_type = config.gauge_type or 'tilted'
        })
    else
        error("Unknown layer type: " .. layerType)
    end
end

-- Create a model from this glyph's specification
function Neuroglyph:createModel(config)
    local nnn = require 'nnn'
    config = config or {}

    local numLayers = config.numLayers or 3
    local hiddenDim = config.hiddenDim or self.fiberDim
    local activation = config.activation or 'tanh'
    local useResidual = config.useResidual or false

    local model = nnn.GUSequential()

    for i = 1, numLayers do
        -- Add linear layer
        local inDim = (i == 1) and self.fiberDim or hiddenDim
        local outDim = (i == numLayers) and self.fiberDim or hiddenDim

        model:add(nnn.GULinear(inDim, outDim, {applyTo = 'fiber'}))

        -- Add activation (except last layer)
        if i < numLayers then
            if activation == 'relu' then
                model:add(nnn.GUReLU({applyTo = 'fiber'}))
            elseif activation == 'tanh' then
                model:add(nnn.GUTanh({applyTo = 'fiber'}))
            elseif activation == 'sigmoid' then
                model:add(nnn.GUSigmoid({applyTo = 'fiber'}))
            end

            -- Optional dropout
            if config.dropout and config.dropout > 0 then
                model:add(nnn.GUDropout(config.dropout, {applyTo = 'fiber'}))
            end
        end
    end

    return model
end

-- Clone the neuroglyph
function Neuroglyph:clone()
    local clone = Neuroglyph.create({
        name = self.name .. '_clone',
        glyphType = self.glyphType,
        baseDim = self.baseDim,
        fiberDim = self.fiberDim,
        gaugeGroup = self.gaugeGroup,
        depth = self.depth
    })

    -- Deep copy topology
    local function deepCopy(t)
        if type(t) ~= 'table' then return t end
        local copy = {}
        for k, v in pairs(t) do
            copy[k] = deepCopy(v)
        end
        return copy
    end

    clone.topology = deepCopy(self.topology)
    clone.invariants = deepCopy(self.invariants)
    clone.layers = deepCopy(self.layers)

    return clone
end

-- String representation
function Neuroglyph:__tostring()
    local str = string.format(
        "Neuroglyph {\n" ..
        "  name: %s\n" ..
        "  type: %s\n" ..
        "  geometry: base=%dD, fiber=%dD\n" ..
        "  gauge: %s (%s)\n" ..
        "  depth: %d\n" ..
        "  signature: %s\n" ..
        "}",
        self.name,
        self.glyphType,
        self.baseDim,
        self.fiberDim,
        self.gaugeGroup,
        Neuroglyph.GAUGE_GROUPS[self.gaugeGroup] or 'Unknown',
        self.depth,
        self:signature()
    )
    return str
end

-- ============================================================================
-- Visualization
-- ============================================================================

function Neuroglyph:visualize()
    print("╔══════════════════════════════════════════════════════╗")
    print("║            GEONESTOR NEUROGLYPH                      ║")
    print("╠══════════════════════════════════════════════════════╣")
    print(string.format("║  Name:      %-40s ║", self.name))
    print(string.format("║  Type:      %-40s ║", self.glyphType))
    print(string.format("║  Signature: %-40s ║", self:signature()))
    print("╠══════════════════════════════════════════════════════╣")
    print("║  GEOMETRY                                            ║")
    print(string.format("║    Base Space:    %d-dimensional                       ║", self.baseDim))
    print(string.format("║    Fiber Space:   %d-dimensional                      ║", self.fiberDim))
    print(string.format("║    Chimeric Dim:  %d-dimensional                      ║", self.baseDim + self.fiberDim))
    print("╠══════════════════════════════════════════════════════╣")
    print("║  SYMMETRY                                            ║")
    print(string.format("║    Gauge Group:   %s (%s)                   ║",
        self.gaugeGroup,
        (Neuroglyph.GAUGE_GROUPS[self.gaugeGroup] or 'Unknown'):sub(1, 15)))
    print(string.format("║    Nesting Depth: %d                                    ║", self.depth))
    print("╠══════════════════════════════════════════════════════╣")
    print("║  PRIME SIGNATURE                                     ║")
    local sig = self:primeSignature()
    print(string.format("║    Base:  [%s]                                    ║",
        table.concat(sig.base, "·"):sub(1, 20)))
    print(string.format("║    Fiber: [%s]                                   ║",
        table.concat(sig.fiber, "·"):sub(1, 20)))
    print("╠══════════════════════════════════════════════════════╣")
    if self.invariants and next(self.invariants) then
        print("║  INVARIANTS                                          ║")
        if self.invariants.count then
            print(string.format("║    Count:     %d                                      ║", self.invariants.count))
        end
        if self.invariants.totalNorm then
            print(string.format("║    Norm:      %.4f                                 ║", self.invariants.totalNorm))
        end
    end
    print("╚══════════════════════════════════════════════════════╝")
end

-- ============================================================================
-- Module Registration
-- ============================================================================

-- Register with nnn module if available
local function register()
    local ok, nnn = pcall(require, 'nnn')
    if ok then
        nnn.Neuroglyph = Neuroglyph
        nnn.gu.Neuroglyph = Neuroglyph
    end
end

register()

return Neuroglyph
