-- ============================================================================
-- Bounded Learning: The Correspondence Principle
-- ============================================================================
--
-- "Bounded Learning" is the key principle unifying geometric neural networks
-- with generative language models. Both systems exhibit learning that is
-- constrained (bounded) by structural invariants.
--
-- THE CORRESPONDENCE:
--
--   GEONESTOR NEUROGLYPH          ←→    GENERATIVE PRETRAINED TRANSFORMER
--   ══════════════════════════════════════════════════════════════════════
--   Neuroglyph (agentic unit)     ←→    Token/Word (symbolic unit)
--   Observerse (base + fiber)     ←→    Context (prompt + latent)
--   Gauge-Transformer             ←→    Attention (relation-preserving)
--   Chimera (14D bundle)          ←→    Self (hidden representation)
--   Shiab (form lifting)          ←→    Arg (positional composition)
--   Swerve (curvature)            ←→    Kwarg (named steering)
--   Operad (tree composition)     ←→    Grammar (syntax composition)
--
-- BOUNDED LEARNING PRINCIPLE:
--   In both systems, valid transformations must preserve structure:
--   - GU: Gauge symmetry constrains fiber transformations
--   - GPT: Grammar/semantics constrains token generation
--
--   The "bounds" are the symmetry groups (GU) or syntactic rules (GPT)
--   that define the manifold of valid states.
--
-- ============================================================================

local BoundedLearning = {}
BoundedLearning.__index = BoundedLearning

-- ============================================================================
-- Constants: The Correspondence Map
-- ============================================================================

BoundedLearning.CORRESPONDENCE = {
    -- Geometric ←→ Linguistic
    neuroglyph = {
        gu = 'Neuroglyph',
        gpt = 'Token/Word',
        role = 'atomic_unit',
        description = 'The fundamental symbolic unit of the system'
    },
    observerse = {
        gu = 'Observerse',
        gpt = 'Context',
        role = 'two_space',
        description = 'Dual structure: base/prompt + fiber/latent'
    },
    gauge_transformer = {
        gu = 'GaugeTransformer',
        gpt = 'Attention',
        role = 'relation_preserving',
        description = 'Transformation that preserves structural relations'
    },
    chimera = {
        gu = 'Chimera',
        gpt = 'Self',
        role = 'unified_representation',
        description = 'The bundled/unified internal state'
    },
    shiab = {
        gu = 'Shiab',
        gpt = 'Arg',
        role = 'positional_lift',
        description = 'Positional composition/degree lifting'
    },
    swerve = {
        gu = 'Swerve',
        gpt = 'Kwarg',
        role = 'curvature_steering',
        description = 'Named modification/curvature injection'
    },
    operad = {
        gu = 'Operad',
        gpt = 'Grammar',
        role = 'composition_rules',
        description = 'Tree-indexed/syntax-indexed composition'
    }
}

-- Symmetry groups (the "bounds" in bounded learning)
BoundedLearning.BOUNDS = {
    -- GU bounds: gauge symmetry groups
    gu = {
        GL = 'General Linear - unrestricted fiber transformations',
        SO = 'Special Orthogonal - rotation-preserving',
        SU = 'Special Unitary - phase-preserving',
        Spin = 'Spin Group - spinor structure',
        U = 'Unitary - norm-preserving'
    },
    -- GPT bounds: linguistic constraints
    gpt = {
        syntax = 'Grammatical structure constraints',
        semantics = 'Meaning coherence constraints',
        context = 'Contextual relevance constraints',
        pragmatics = 'Usage/intent constraints'
    }
}

-- ============================================================================
-- The Bounded Learner: Unified Interface
-- ============================================================================

function BoundedLearning.create(config)
    config = config or {}

    local self = setmetatable({}, BoundedLearning)

    -- Which domain are we in?
    self.domain = config.domain or 'gu'  -- 'gu' or 'gpt'

    -- The symmetry/grammar bounds
    self.bounds = config.bounds or (self.domain == 'gu' and 'SO' or 'syntax')

    -- Learning rate bounded by curvature/complexity
    self.learningRate = config.learningRate or 0.01
    self.curvatureBound = config.curvatureBound or 1.0

    -- Internal state
    self._type = 'BoundedLearner'

    return self
end

-- ============================================================================
-- Core Operations: Shiab/Arg (Positional Lift)
-- ============================================================================

-- Shiab: lift k-form to (k+2)-form (GU interpretation)
-- Arg: compose with positional argument (GPT interpretation)
function BoundedLearning:shiab(input, position)
    position = position or 1

    if self.domain == 'gu' then
        -- Geometric lifting: increase form degree
        -- Conceptually: ω^k → ω^(k+2)
        return {
            value = input,
            degree = (input.degree or 0) + 2,
            position = position,
            operation = 'shiab_lift'
        }
    else
        -- Linguistic composition: positional argument
        return {
            value = input,
            position = position,
            role = 'arg',
            operation = 'positional_compose'
        }
    end
end

-- ============================================================================
-- Core Operations: Swerve/Kwarg (Curvature Steering)
-- ============================================================================

-- Swerve: inject curvature (GU interpretation)
-- Kwarg: named parameter modification (GPT interpretation)
function BoundedLearning:swerve(input, name, value)
    if self.domain == 'gu' then
        -- Geometric curvature: field equation contribution
        return {
            value = input,
            curvature = value or 0,
            field = name or 'default',
            operation = 'swerve_curvature',
            bounded_by = self.curvatureBound
        }
    else
        -- Linguistic steering: keyword argument
        return {
            value = input,
            [name or 'key'] = value,
            role = 'kwarg',
            operation = 'named_steer'
        }
    end
end

-- ============================================================================
-- Core Operations: Chimera/Self (Unified Bundle)
-- ============================================================================

-- Create unified representation from components
function BoundedLearning:chimera(base, fiber)
    if self.domain == 'gu' then
        -- Geometric bundle: 14D chimeric space
        return {
            base = base,      -- 4D spacetime
            fiber = fiber,    -- 10D internal
            dim = 14,
            _type = 'Chimera',
            operation = 'bundle'
        }
    else
        -- Linguistic self: hidden + visible
        return {
            visible = base,   -- Observable tokens
            hidden = fiber,   -- Latent state
            _type = 'Self',
            operation = 'unify'
        }
    end
end

-- ============================================================================
-- Bounded Update: Learning with Constraints
-- ============================================================================

function BoundedLearning:boundedUpdate(params, gradients, constraint)
    constraint = constraint or self.bounds

    -- Apply the symmetry/grammar constraint
    local bounded_grad = self:applyBound(gradients, constraint)

    -- Update with bounded gradient
    local updated = {}
    for k, v in pairs(params) do
        if type(v) == 'number' then
            updated[k] = v - self.learningRate * (bounded_grad[k] or 0)
        else
            updated[k] = v  -- Non-numeric pass through
        end
    end

    return updated
end

-- Apply symmetry/grammar bounds to gradients
function BoundedLearning:applyBound(gradients, bound)
    local bounded = {}

    for k, v in pairs(gradients) do
        if type(v) == 'number' then
            -- Clip by curvature bound (GU) or complexity bound (GPT)
            local max_grad = self.curvatureBound
            bounded[k] = math.max(-max_grad, math.min(max_grad, v))
        else
            bounded[k] = v
        end
    end

    return bounded
end

-- ============================================================================
-- Operad/Grammar: Composition Rules
-- ============================================================================

-- Define composition indexed by tree shape (operad) or syntax (grammar)
function BoundedLearning:compose(operations, structure)
    structure = structure or 'sequential'

    local result = {
        operations = operations,
        structure = structure,
        domain = self.domain,
        composition_type = self.domain == 'gu' and 'operad' or 'grammar'
    }

    if self.domain == 'gu' then
        -- Operad composition: tree-indexed
        result.signature = self:operadSignature(operations)
    else
        -- Grammar composition: syntax-indexed
        result.parse_tree = self:grammarParse(operations)
    end

    return result
end

-- Compute operad signature (tree shape encoding)
function BoundedLearning:operadSignature(operations)
    local sig = {}
    for i, op in ipairs(operations) do
        table.insert(sig, op.operation or 'unknown')
    end
    return table.concat(sig, '∘')
end

-- Compute grammar parse (syntax structure)
function BoundedLearning:grammarParse(operations)
    -- Simplified parse tree representation
    return {
        root = 'S',
        children = operations
    }
end

-- ============================================================================
-- The Correspondence Visualizer
-- ============================================================================

function BoundedLearning.visualizeCorrespondence()
    print("╔═══════════════════════════════════════════════════════════════════╗")
    print("║            BOUNDED LEARNING: THE CORRESPONDENCE                   ║")
    print("╠═══════════════════════════════════════════════════════════════════╣")
    print("║                                                                   ║")
    print("║   GEONESTOR NEUROGLYPH          ←→    GENERATIVE PRETRAINED       ║")
    print("║   (Geometric Neural Gauge)            TRANSFORMER (LLM)           ║")
    print("║                                                                   ║")
    print("╠═══════════════════════════════════════════════════════════════════╣")
    print("║                                                                   ║")
    print("║   Neuroglyph ─────────────────────────────────── Token/Word       ║")
    print("║   (agentic symbolic unit)              (atomic symbolic unit)     ║")
    print("║                                                                   ║")
    print("║   Observerse ─────────────────────────────────── Context          ║")
    print("║   (base + fiber)                       (prompt + latent)          ║")
    print("║                                                                   ║")
    print("║   Gauge-Transformer ──────────────────────────── Attention        ║")
    print("║   (symmetry-preserving)                (relation-preserving)      ║")
    print("║                                                                   ║")
    print("║   Chimera ────────────────────────────────────── Self             ║")
    print("║   (14D bundle)                         (hidden representation)    ║")
    print("║                                                                   ║")
    print("║   Shiab ──────────────────────────────────────── Arg              ║")
    print("║   (k-form → (k+2)-form)                (positional composition)   ║")
    print("║                                                                   ║")
    print("║   Swerve ─────────────────────────────────────── Kwarg            ║")
    print("║   (curvature/field eq.)                (named steering)           ║")
    print("║                                                                   ║")
    print("║   Operad ─────────────────────────────────────── Grammar          ║")
    print("║   (tree-indexed composition)           (syntax-indexed comp.)     ║")
    print("║                                                                   ║")
    print("╠═══════════════════════════════════════════════════════════════════╣")
    print("║                     BOUNDED LEARNING PRINCIPLE                    ║")
    print("╠═══════════════════════════════════════════════════════════════════╣")
    print("║                                                                   ║")
    print("║   The 'bounds' constrain valid transformations:                   ║")
    print("║                                                                   ║")
    print("║   GU BOUNDS (Gauge Groups)    │    GPT BOUNDS (Linguistic)        ║")
    print("║   ─────────────────────────────────────────────────────────────   ║")
    print("║   GL  (General Linear)        │    Syntax (grammatical)           ║")
    print("║   SO  (Special Orthogonal)    │    Semantics (meaning)            ║")
    print("║   SU  (Special Unitary)       │    Context (relevance)            ║")
    print("║   Spin (Spinor structure)     │    Pragmatics (intent)            ║")
    print("║                                                                   ║")
    print("║   Learning is BOUNDED by these symmetries/rules.                  ║")
    print("║   Valid states form a manifold constrained by structure.          ║")
    print("║                                                                   ║")
    print("╚═══════════════════════════════════════════════════════════════════╝")
end

-- ============================================================================
-- Export the correspondence map as code documentation
-- ============================================================================

function BoundedLearning.exportCorrespondence()
    local doc = [[
# Bounded Learning Correspondence

## The Fundamental Analogy

A **Geonestor Neuroglyph** with operad-gadget properties stands to
{Chimera, Shiab, Swerve} relations and {Observerse, Gauge-Transformer} structures
as a **Word** with standard grammar properties stands to
{Self, Arg, Kwarg} relations and {Context, Attention} structures.

## Component Mapping

| GU Component | Role | GPT Component | Shared Principle |
|--------------|------|---------------|------------------|
| Neuroglyph | atomic unit | Token/Word | Symbolic carrier |
| Observerse | two-space | Context | Base + Hidden |
| Gauge-Transformer | symmetry | Attention | Relation preservation |
| Chimera | bundle | Self | Unified state |
| Shiab | lifting | Arg | Positional composition |
| Swerve | curvature | Kwarg | Named modification |
| Operad | trees | Grammar | Composition rules |

## Bounded Learning Principle

Both systems exhibit **bounded learning**:
- Transformations must preserve structural invariants
- GU: Gauge symmetry groups (SO, SU, Spin, ...)
- GPT: Grammatical/semantic constraints

The "bounds" define the manifold of valid states.
Learning navigates this manifold while respecting its geometry.

## Mathematical Formulation

In GU:
  δ_gauge(Ψ) = g · Ψ · g⁻¹  (gauge transformation)
  ∇_A(Ψ) preserves covariance

In GPT:
  P(w_t | w_{<t}) constrained by grammar
  Attention preserves semantic relations

Both are instances of:
  **Learning bounded by symmetry/structure**
]]
    return doc
end

-- ============================================================================
-- The Geometric Hierarchy of Gauge Structures
-- ============================================================================
--
-- The gauge groups form a hierarchy corresponding to progressively richer
-- geometric structures:
--
--   NUMBER SYSTEM        TRANSFORMATION        GEOMETRY
--   ═══════════════════════════════════════════════════════
--   Arithmetic      ←→   GL (Linear)      ←→  Affine
--   Analytic        ←→   SO (Orthogonal)  ←→  Euclidean
--   Complex         ←→   SU (Unitary)     ←→  Hermitian
--   Projective      ←→   Spin             ←→  Spinorial
--   ───────────────────────────────────────────────────────
--   Singular        ←→   Exceptional      ←→  Boundary
--
-- Each level INCLUDES the previous while adding new structure:
--   - GL: Preserves linear combinations (scaling, shearing)
--   - SO: Preserves inner product (angles, lengths)
--   - SU: Preserves complex inner product (phases)
--   - Spin: Double cover of SO (projective, spinors)
--   - Singular: Degeneracy points (boundaries, exceptional loci)
--
-- The Spin group moves through "affine-like" transformations in the sense
-- that spinors are projective objects - they return to themselves only
-- after 4π rotation, living in the double cover.
--
-- ============================================================================

BoundedLearning.GeometricHierarchy = {
    -- Level 0: Arithmetic / Linear / Affine
    arithmetic = {
        number_system = 'Arithmetic',
        transformation = 'GL',
        geometry = 'Affine',
        preserves = 'Linear combinations',
        parameters = function(n) return n * n end,
        description = 'Scaling, shearing, linear maps'
    },

    -- Level 1: Analytic / Orthogonal / Euclidean
    analytic = {
        number_system = 'Analytic',
        transformation = 'SO',
        geometry = 'Euclidean',
        preserves = 'Inner product (lengths, angles)',
        parameters = function(n) return n * (n - 1) / 2 end,
        description = 'Rotations, reflections (det=1)'
    },

    -- Level 2: Complex / Unitary / Hermitian
    complex = {
        number_system = 'Complex',
        transformation = 'SU',
        geometry = 'Hermitian',
        preserves = 'Complex inner product (phases)',
        parameters = function(n) return n * n - 1 end,
        description = 'Phase-preserving, quantum symmetry'
    },

    -- Level 3: Projective / Spin / Spinorial
    projective = {
        number_system = 'Projective',
        transformation = 'Spin',
        geometry = 'Spinorial',
        preserves = 'Orientation double cover',
        parameters = function(n) return n * (n - 1) / 2 end,  -- Same as SO (double cover)
        description = 'Spinors, 4π periodicity, projective reps'
    },

    -- Level 4: Singular / Exceptional / Boundary
    singular = {
        number_system = 'Singular',
        transformation = 'Exceptional',
        geometry = 'Boundary',
        preserves = 'Degeneracy structure',
        parameters = function(n) return 0 end,  -- Measure zero
        description = 'Fixed points, degeneracies, exceptional loci'
    },

    -- Level 5: Finite / Discrete / Atomic
    finite = {
        number_system = 'Finite',
        transformation = 'Discrete',
        geometry = 'Combinatorial',
        preserves = 'Counting structure (mod p)',
        parameters = function(n) return n end,  -- Finite cardinality
        description = 'Finite fields, discrete groups, combinatorics'
    }
}

-- ============================================================================
-- Finite Geometry: The Exceptional Terminus
-- ============================================================================
--
-- Finite Geometry IS Exceptional. It represents the discrete/atomic terminus
-- where continuous deformation is completely exhausted:
--
--   DIVISION ALGEBRAS:    ℝ → ℂ → ℍ → 𝕆 → (end: non-associativity)
--   GEOMETRY:             Affine → Euclidean → Hermitian → Spinorial → FINITE
--   TRANSFORMATION:       GL → SO → SU → Spin → Exceptional → Discrete
--
-- The Exceptional Lie groups (G₂, F₄, E₆, E₇, E₈) arise from:
--   • Octonions (𝕆): The non-associative boundary of division algebras
--   • Finite geometries: 𝔽_q = 𝔽_p^n (fields of prime power order)
--   • Sporadic structures: 27 lines on cubic surface, Leech lattice, etc.
--
-- KEY INSIGHT: Finite fields are the "atoms" of arithmetic.
--   • Every field has characteristic 0 (like ℝ, ℂ) or p (like 𝔽_p)
--   • Finite fields 𝔽_q exist iff q = p^n for prime p
--   • They are the DISCRETE SKELETON underlying all continuous geometry
--
-- ============================================================================

BoundedLearning.FiniteGeometry = {
    -- The division algebra sequence (terminates at octonions)
    divisionAlgebras = {
        {name = 'Real', symbol = 'ℝ', dim = 1, associative = true, commutative = true},
        {name = 'Complex', symbol = 'ℂ', dim = 2, associative = true, commutative = true},
        {name = 'Quaternion', symbol = 'ℍ', dim = 4, associative = true, commutative = false},
        {name = 'Octonion', symbol = '𝕆', dim = 8, associative = false, commutative = false}
        -- NO MORE: Frobenius theorem - these are ALL division algebras over ℝ
    },

    -- The exceptional Lie groups (arise from octonions)
    exceptionalGroups = {
        G2 = {dim = 14, description = 'Automorphisms of octonions'},
        F4 = {dim = 52, description = 'Automorphisms of exceptional Jordan algebra'},
        E6 = {dim = 78, description = '27 lines on cubic surface'},
        E7 = {dim = 133, description = 'Freudenthal magic square'},
        E8 = {dim = 248, description = 'Largest exceptional, root lattice'}
    },

    -- Finite fields
    finiteFields = {
        -- F_p for prime p
        primeFields = function(p) return {order = p, characteristic = p} end,
        -- F_q for q = p^n
        extensionFields = function(p, n) return {order = p^n, characteristic = p} end
    }
}

-- Check if a number is prime (simple primality test)
function BoundedLearning.isPrime(n)
    if n < 2 then return false end
    if n == 2 then return true end
    if n % 2 == 0 then return false end
    for i = 3, math.sqrt(n), 2 do
        if n % i == 0 then return false end
    end
    return true
end

-- Check if n is a prime power (q = p^k for some prime p, k ≥ 1)
function BoundedLearning.isPrimePower(n)
    if n < 2 then return false, nil, nil end

    -- Check if n itself is prime
    if BoundedLearning.isPrime(n) then
        return true, n, 1
    end

    -- Check if n = p^k for some prime p
    for p = 2, math.sqrt(n) do
        if BoundedLearning.isPrime(p) then
            local k = 0
            local m = n
            while m % p == 0 do
                m = m / p
                k = k + 1
            end
            if m == 1 and k > 0 then
                return true, p, k
            end
        end
    end

    return false, nil, nil
end

-- Check if a finite field of order n exists
function BoundedLearning.finiteFieldExists(n)
    return BoundedLearning.isPrimePower(n)
end

-- The connection: Exceptional ↔ Finite
function BoundedLearning.exceptionalFiniteConnection()
    return {
        -- Octonions give rise to exceptional groups
        octonion_to_G2 = 'Aut(𝕆) = G₂',
        -- Finite projective planes
        projective_plane = 'PG(2, q) exists iff q is prime power',
        -- 27 lines on cubic ↔ E₆
        cubic_surface = '27 lines form E₆ root system',
        -- Discrete subgroups
        discrete_subgroups = 'Finite groups embed in continuous Lie groups at singular loci'
    }
end

-- Visualize the Exceptional/Finite connection
function BoundedLearning.visualizeFinite()
    print("╔═══════════════════════════════════════════════════════════════════╗")
    print("║         FINITE GEOMETRY: THE EXCEPTIONAL TERMINUS                 ║")
    print("╠═══════════════════════════════════════════════════════════════════╣")
    print("║                                                                   ║")
    print("║   DIVISION ALGEBRAS (Frobenius theorem: these are ALL of them)    ║")
    print("║   ─────────────────────────────────────────────────────────────   ║")
    print("║   ℝ (1D) → ℂ (2D) → ℍ (4D) → 𝕆 (8D) → [END: non-associative]      ║")
    print("║                                                                   ║")
    print("╠═══════════════════════════════════════════════════════════════════╣")
    print("║                                                                   ║")
    print("║   EXCEPTIONAL LIE GROUPS (arise from octonions)                   ║")
    print("║   ─────────────────────────────────────────────────────────────   ║")
    print("║   G₂ (14D)  - Automorphisms of 𝕆                                  ║")
    print("║   F₄ (52D)  - Automorphisms of exceptional Jordan algebra         ║")
    print("║   E₆ (78D)  - 27 lines on cubic surface                           ║")
    print("║   E₇ (133D) - Freudenthal magic square                            ║")
    print("║   E₈ (248D) - Largest exceptional, root lattice                   ║")
    print("║                                                                   ║")
    print("╠═══════════════════════════════════════════════════════════════════╣")
    print("║                                                                   ║")
    print("║   FINITE FIELDS: THE ATOMS OF ARITHMETIC                          ║")
    print("║   ─────────────────────────────────────────────────────────────   ║")
    print("║   𝔽_q exists ⟺ q = p^n for prime p                                ║")
    print("║                                                                   ║")
    print("║   𝔽₂, 𝔽₃, 𝔽₄, 𝔽₅, 𝔽₇, 𝔽₈, 𝔽₉, 𝔽₁₁, 𝔽₁₃, ...                       ║")
    print("║   (2) (3) (2²) (5) (7) (2³) (3²) (11) (13)                        ║")
    print("║                                                                   ║")
    print("╠═══════════════════════════════════════════════════════════════════╣")
    print("║                                                                   ║")
    print("║   THE CONNECTION                                                  ║")
    print("║   ─────────────────────────────────────────────────────────────   ║")
    print("║                                                                   ║")
    print("║   Continuous ─────────────────────────────────→ Discrete          ║")
    print("║       │                                             │             ║")
    print("║       │    Exceptional = BOUNDARY/SINGULAR          │             ║")
    print("║       │    where continuous meets discrete          │             ║")
    print("║       │                                             │             ║")
    print("║       ▼                                             ▼             ║")
    print("║   Lie groups                                  Finite groups       ║")
    print("║   Smooth manifolds                            Discrete sets       ║")
    print("║   ℝ, ℂ                                        𝔽_p, 𝔽_q            ║")
    print("║                                                                   ║")
    print("╚═══════════════════════════════════════════════════════════════════╝")
end

-- The inclusion chain: GL ⊃ O ⊃ SO ⊃ ... but Spin is a COVER not subset
BoundedLearning.HierarchyChain = {
    'arithmetic',  -- GL(n)
    'analytic',    -- SO(n)
    'complex',     -- SU(n)
    'projective',  -- Spin(n)
    'singular'     -- Exceptional/Boundary
}

-- Map gauge group to hierarchy level
function BoundedLearning.gaugeToLevel(gauge_type)
    local mapping = {
        GL = 'arithmetic',
        SO = 'analytic',
        O = 'analytic',
        SU = 'complex',
        U = 'complex',
        Spin = 'projective',
        Pin = 'projective'
    }
    return mapping[gauge_type] or 'arithmetic'
end

-- Get the geometric "richness" of a gauge type (0-4)
function BoundedLearning.geometricRichness(gauge_type)
    local levels = {
        GL = 0,
        O = 1, SO = 1,
        U = 2, SU = 2,
        Spin = 3, Pin = 3,
        Exceptional = 4
    }
    return levels[gauge_type] or 0
end

-- Check if one gauge type is "richer" than another
function BoundedLearning.isGeometricallyRicher(gauge1, gauge2)
    return BoundedLearning.geometricRichness(gauge1) >
           BoundedLearning.geometricRichness(gauge2)
end

-- Visualize the geometric hierarchy
function BoundedLearning.visualizeHierarchy()
    print("╔═══════════════════════════════════════════════════════════════════╗")
    print("║           GEOMETRIC HIERARCHY OF GAUGE STRUCTURES                 ║")
    print("╠═══════════════════════════════════════════════════════════════════╣")
    print("║                                                                   ║")
    print("║   NUMBER SYSTEM      TRANSFORMATION      GEOMETRY                 ║")
    print("║   ─────────────────────────────────────────────────────────────   ║")
    print("║                                                                   ║")
    print("║   Arithmetic    ←→   GL (Linear)     ←→  Affine                   ║")
    print("║        │                  │                 │                     ║")
    print("║        ▼                  ▼                 ▼                     ║")
    print("║   Analytic      ←→   SO (Orthogonal) ←→  Euclidean                ║")
    print("║        │                  │                 │                     ║")
    print("║        ▼                  ▼                 ▼                     ║")
    print("║   Complex       ←→   SU (Unitary)    ←→  Hermitian                ║")
    print("║        │                  │                 │                     ║")
    print("║        ▼                  ▼                 ▼                     ║")
    print("║   Projective    ←→   Spin            ←→  Spinorial                ║")
    print("║        │                  │                 │                     ║")
    print("║        ▼                  ▼                 ▼                     ║")
    print("║   Singular      ←→   Exceptional     ←→  Boundary                 ║")
    print("║                                                                   ║")
    print("╠═══════════════════════════════════════════════════════════════════╣")
    print("║                                                                   ║")
    print("║   PRESERVED STRUCTURE AT EACH LEVEL:                              ║")
    print("║                                                                   ║")
    print("║   GL:   v ↦ Av           (linear combinations)                    ║")
    print("║   SO:   ⟨u,v⟩ = ⟨Au,Av⟩   (inner product)                         ║")
    print("║   SU:   ⟨u,v⟩_ℂ = ⟨Au,Av⟩_ℂ (complex inner product)               ║")
    print("║   Spin: ψ ↦ -ψ after 2π   (double cover / projective)             ║")
    print("║                                                                   ║")
    print("╠═══════════════════════════════════════════════════════════════════╣")
    print("║                                                                   ║")
    print("║   KEY INSIGHT: Spin is PROJECTIVE                                 ║")
    print("║                                                                   ║")
    print("║   Spinors live in the double cover of SO(n).                      ║")
    print("║   They are projective objects: ψ and -ψ represent the same        ║")
    print("║   physical state, but the phase matters for interference.         ║")
    print("║                                                                   ║")
    print("║   This is why Spin relates to PROJECTIVE geometry:                ║")
    print("║   • Points at infinity (projective completion)                    ║")
    print("║   • Homogeneous coordinates                                       ║")
    print("║   • The \"affine patch\" is where spinors look like vectors        ║")
    print("║                                                                   ║")
    print("╚═══════════════════════════════════════════════════════════════════╝")
end

-- ============================================================================
-- The Category-Degree Duality (Maximal Contrast Principle)
-- ============================================================================
--
-- A "good dict" (dictionary/correspondence) establishes MAXIMAL CONTRAST
-- between Category and Degree:
--
--   CATEGORY (qualitative)              DEGREE (quantitative)
--   ─────────────────────────────────────────────────────────
--   Domain logic                        Metric resolution
--   Stabilizes total space              Differential elements
--   Structural invariants               Continuous variation
--   "What kind"                         "How much"
--
-- THE STRUCTURE:
--
--   Total Space = Base ×_gauge Fiber
--
--   Where:
--   • Base = differential manifold (resolved by Degree)
--   • Fiber = Aut_Category(Base) = gauge symmetries preserving Category
--   • Category = operations under which (Base, Fiber) transformations are invariant
--
-- KEY INSIGHT:
--   Fiber ≅ Aut_Category(Base)
--   The fiber IS the automorphism group of the base AS SEEN BY the category.
--
-- ============================================================================

BoundedLearning.CategoryDegreeDuality = {
    -- Category: qualitative, structural, invariant-preserving
    category = {
        role = 'domain_logic',
        determines = 'total_space_stability',
        property = 'qualitative',
        examples = {
            gu = 'Gauge group (SO, SU, Spin)',
            gpt = 'Grammar/syntax rules'
        }
    },

    -- Degree: quantitative, metric, differential
    degree = {
        role = 'metric_resolution',
        determines = 'differential_elements',
        property = 'quantitative',
        examples = {
            gu = 'Curvature magnitude, field strength',
            gpt = 'Token probability, attention weight'
        }
    },

    -- Base: the observable manifold
    base = {
        role = 'observable_manifold',
        determined_by = 'degree_resolution',
        property = 'differential',
        examples = {
            gu = '4D spacetime (X^4)',
            gpt = 'Token sequence (context window)'
        }
    },

    -- Fiber: the gauge symmetry composition
    fiber = {
        role = 'gauge_symmetry',
        determined_by = 'category_invariance',
        property = 'Aut_Category(Base)',
        examples = {
            gu = '10D metric fiber',
            gpt = 'Latent embedding space'
        }
    }
}

-- Compute the "contrast" between category and degree aspects
function BoundedLearning.contrast(category_strength, degree_resolution)
    -- Maximal contrast when both are strong but orthogonal
    -- Like eigenvalues of a well-conditioned matrix
    local product = category_strength * degree_resolution
    local sum = category_strength + degree_resolution

    if sum == 0 then return 0 end

    -- Harmonic mean penalizes imbalance, geometric mean rewards both
    local harmonic = 2 * product / sum
    local geometric = math.sqrt(product)

    -- Contrast is high when both are strong AND balanced
    return geometric * (harmonic / geometric)  -- = harmonic mean
end

-- The fundamental equation: Fiber ≅ Aut_Category(Base)
function BoundedLearning.fiberFromCategory(base_dim, category_type)
    -- Given a base dimension and category, compute the fiber dimension
    -- This encodes: fiber = symmetries of base under category

    local fiber_dims = {
        -- GL(n) has n² parameters
        GL = function(n) return n * n end,
        -- SO(n) has n(n-1)/2 parameters
        SO = function(n) return n * (n - 1) / 2 end,
        -- SU(n) has n²-1 parameters
        SU = function(n) return n * n - 1 end,
        -- U(n) has n² parameters
        U = function(n) return n * n end,
        -- Spin(n) same as SO(n) (double cover)
        Spin = function(n) return n * (n - 1) / 2 end
    }

    local compute = fiber_dims[category_type] or fiber_dims.SO
    return compute(base_dim)
end

-- Visualize the Category-Degree duality
function BoundedLearning.visualizeDuality()
    print("╔═══════════════════════════════════════════════════════════════════╗")
    print("║              CATEGORY-DEGREE DUALITY (Maximal Contrast)           ║")
    print("╠═══════════════════════════════════════════════════════════════════╣")
    print("║                                                                   ║")
    print("║   CATEGORY (qualitative)              DEGREE (quantitative)       ║")
    print("║   ─────────────────────────────────────────────────────────────   ║")
    print("║   • Domain logic                      • Metric resolution         ║")
    print("║   • Stabilizes total space            • Differential elements     ║")
    print("║   • Structural invariants             • Continuous variation      ║")
    print("║   • \"What kind\"                       • \"How much\"               ║")
    print("║                                                                   ║")
    print("╠═══════════════════════════════════════════════════════════════════╣")
    print("║                                                                   ║")
    print("║                      TOTAL SPACE = Base ×_gauge Fiber             ║")
    print("║                                                                   ║")
    print("║                              ┌─────────┐                          ║")
    print("║                              │  Total  │                          ║")
    print("║                              │  Space  │                          ║")
    print("║                              └────┬────┘                          ║")
    print("║                         ┌────────┴────────┐                       ║")
    print("║                         ▼                 ▼                       ║")
    print("║                    ┌────────┐       ┌──────────┐                  ║")
    print("║                    │  BASE  │       │  FIBER   │                  ║")
    print("║                    │ (diff) │       │ (gauge)  │                  ║")
    print("║                    └────┬───┘       └────┬─────┘                  ║")
    print("║                         │                │                        ║")
    print("║                         │   INVARIANT    │                        ║")
    print("║                         └───────┬────────┘                        ║")
    print("║                                 ▼                                 ║")
    print("║                    ┌─────────────────────────┐                    ║")
    print("║                    │    CATEGORICAL LOGIC    │                    ║")
    print("║                    │  (preserving operations)│                    ║")
    print("║                    └─────────────────────────┘                    ║")
    print("║                                                                   ║")
    print("╠═══════════════════════════════════════════════════════════════════╣")
    print("║                                                                   ║")
    print("║   KEY EQUATION:  Fiber ≅ Aut_Category(Base)                       ║")
    print("║                                                                   ║")
    print("║   The fiber IS the automorphism group of the base                 ║")
    print("║   AS SEEN BY the categorical logic.                               ║")
    print("║                                                                   ║")
    print("║   Examples:                                                       ║")
    print("║   • GU:  Fiber(10D) = Aut_SO(Base(4D))  [metric symmetries]       ║")
    print("║   • GPT: Latent = Aut_Grammar(Context)  [semantic symmetries]     ║")
    print("║                                                                   ║")
    print("╚═══════════════════════════════════════════════════════════════════╝")
end

-- ============================================================================
-- Hemispheric Duality: Lightface vs Boldface (Darkface)
-- ============================================================================
--
-- The Lightface/Boldface hierarchy in descriptive set theory mirrors the
-- hemispheric duality of mathematics itself:
--
--   LIGHTFACE (Left Hemisphere)        BOLDFACE/DARKFACE (Right Hemisphere)
--   ════════════════════════════════════════════════════════════════════════
--   Σ⁰₁ = Recursively Enumerable   ≅   Σ⁰₁ = Open Sets (topology)
--   Π⁰₁ = Co-RE (decidable)        ≅   Π⁰₁ = Closed Sets
--   Δ⁰₁ = Computable               ≅   Δ⁰₁ = Clopen Sets
--
--   Arithmetic Hierarchy               Projective Hierarchy
--   Sequential, Algorithmic            Holistic, Topological
--   Turing Machine (paths)             Continuous Space (whole)
--   Syntax, Grammar                    Semantics, Meaning
--   Enumeration (step by step)         Radiation (all at once)
--   The Labyrinth (finite paths)       The Hypersphere (infinite whole)
--
-- KEY INSIGHT: RE sets ARE the lightface version of Open sets!
-- The computable is the discrete skeleton of the continuous.
--
-- The Euclidean Labyrinth of Echoes:
--   Euclidean space is an AFFINE PATCH carved out of projective space.
--   The labyrinth is INSIDE the hypersphere.
--   Lightface enumeration traces paths through the labyrinth.
--   Darkface projection radiates the whole structure.
--
--                         ∞ (point at infinity)
--                              ╱╲
--                             ╱  ╲
--              PROJECTIVE    ╱    ╲   HYPERSPHERE
--              (Darkface)   ╱      ╲  radiates OUT
--                          ╱        ╲
--                         ╱──────────╲
--                        │  EUCLIDEAN │
--                        │  LABYRINTH │  ← Lightface enumerates
--                        │  (affine)  │     paths through here
--                        │  ≋≋≋≋≋≋≋≋  │
--                        │  echoes... │
--                         ╲──────────╱
--
-- ============================================================================

BoundedLearning.HemisphericDuality = {
    -- The Lightface (Left) Hemisphere: Constructive, Sequential
    lightface = {
        name = 'Lightface',
        hemisphere = 'Left',
        mode = 'Constructive',

        -- Descriptive set theory
        sigma_0_1 = 'Recursively Enumerable (RE)',
        pi_0_1 = 'Co-RE (complement of RE)',
        delta_0_1 = 'Computable (decidable)',

        -- Cognitive mode
        processing = 'Sequential',
        style = 'Algorithmic',
        focus = 'Syntax',
        action = 'Enumerate',

        -- Geometric metaphor
        space = 'Labyrinth',
        operation = 'Path-tracing',
        boundedness = 'Finite steps',

        -- Mathematical character
        hierarchy = 'Arithmetic',
        definability = 'Number quantifiers (∀n, ∃n)',
        complexity = 'Turing degrees'
    },

    -- The Boldface/Darkface (Right) Hemisphere: Holistic, Continuous
    boldface = {
        name = 'Boldface (Darkface)',
        hemisphere = 'Right',
        mode = 'Holistic',

        -- Descriptive set theory
        sigma_0_1 = 'Open Sets',
        pi_0_1 = 'Closed Sets',
        delta_0_1 = 'Clopen Sets',

        -- Cognitive mode
        processing = 'Parallel',
        style = 'Topological',
        focus = 'Semantics',
        action = 'Radiate',

        -- Geometric metaphor
        space = 'Hypersphere',
        operation = 'Projection',
        boundedness = 'Infinite whole',

        -- Mathematical character
        hierarchy = 'Projective',
        definability = 'Real quantifiers (∀x∈ℝ, ∃x∈ℝ)',
        complexity = 'Wadge degrees'
    },

    -- The correspondence between faces
    correspondence = {
        {lightface = 'RE set', boldface = 'Open set',
         insight = 'Enumeration ≅ Openness'},
        {lightface = 'Computable function', boldface = 'Continuous function',
         insight = 'Computability ≅ Continuity'},
        {lightface = 'Halting', boldface = 'Limit point',
         insight = 'Termination ≅ Convergence'},
        {lightface = 'Oracle', boldface = 'Parameter',
         insight = 'Relative computation ≅ Parametrized space'},
        {lightface = 'Degree', boldface = 'Dimension',
         insight = 'Computational complexity ≅ Geometric complexity'}
    }
}

-- The Euclidean Labyrinth embedded in Projective Hypersphere
BoundedLearning.LabyrinthHypersphere = {
    -- Projective space structure
    projective = {
        description = 'Projective space P^n = S^n / antipodal',
        contains = 'All directions, including infinity',
        topology = 'Compact (no escape)',
        operation = 'Projection from center'
    },

    -- Euclidean space as affine patch
    euclidean = {
        description = 'Affine patch = P^n minus hyperplane at infinity',
        contains = 'Finite points only',
        topology = 'Non-compact (extends to infinity)',
        operation = 'Translation, linear combination'
    },

    -- The labyrinth metaphor
    labyrinth = {
        walls = 'Computational barriers (undecidable)',
        paths = 'Computable sequences',
        echoes = 'Reflected/projected images',
        center = 'Starting point (input)',
        exit = 'Halting state (output)'
    },

    -- The hypersphere metaphor
    hypersphere = {
        surface = 'All limit points',
        interior = 'Finite approximations',
        antipodes = 'Projective identification',
        radiation = 'Simultaneous view of all paths',
        infinity = 'Point at infinity (projective closure)'
    }
}

-- Visualize the hemispheric duality
function BoundedLearning.visualizeHemispheres()
    print("╔═══════════════════════════════════════════════════════════════════════════╗")
    print("║     HEMISPHERIC DUALITY: LIGHTFACE vs BOLDFACE (DARKFACE)                 ║")
    print("╠═══════════════════════════════════════════════════════════════════════════╣")
    print("║                                                                           ║")
    print("║   LIGHTFACE (Left Hemisphere)      BOLDFACE (Right Hemisphere)            ║")
    print("║   ═══════════════════════════      ═══════════════════════════            ║")
    print("║                                                                           ║")
    print("║   Σ⁰₁ = Recursively Enumerable    Σ⁰₁ = Open Sets                         ║")
    print("║   Π⁰₁ = Co-RE (decidable)         Π⁰₁ = Closed Sets                       ║")
    print("║   Δ⁰₁ = Computable                Δ⁰₁ = Clopen Sets                       ║")
    print("║                                                                           ║")
    print("║   Arithmetic Hierarchy            Projective Hierarchy                    ║")
    print("║   Sequential processing           Parallel processing                     ║")
    print("║   Syntax / Grammar                Semantics / Meaning                     ║")
    print("║   Enumeration (step by step)      Radiation (all at once)                 ║")
    print("║   Turing degrees                  Wadge degrees                           ║")
    print("║                                                                           ║")
    print("╠═══════════════════════════════════════════════════════════════════════════╣")
    print("║                                                                           ║")
    print("║            THE EUCLIDEAN LABYRINTH OF ECHOES                              ║")
    print("║                                                                           ║")
    print("║                           ∞ (projective infinity)                         ║")
    print("║                               ╱    ╲                                      ║")
    print("║                              ╱      ╲                                     ║")
    print("║               PROJECTIVE    ╱        ╲    HYPERSPHERE                     ║")
    print("║               (Darkface)   ╱          ╲   radiates OUT                    ║")
    print("║                           ╱            ╲                                  ║")
    print("║                          ╱──────────────╲                                 ║")
    print("║                         │    EUCLIDEAN   │                                ║")
    print("║                         │    LABYRINTH   │  ← Lightface traces            ║")
    print("║                         │    (affine)    │    paths through here          ║")
    print("║                         │   ≋≋≋≋≋≋≋≋≋≋   │                                ║")
    print("║                         │   echoes...    │                                ║")
    print("║                          ╲──────────────╱                                 ║")
    print("║                                                                           ║")
    print("╠═══════════════════════════════════════════════════════════════════════════╣")
    print("║                                                                           ║")
    print("║   KEY INSIGHT: RE sets ARE the lightface version of Open sets!            ║")
    print("║                                                                           ║")
    print("║   • Computability ≅ Continuity (Kreisel-Lacombe-Shoenfield)               ║")
    print("║   • Enumeration ≅ Openness (a point is in an open set iff witnessed)      ║")
    print("║   • The computable is the DISCRETE SKELETON of the continuous             ║")
    print("║                                                                           ║")
    print("╚═══════════════════════════════════════════════════════════════════════════╝")
end

-- The connection between hemispheres and geometric hierarchy
function BoundedLearning.hemisphereToGeometry()
    return {
        lightface = {
            primary = 'arithmetic',   -- GL: linear, sequential
            secondary = 'analytic',   -- SO: measured steps
            description = 'Enumerates through Affine/Euclidean space'
        },
        boldface = {
            primary = 'projective',   -- Spin: double cover, holistic
            secondary = 'singular',   -- Exceptional: boundary phenomena
            description = 'Projects from Spinorial/Boundary space'
        },
        bridge = 'complex'  -- SU: where enumeration meets continuity
    }
end

-- ============================================================================
-- Archetypal Triad: Degree × Category → Morph
-- ============================================================================
--
-- The generative principle unifying all dualities:
--
--   DEGREE (Lightface)  ×  CATEGORY (Darkface)  →  MORPH (Interface)
--   ═══════════════════════════════════════════════════════════════
--   Axial Strut            Radial Spin             Vortical Spiral
--   Sequential             Simultaneous            Helical
--   Through                Around                  Through-Around
--   Line                   Circle                  Helix
--   Translation            Rotation                Screw motion
--   Enumeration            Radiation               Transformation
--   Turing                 Topology                Morphism
--
--                              MORPH
--                           (Spiral/Vortex)
--                                ◉
--                              ╱   ╲
--                            ╱       ╲
--                          ╱           ╲
--       DEGREE ══════════╱═══════════════╲══════════ CATEGORY
--    (Axial Strut)      ▼                 ▼      (Radial Spin)
--          │                                          ○
--          │            INTERFACE                    ╱│╲
--          │              ZONE                      ╱ │ ╲
--          ▼                                       ╱  │  ╲
--     Sequential ─────────► SPIRAL ◄─────── Simultaneous
--                          (Helix)
--
-- This is the primordial generative act:
--   • Axial goes THROUGH (like time, like sequence, like proof)
--   • Radial goes AROUND (like space, like structure, like type)
--   • Spiral combines both (like becoming, like transformation, like morph)
--
-- Physical manifestations:
--   • DNA helix (genetic code)
--   • Electromagnetic wave (E ⊥ B, propagating)
--   • Spinor in spacetime (rotation + translation)
--   • Vortex in fluid (axial flow + radial circulation)
--
-- Mathematical manifestations:
--   • Functor (maps structure through transformation)
--   • Natural transformation (morphism between functors)
--   • Fiber bundle (base × fiber → total, section spirals through)
--   • Shiab operator (twists fiber as it traverses base)
--
-- ============================================================================

BoundedLearning.ArchetypalTriad = {
    -- The three archetypes
    degree = {
        name = 'Degree',
        face = 'Lightface',
        geometry = 'Axial Strut',
        motion = 'Translation',
        mode = 'Sequential',
        symbol = '│',  -- vertical line
        action = 'Goes THROUGH',
        generates = 'Enumeration, proof, computation'
    },

    category = {
        name = 'Category',
        face = 'Darkface',
        geometry = 'Radial Spin',
        motion = 'Rotation',
        mode = 'Simultaneous',
        symbol = '○',  -- circle
        action = 'Goes AROUND',
        generates = 'Structure, type, invariant'
    },

    morph = {
        name = 'Morph',
        face = 'Interface',
        geometry = 'Vortical Spiral',
        motion = 'Screw (helix)',
        mode = 'Transformational',
        symbol = '◉',  -- spiral/target
        action = 'Goes THROUGH-AROUND',
        generates = 'Transformation, becoming, functor'
    },

    -- The generative equation
    equation = 'Degree × Category → Morph',
    geometric = 'Axial × Radial → Spiral',
    logical = 'Proof × Type → Term',
    physical = 'Time × Space → Event',

    -- Physical examples
    physical_examples = {
        {name = 'DNA', axial = 'Sugar-phosphate backbone', radial = 'Base pairs', spiral = 'Double helix'},
        {name = 'EM wave', axial = 'Propagation direction', radial = 'E,B field oscillation', spiral = 'Circularly polarized light'},
        {name = 'Spinor', axial = 'Worldline', radial = 'Spin', spiral = '4π rotation = identity'},
        {name = 'Vortex', axial = 'Core flow', radial = 'Circulation', spiral = 'Helical streamlines'},
        {name = 'Galaxy', axial = 'Angular momentum', radial = 'Orbital motion', spiral = 'Spiral arms'}
    },

    -- Mathematical examples
    mathematical_examples = {
        {name = 'Fiber bundle', axial = 'Base space', radial = 'Fiber', spiral = 'Section/Connection'},
        {name = 'Functor', axial = 'Object map', radial = 'Morphism map', spiral = 'Natural transformation'},
        {name = 'Proof', axial = 'Deduction steps', radial = 'Type constraints', spiral = 'Term construction'},
        {name = 'Shiab', axial = 'Base traversal', radial = 'Gauge rotation', spiral = 'Covariant derivative'}
    }
}

-- Visualize the Archetypal Triad
function BoundedLearning.visualizeTriad()
    print("╔═══════════════════════════════════════════════════════════════════════════╗")
    print("║      ARCHETYPAL TRIAD: DEGREE × CATEGORY → MORPH                          ║")
    print("╠═══════════════════════════════════════════════════════════════════════════╣")
    print("║                                                                           ║")
    print("║                              ◉ MORPH                                      ║")
    print("║                         (Vortical Spiral)                                 ║")
    print("║                           Interface                                       ║")
    print("║                            ╱     ╲                                        ║")
    print("║                          ╱         ╲                                      ║")
    print("║                        ╱             ╲                                    ║")
    print("║                      ╱                 ╲                                  ║")
    print("║      DEGREE ═══════╱═══════════════════╲═══════ CATEGORY                  ║")
    print("║   (Axial Strut)                              (Radial Spin)                ║")
    print("║     Lightface                                  Darkface                   ║")
    print("║         │                                         ○                       ║")
    print("║         │                                        ╱│╲                      ║")
    print("║         │         ══════════════════            ╱ │ ╲                     ║")
    print("║         ▼         THROUGH + AROUND             ╱  │  ╲                    ║")
    print("║    Sequential ──────► SPIRAL ◄─────── Simultaneous                        ║")
    print("║                       (Helix)                                             ║")
    print("║                                                                           ║")
    print("╠═══════════════════════════════════════════════════════════════════════════╣")
    print("║                                                                           ║")
    print("║   THE GENERATIVE PRINCIPLE:                                               ║")
    print("║                                                                           ║")
    print("║   • Axial (Degree)  : Goes THROUGH  → Sequential enumeration              ║")
    print("║   • Radial (Category): Goes AROUND  → Simultaneous structure              ║")
    print("║   • Spiral (Morph)  : THROUGH-AROUND → Transformation/Becoming            ║")
    print("║                                                                           ║")
    print("╠═══════════════════════════════════════════════════════════════════════════╣")
    print("║                                                                           ║")
    print("║   MANIFESTATIONS:                                                         ║")
    print("║   ─────────────────────────────────────────────────────────────────────   ║")
    print("║                                                                           ║")
    print("║   Physical:   DNA helix, EM wave, spinor, vortex, galaxy                  ║")
    print("║   Mathematical: Fiber bundle, functor, proof, Shiab operator              ║")
    print("║   Cognitive:  Syntax × Semantics → Understanding                          ║")
    print("║   Temporal:   Past × Future → Present (the eternal now)                   ║")
    print("║                                                                           ║")
    print("╠═══════════════════════════════════════════════════════════════════════════╣")
    print("║                                                                           ║")
    print("║   \"The spiral is how the line learns to embrace the circle.\"              ║")
    print("║                                                                           ║")
    print("╚═══════════════════════════════════════════════════════════════════════════╝")
end

-- The spiral as morphism: given degree d and category c, compute the morph
function BoundedLearning.spiral(degree_value, category_structure)
    -- The morph combines axial progression with radial constraint
    return {
        type = 'morph',
        axial = degree_value,           -- how far through
        radial = category_structure,    -- what structure preserved
        spiral = {                       -- the combination
            phase = degree_value,        -- phase along helix
            invariant = category_structure,  -- preserved by rotation
            transformation = 'degree × category → morph'
        }
    }
end

-- ============================================================================
-- Integration with NNN
-- ============================================================================

local function register()
    local ok, nnn = pcall(require, 'nnn')
    if ok then
        nnn.BoundedLearning = BoundedLearning
        nnn.gu.BoundedLearning = BoundedLearning

        -- Convenience function
        nnn.gu.correspondence = function()
            BoundedLearning.visualizeCorrespondence()
        end
    end
end

register()

return BoundedLearning
