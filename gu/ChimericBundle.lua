-- ============================================================================
-- ChimericBundle: The 14-Dimensional Bundle Structure of Geometric Unity
-- ============================================================================
-- The Chimeric Bundle is a direct sum of the vertical tangent bundle along
-- the fibers (10 dimensions) with the pullback horizontal bundle from the
-- base space (4 dimensions), yielding a 14-dimensional bundle.
--
-- Key properties:
-- - Has an a priori metric (metrics on base and fiber are perpendicular)
-- - Almost canonically isomorphic to tangent/cotangent bundle
-- - Missing data is exactly that of a connection
-- - Fermions are defined on this bundle (128-dim spinors from 2^7)
-- ============================================================================

local ChimericBundle = {}

-- Default dimensions
ChimericBundle.BASE_DIM = 4
ChimericBundle.FIBER_DIM = 10
ChimericBundle.TOTAL_DIM = 14
ChimericBundle.SPINOR_DIM = 128  -- 2^(14/2) = 2^7

-- ============================================================================
-- ChimericVector: A vector in the Chimeric Bundle
-- ============================================================================

local ChimericVector = {}

function ChimericVector.create(tensor)
    assert(torch.isTensor(tensor), "Input must be a tensor")
    
    local dim = tensor:dim()
    local last_dim = tensor:size(dim)
    assert(last_dim == ChimericBundle.TOTAL_DIM, 
        string.format("Last dimension must be %d, got %d", 
            ChimericBundle.TOTAL_DIM, last_dim))
    
    local vector = {
        data = tensor,
        _type = 'ChimericVector'
    }
    
    setmetatable(vector, {
        __tostring = function(t)
            return string.format("ChimericVector [%s]", 
                table.concat(t.data:size():totable(), "x"))
        end,
        __index = ChimericVector
    })
    
    return vector
end

-- Get base component (first 4 dimensions)
function ChimericVector:base()
    local dim = self.data:dim()
    return self.data:narrow(dim, 1, ChimericBundle.BASE_DIM)
end

-- Get fiber component (last 10 dimensions)
function ChimericVector:fiber()
    local dim = self.data:dim()
    return self.data:narrow(dim, ChimericBundle.BASE_DIM + 1, ChimericBundle.FIBER_DIM)
end

-- Convert to ObserverseTensor
function ChimericVector:toObserverse()
    local ObserverseTensor = require 'gu.ObserverseTensor'
    return ObserverseTensor.create(self:base():clone(), self:fiber():clone())
end

-- Clone
function ChimericVector:clone()
    return ChimericVector.create(self.data:clone())
end

-- Compute the Chimeric metric (base and fiber perpendicular)
function ChimericVector:metricNorm()
    local base_norm = self:base():norm()
    local fiber_norm = self:fiber():norm()
    return math.sqrt(base_norm^2 + fiber_norm^2)
end

ChimericBundle.Vector = ChimericVector

-- ============================================================================
-- ChimericSpinor: A spinor on the Chimeric Bundle
-- ============================================================================

local ChimericSpinor = {}

function ChimericSpinor.create(tensor)
    assert(torch.isTensor(tensor), "Input must be a tensor")
    
    local dim = tensor:dim()
    local last_dim = tensor:size(dim)
    assert(last_dim == ChimericBundle.SPINOR_DIM,
        string.format("Last dimension must be %d, got %d",
            ChimericBundle.SPINOR_DIM, last_dim))
    
    local spinor = {
        data = tensor,
        _type = 'ChimericSpinor'
    }
    
    setmetatable(spinor, {
        __tostring = function(t)
            return string.format("ChimericSpinor [%s]",
                table.concat(t.data:size():totable(), "x"))
        end,
        __index = ChimericSpinor
    })
    
    return spinor
end

-- Clone
function ChimericSpinor:clone()
    return ChimericSpinor.create(self.data:clone())
end

-- Compute norm
function ChimericSpinor:norm()
    return self.data:norm()
end

ChimericBundle.Spinor = ChimericSpinor

-- ============================================================================
-- Factory Functions
-- ============================================================================

-- Create a random Chimeric vector
function ChimericBundle.randomVector(batch_size)
    batch_size = batch_size or 1
    local tensor = torch.randn(batch_size, ChimericBundle.TOTAL_DIM)
    return ChimericVector.create(tensor)
end

-- Create a zero Chimeric vector
function ChimericBundle.zeroVector(batch_size)
    batch_size = batch_size or 1
    local tensor = torch.zeros(batch_size, ChimericBundle.TOTAL_DIM)
    return ChimericVector.create(tensor)
end

-- Create a random Chimeric spinor
function ChimericBundle.randomSpinor(batch_size)
    batch_size = batch_size or 1
    local tensor = torch.randn(batch_size, ChimericBundle.SPINOR_DIM)
    return ChimericSpinor.create(tensor)
end

-- Create a zero Chimeric spinor
function ChimericBundle.zeroSpinor(batch_size)
    batch_size = batch_size or 1
    local tensor = torch.zeros(batch_size, ChimericBundle.SPINOR_DIM)
    return ChimericSpinor.create(tensor)
end

-- Create from ObserverseTensor
function ChimericBundle.fromObserverse(observerse)
    local dim = observerse.base:dim()
    local chimeric = torch.cat({observerse.base, observerse.fiber}, dim)
    return ChimericVector.create(chimeric)
end

-- ============================================================================
-- Metric Operations
-- ============================================================================

-- Compute the Chimeric metric between two vectors
-- The metric is block-diagonal with base and fiber perpendicular
function ChimericBundle.metric(v1, v2)
    local base_inner = torch.dot(v1:base():view(-1), v2:base():view(-1))
    local fiber_inner = torch.dot(v1:fiber():view(-1), v2:fiber():view(-1))
    return base_inner + fiber_inner
end

-- Project onto base subspace
function ChimericBundle.projectBase(vector)
    local result = torch.zeros(vector.data:size())
    local dim = result:dim()
    result:narrow(dim, 1, ChimericBundle.BASE_DIM):copy(vector:base())
    return ChimericVector.create(result)
end

-- Project onto fiber subspace
function ChimericBundle.projectFiber(vector)
    local result = torch.zeros(vector.data:size())
    local dim = result:dim()
    result:narrow(dim, ChimericBundle.BASE_DIM + 1, ChimericBundle.FIBER_DIM):copy(vector:fiber())
    return ChimericVector.create(result)
end

return ChimericBundle
