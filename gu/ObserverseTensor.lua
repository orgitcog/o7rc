-- ============================================================================
-- ObserverseTensor: The Two-Space Structure of Geometric Unity
-- ============================================================================
-- The Observerse represents a fundamental departure from standard spacetime
-- models by replacing a single 4-dimensional spacetime manifold with a coupled
-- two-space structure. This is analogous to a stadium with a playing field
-- (base space) and stands (fiber space).
-- ============================================================================

local ObserverseTensor = {}

-- ============================================================================
-- Constructor
-- ============================================================================

function ObserverseTensor.create(base_space_tensor, fiber_space_tensor)
    -- Validate inputs
    assert(torch.isTensor(base_space_tensor), 
        "base_space_tensor must be a torch.Tensor")
    assert(torch.isTensor(fiber_space_tensor), 
        "fiber_space_tensor must be a torch.Tensor")
    
    -- Create the observerse structure
    local observerse = {
        base = base_space_tensor,
        fiber = fiber_space_tensor,
        _type = 'ObserverseTensor'
    }
    
    -- Set metatable for custom behavior
    setmetatable(observerse, {
        __tostring = function(t)
            return string.format(
                "ObserverseTensor {\n" ..
                "  base:  %s [%s]\n" ..
                "  fiber: %s [%s]\n" ..
                "}",
                table.concat(t.base:size():totable(), "x"),
                t.base:type(),
                table.concat(t.fiber:size():totable(), "x"),
                t.fiber:type()
            )
        end,
        
        __index = ObserverseTensor
    })
    
    return observerse
end

-- ============================================================================
-- Instance Methods
-- ============================================================================

-- Clone the ObserverseTensor
function ObserverseTensor:clone()
    return ObserverseTensor.create(
        self.base:clone(),
        self.fiber:clone()
    )
end

-- Get the total dimension (base + fiber)
function ObserverseTensor:totalDim()
    return self.base:size(self.base:dim()) + self.fiber:size(self.fiber:dim())
end

-- Convert to a single Chimeric Bundle tensor
function ObserverseTensor:toChimeric()
    local dim = self.base:dim()
    return torch.cat({self.base, self.fiber}, dim)
end

-- Apply a function to both components
function ObserverseTensor:map(func)
    return ObserverseTensor.create(
        func(self.base),
        func(self.fiber)
    )
end

-- Apply different functions to base and fiber
function ObserverseTensor:mapComponents(base_func, fiber_func)
    return ObserverseTensor.create(
        base_func(self.base),
        fiber_func(self.fiber)
    )
end

-- Type conversion
function ObserverseTensor:type(type_str)
    return ObserverseTensor.create(
        self.base:type(type_str),
        self.fiber:type(type_str)
    )
end

-- Move to CUDA
function ObserverseTensor:cuda()
    return self:type('torch.CudaTensor')
end

-- Move to CPU
function ObserverseTensor:float()
    return self:type('torch.FloatTensor')
end

function ObserverseTensor:double()
    return self:type('torch.DoubleTensor')
end

-- Zero out the tensors
function ObserverseTensor:zero()
    self.base:zero()
    self.fiber:zero()
    return self
end

-- Fill with a value
function ObserverseTensor:fill(value)
    self.base:fill(value)
    self.fiber:fill(value)
    return self
end

-- Add another ObserverseTensor
function ObserverseTensor:add(other, scale)
    scale = scale or 1
    if type(other) == 'number' then
        self.base:add(other)
        self.fiber:add(other)
    else
        self.base:add(scale, other.base)
        self.fiber:add(scale, other.fiber)
    end
    return self
end

-- Multiply by a scalar
function ObserverseTensor:mul(scalar)
    self.base:mul(scalar)
    self.fiber:mul(scalar)
    return self
end

-- Compute the norm
function ObserverseTensor:norm(p)
    p = p or 2
    local base_norm = self.base:norm(p)
    local fiber_norm = self.fiber:norm(p)
    return math.sqrt(base_norm^2 + fiber_norm^2)
end

-- ============================================================================
-- Static Methods
-- ============================================================================

-- Create from a single Chimeric Bundle tensor
function ObserverseTensor.fromChimeric(chimeric, base_dim)
    base_dim = base_dim or 4
    local dim = chimeric:dim()
    local total_dim = chimeric:size(dim)
    local fiber_dim = total_dim - base_dim
    
    local base = chimeric:narrow(dim, 1, base_dim)
    local fiber = chimeric:narrow(dim, base_dim + 1, fiber_dim)
    
    return ObserverseTensor.create(base:clone(), fiber:clone())
end

-- Create a zero ObserverseTensor
function ObserverseTensor.zeros(base_size, fiber_size)
    return ObserverseTensor.create(
        torch.zeros(base_size),
        torch.zeros(fiber_size)
    )
end

-- Create a random ObserverseTensor
function ObserverseTensor.randn(base_size, fiber_size)
    return ObserverseTensor.create(
        torch.randn(base_size),
        torch.randn(fiber_size)
    )
end

-- Check if something is an ObserverseTensor
function ObserverseTensor.isObserverseTensor(obj)
    return type(obj) == 'table' and obj._type == 'ObserverseTensor'
end

return ObserverseTensor
