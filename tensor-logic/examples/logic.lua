--[[
LOGIC PROGRAMMING IN TENSOR LOGIC

This demonstrates how classical logic programming (like Datalog/Prolog) maps
directly to tensor operations through Einstein summation.

THE KEY INSIGHT:
A logical rule like:
   Ancestor(x,z) ← Parent(x,y), Ancestor(y,z)

Is mathematically equivalent to:
   Ancestor[x,z] = Σ_y Parent[x,y] · Ancestor[y,z]

This is because:
- In logic, the rule says "x is an ancestor of z if there exists some y
  such that x is the parent of y AND y is an ancestor of z"
- The comma (,) in logic is conjunction (AND)
- The existential "there exists y" is captured by the summation Σ_y
- In Boolean tensors, multiplication is AND, and summation followed by
  threshold (>0) gives us OR/EXISTS
--]]

local core = require 'tensor-logic.core'

local M = {}

--[[
Example: Family relationships

Objects: Alice(1), Bob(2), Charlie(3), Diana(4)

Facts:
  Parent(Alice, Bob)    - Alice is Bob's parent
  Parent(Bob, Charlie)  - Bob is Charlie's parent
  Parent(Bob, Diana)    - Bob is Diana's parent

Rules:
  Ancestor(x,y) ← Parent(x,y)                    [Parents are ancestors]
  Ancestor(x,z) ← Ancestor(x,y), Parent(y,z)    [Transitive closure]

Query: Find all ancestor relationships
--]]
function M.runLogicProgramExample()
    local steps = {}
    
    -- Define the Parent relation as a Boolean tensor
    -- Parent[x,y] = 1 if x is a parent of y, 0 otherwise
    --
    -- Matrix representation (rows = parent, cols = child):
    --          Alice  Bob  Charlie  Diana
    -- Alice      0     1      0       0
    -- Bob        0     0      1       1
    -- Charlie    0     0      0       0
    -- Diana      0     0      0       0
    local Parent = core.fromMatrix('Parent', {'x', 'y'}, {
        {0, 1, 0, 0}, -- Alice is parent of Bob
        {0, 0, 1, 1}, -- Bob is parent of Charlie and Diana
        {0, 0, 0, 0}, -- Charlie has no children
        {0, 0, 0, 0}, -- Diana has no children
    })
    
    table.insert(steps, {
        name = 'Parent Relation',
        explanation = 'The Parent relation encoded as a Boolean tensor.\n' ..
                     'Parent[x,y] = 1 means "x is the parent of y".\n\n' ..
                     'Parent[Alice,Bob] = 1\n' ..
                     'Parent[Bob,Charlie] = 1\n' ..
                     'Parent[Bob,Diana] = 1',
        tensor = Parent,
        tensorString = core.tensorToString(Parent, 0)
    })
    
    -- Rule 1: Ancestor(x,y) ← Parent(x,y)
    -- "Parents are ancestors"
    -- This is just copying Parent to Ancestor
    local Ancestor = core.clone(Parent)
    Ancestor.name = 'Ancestor'
    
    table.insert(steps, {
        name = 'Rule 1: Ancestor ← Parent',
        explanation = 'First rule: Every parent is an ancestor.\n\n' ..
                     'Tensor Logic:  Ancestor[x,y] = Parent[x,y]\n\n' ..
                     'This is a base case - direct copy of the Parent tensor.',
        tensor = core.clone(Ancestor),
        tensorString = core.tensorToString(Ancestor, 0)
    })
    
    -- Rule 2: Ancestor(x,z) ← Ancestor(x,y), Parent(y,z)
    -- "If x is an ancestor of y, and y is parent of z, then x is ancestor of z"
    --
    -- In tensor notation:
    --   Ancestor[x,z] += Σ_y Ancestor[x,y] · Parent[y,z]
    --
    -- This is matrix multiplication! Einstein summation notation: "xy,yz->xz"
    
    -- Iterate until fixpoint (no new ancestors found)
    local changed = true
    local iteration = 0
    
    while changed and iteration < 10 do
        iteration = iteration + 1
        
        -- Compute new ancestors via the transitive rule
        -- This einsum computes: NewAncestors[x,z] = Σ_y Ancestor[x,y] · Parent[y,z]
        local newAncestors = core.einsum('xy,yz->xz', Ancestor, Parent)
        
        -- Apply threshold: any positive value becomes 1 (Boolean OR semantics)
        local thresholded = core.threshold(newAncestors)
        
        -- Combine with existing ancestors (logical OR)
        local combined = core.add(Ancestor, thresholded)
        local combinedThresholded = core.threshold(combined)
        
        -- Check if anything changed
        changed = false
        for i = 1, #Ancestor.data do
            if combinedThresholded.data[i] ~= Ancestor.data[i] then
                changed = true
                break
            end
        end
        
        Ancestor = combinedThresholded
        Ancestor.name = 'Ancestor'
        
        if changed then
            table.insert(steps, {
                name = string.format('Rule 2 (iteration %d)', iteration),
                explanation = 'Transitive rule: Ancestor[x,z] ← Ancestor[x,y], Parent[y,z]\n\n' ..
                             'Tensor Logic:  Ancestor[x,z] = threshold(Σ_y Ancestor[x,y] · Parent[y,z])\n\n' ..
                             'Einstein summation "xy,yz->xz" computes:\n' ..
                             '- For each pair (x,z), sum over all intermediate y\n' ..
                             '- Where both Ancestor[x,y]=1 AND Parent[y,z]=1\n' ..
                             '- This is JOIN on y, PROJECT onto (x,z)\n\n' ..
                             'New ancestors discovered in this iteration!',
                tensor = core.clone(Ancestor),
                tensorString = core.tensorToString(Ancestor, 0)
            })
        end
    end
    
    table.insert(steps, {
        name = 'Final: Ancestor Relation (Fixpoint)',
        explanation = string.format('Fixpoint reached after %d iteration(s).\n\n' ..
                     'Final Ancestor relation:\n' ..
                     '- Ancestor(Alice, Bob)     ✓  (direct parent)\n' ..
                     '- Ancestor(Alice, Charlie) ✓  (grandparent via Bob)\n' ..
                     '- Ancestor(Alice, Diana)   ✓  (grandparent via Bob)\n' ..
                     '- Ancestor(Bob, Charlie)   ✓  (direct parent)\n' ..
                     '- Ancestor(Bob, Diana)     ✓  (direct parent)\n\n' ..
                     'This is the DEDUCTIVE CLOSURE - all facts that can be derived from\n' ..
                     'the rules and base facts through logical inference.', iteration),
        tensor = Ancestor,
        tensorString = core.tensorToString(Ancestor, 0)
    })
    
    return {
        title = 'Logic Programming: Family Relationships',
        description = 'This example shows how Datalog-style logic programming maps to tensor operations.\n\n' ..
                     'Datalog rule:\n' ..
                     '  Ancestor(x, z) ← Ancestor(x, y), Parent(y, z)\n\n' ..
                     'Tensor Logic equation:\n' ..
                     '  Ancestor[x, z] = Ancestor[x, y] Parent[y, z]\n\n' ..
                     'This equation:\n' ..
                     '1. Joins Ancestor[x, y] and Parent[y, z] on the common index y\n' ..
                     '2. Sums over y (implicit in einsum)\n' ..
                     '3. Projects onto [x, z]',
        steps = steps
    }
end

return M
