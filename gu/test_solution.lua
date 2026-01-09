-- ============================================================================
-- Test Suite for Geometric Unity Solution Module
-- ============================================================================

require 'torch'

-- Load GU module
local gu_path = paths.dirname(paths.thisfile())
package.path = package.path .. ';' .. gu_path .. '/../?.lua'

require 'init'
require 'gu'

local Solution = require 'gu.Solution'

-- ============================================================================
-- Test Utilities
-- ============================================================================

local tests_passed = 0
local tests_failed = 0

local function test(name, fn)
    local ok, err = pcall(fn)
    if ok then
        print("[PASS] " .. name)
        tests_passed = tests_passed + 1
    else
        print("[FAIL] " .. name)
        print("       Error: " .. tostring(err))
        tests_failed = tests_failed + 1
    end
end

local function assert_eq(a, b, msg)
    if a ~= b then
        error(string.format("%s: expected %s, got %s", msg or "assertion failed", tostring(b), tostring(a)))
    end
end

local function assert_near(a, b, tol, msg)
    tol = tol or 1e-6
    if math.abs(a - b) > tol then
        error(string.format("%s: expected %s, got %s (tolerance %s)",
            msg or "assertion failed", tostring(b), tostring(a), tostring(tol)))
    end
end

-- ============================================================================
-- Tests
-- ============================================================================

print("")
print("═══════════════════════════════════════════════════════════════════")
print("           GEOMETRIC UNITY SOLUTION: TEST SUITE")
print("═══════════════════════════════════════════════════════════════════")
print("")

-- Test 1: Constants
test("Constants are correct", function()
    assert_eq(Solution.Constants.BASE_DIM, 4, "BASE_DIM")
    assert_eq(Solution.Constants.FIBER_DIM, 10, "FIBER_DIM")
    assert_eq(Solution.Constants.CHIMERIC_DIM, 14, "CHIMERIC_DIM")
    assert_eq(Solution.Constants.SPINOR_DIM, 128, "SPINOR_DIM")
    assert_eq(Solution.Constants.WEYL_SPINOR_DIM, 64, "WEYL_SPINOR_DIM")
end)

-- Test 2: Observerse creation
test("Observerse creation", function()
    local obs = Solution.Observerse()
    assert_eq(obs._type, 'GU_Observerse', "type")
    assert_eq(obs.base_dim, 4, "base_dim")
    assert_eq(obs.fiber_dim, 10, "fiber_dim")
    assert_eq(obs.total_dim, 14, "total_dim")
    assert(obs.metric ~= nil, "metric exists")
    assert(obs.connection ~= nil, "connection exists")
end)

-- Test 3: Observerse custom config
test("Observerse custom config", function()
    local obs = Solution.Observerse({base_dim = 3, fiber_dim = 6})
    assert_eq(obs.base_dim, 3, "custom base_dim")
    assert_eq(obs.fiber_dim, 6, "custom fiber_dim")
    assert_eq(obs.total_dim, 9, "custom total_dim")
end)

-- Test 4: Lagrangian creation
test("Lagrangian creation", function()
    local lag = gu.Lagrangian()
    assert_eq(lag.fiber_dim, 10, "fiber_dim")
    assert_eq(lag.base_dim, 4, "base_dim")
    assert_eq(lag.spinor_dim, 128, "spinor_dim")
end)

-- Test 5: Lagrangian forward pass
test("Lagrangian forward pass", function()
    local lag = gu.Lagrangian()
    local fields = {
        curvature = torch.randn(10, 10),
        spinor = torch.randn(128),
        metric = torch.eye(4)
    }
    local L, components = lag:computeLagrangian(fields)
    assert(type(L) == 'number', "L is number")
    assert(components.gravity ~= nil, "gravity component")
    assert(components.gauge ~= nil, "gauge component")
    assert(components.spinor ~= nil, "spinor component")
end)

-- Test 6: Field Equations creation
test("Field Equations creation", function()
    local fe = Solution.FieldEquations()
    assert_eq(fe.fiber_dim, 10, "fiber_dim")
    assert_eq(fe.base_dim, 4, "base_dim")
    assert(fe.shiab ~= nil, "shiab exists")
    assert(fe.swerve ~= nil, "swerve exists")
end)

-- Test 7: Field Equations residual
test("Field Equations residual", function()
    local fe = Solution.FieldEquations()
    local curvature = torch.randn(10, 10)
    local torsion = torch.zeros(10, 10)
    local matter = torch.zeros(10)

    local residual, norm = fe:residual(curvature, torsion, matter)
    assert(residual ~= nil, "residual exists")
    assert(type(norm) == 'number', "norm is number")
    assert(norm >= 0, "norm is non-negative")
end)

-- Test 8: Field Equations solve
test("Field Equations solve", function()
    local fe = Solution.FieldEquations()
    local initial = torch.randn(10, 10) * 0.1  -- Small initial curvature
    local torsion = torch.zeros(10, 10)
    local matter = torch.zeros(10)

    local result, info = fe:solve(initial, torsion, matter, {max_iter = 10})
    assert(result ~= nil, "result exists")
    assert(info.iterations > 0, "iterations > 0")
    assert(info.residual >= 0, "residual >= 0")
end)

-- Test 9: Endogenous Observer creation
test("Endogenous Observer creation", function()
    local obs = Solution.EndogenousObserver.create()
    assert_eq(obs._type, 'EndogenousObserver', "type")
    assert_eq(obs.base_dim, 4, "base_dim")
    assert_eq(obs.fiber_dim, 10, "fiber_dim")
    assert(obs.section ~= nil, "section exists")
end)

-- Test 10: Endogenous Observer embed
test("Endogenous Observer embed", function()
    local obs = Solution.EndogenousObserver.create()
    local base_point = torch.randn(4)
    local embedded = obs:embed(base_point)

    assert(embedded.base ~= nil, "base exists")
    assert(embedded.fiber ~= nil, "fiber exists")
    assert(embedded.total ~= nil, "total exists")
    assert_eq(embedded.base:size(1), 4, "base dim")
    assert_eq(embedded.total:size(1), 14, "total dim")
end)

-- Test 11: Endogenous Observer project
test("Endogenous Observer project", function()
    local obs = Solution.EndogenousObserver.create()
    local total = torch.randn(14)
    local projected = obs:project(total)

    assert_eq(projected:size(1), 4, "projected dim")
end)

-- Test 12: Endogenous Observer self-observe
test("Endogenous Observer self-observe", function()
    local obs = Solution.EndogenousObserver.create()
    local result = obs:selfObserve()

    assert(result.message ~= nil, "message exists")
    assert(result.section ~= nil, "section exists")
    assert(result.interpretation ~= nil, "interpretation exists")
end)

-- Test 13: Einstein-Yang-Mills creation
test("Einstein-Yang-Mills creation", function()
    local eym = Solution.EinsteinYangMills.create()
    assert_eq(eym._type, 'EinsteinYangMills', "type")
    assert_eq(eym.base_dim, 4, "base_dim")
    assert_eq(eym.fiber_dim, 10, "fiber_dim")
end)

-- Test 14: Einstein-Yang-Mills compute curvature
test("Einstein-Yang-Mills compute curvature", function()
    local eym = Solution.EinsteinYangMills.create()
    local curv = eym:computeCurvature()

    assert(curv.gauge ~= nil, "gauge exists")
    assert(curv.gravity ~= nil, "gravity exists")
    assert(curv.unified ~= nil, "unified exists")
end)

-- Test 15: Einstein-Yang-Mills unification theorem
test("Einstein-Yang-Mills unification theorem", function()
    local eym = Solution.EinsteinYangMills.create()
    local theorem = eym:unificationTheorem()

    assert(type(theorem) == 'string', "theorem is string")
    assert(theorem:find("UNIFICATION") ~= nil, "contains UNIFICATION")
end)

-- Test 16: Spinor Field Equations creation
test("Spinor Field Equations creation", function()
    local sfe = Solution.SpinorFieldEquations.create()
    assert_eq(sfe._type, 'SpinorFieldEquations', "type")
    assert_eq(sfe.spinor_dim, 128, "spinor_dim")
    assert_eq(sfe.weyl_dim, 64, "weyl_dim")
end)

-- Test 17: Spinor Field Equations Dirac operator
test("Spinor Field Equations Dirac operator", function()
    local sfe = Solution.SpinorFieldEquations.create()
    local spinor = torch.randn(128)
    local connection = {}
    for i = 1, 14 do
        connection[i] = 0.1
    end

    local D_psi = sfe:diracOperator(spinor, connection)
    assert_eq(D_psi:size(1), 128, "output dim")
end)

-- Test 18: Spinor decomposition to SM
test("Spinor decomposition to SM", function()
    local sfe = Solution.SpinorFieldEquations.create()
    local spinor = torch.randn(128)
    local decomp = sfe:decomposeToSM(spinor)

    assert(decomp.generation_1 ~= nil, "gen 1 exists")
    assert(decomp.generation_2 ~= nil, "gen 2 exists")
    assert(decomp.generation_3 ~= nil, "gen 3 exists")
    assert(decomp.exotic ~= nil, "exotic exists")
end)

-- Test 19: Anomaly cancellation
test("Anomaly cancellation", function()
    local ac = Solution.AnomalyCancellation

    assert_eq(ac.dimensions.base, 4, "base")
    assert_eq(ac.dimensions.fiber, 10, "fiber")
    assert_eq(ac.dimensions.total, 14, "total")
    assert_eq(ac.dimensions.spinor, 128, "spinor")

    local explanation = ac.explanation()
    assert(type(explanation) == 'string', "explanation is string")

    local check = ac.checkCancellation()
    assert(check.cancelled == true, "anomalies cancel")
end)

-- Test 20: Complete Solution creation
test("Complete Solution creation", function()
    local solver = Solution.CompleteSolution.create()
    assert_eq(solver._type, 'GU_CompleteSolution', "type")
    assert(solver.observerse ~= nil, "observerse exists")
    assert(solver.field_equations ~= nil, "field_equations exists")
    assert(solver.observer ~= nil, "observer exists")
    assert(solver.unification ~= nil, "unification exists")
    assert(solver.spinor_equations ~= nil, "spinor_equations exists")
    assert(solver.lagrangian ~= nil, "lagrangian exists")
end)

-- Test 21: Complete Solution solve
test("Complete Solution solve", function()
    local solver = Solution.CompleteSolution.create()
    local initial = {
        curvature = torch.randn(10, 10) * 0.1,
        torsion = torch.zeros(10, 10),
        matter = torch.zeros(10),
        spinor = torch.randn(128)
    }

    local result = solver:solve(initial, {max_iter = 5})

    assert(result ~= nil, "result exists")
    assert(result.curvature ~= nil, "curvature exists")
    assert(result.field_equations ~= nil, "field_equations exists")
    assert(result.einstein_tensor ~= nil, "einstein_tensor exists")
    assert(result.spacetime_metric ~= nil, "spacetime_metric exists")
    assert(result.yang_mills ~= nil, "yang_mills exists")
    assert(result.lagrangian ~= nil, "lagrangian exists")
    assert(result.observer ~= nil, "observer exists")
end)

-- Test 22: Quick solve
test("Quick solve", function()
    local result = Solution.quickSolve({max_iter = 3})

    assert(result ~= nil, "result exists")
    assert(result.curvature ~= nil, "curvature exists")
    assert(type(result.lagrangian) == 'number', "lagrangian is number")
end)

-- Test 23: Solution.create factory
test("Solution.create factory", function()
    local solver = Solution.create()
    assert_eq(solver._type, 'GU_CompleteSolution', "type")
end)

-- Test 24: Solution info display
test("Solution info display", function()
    -- Just check it doesn't error
    local old_print = print
    local output = {}
    print = function(s) table.insert(output, s) end
    Solution.info()
    print = old_print

    assert(#output > 0, "produced output")
end)

-- Test 25: Solution visualize
test("Solution visualize", function()
    -- Just check it doesn't error
    local old_print = print
    local output = {}
    print = function(s) table.insert(output, s) end
    Solution.visualize()
    print = old_print

    assert(#output > 0, "produced output")
end)

-- ============================================================================
-- Summary
-- ============================================================================

print("")
print("═══════════════════════════════════════════════════════════════════")
print(string.format("RESULTS: %d passed, %d failed", tests_passed, tests_failed))
print("═══════════════════════════════════════════════════════════════════")
print("")

if tests_failed > 0 then
    os.exit(1)
else
    print("All tests passed!")
    os.exit(0)
end
