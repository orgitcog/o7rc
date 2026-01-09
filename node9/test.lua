#!/usr/bin/env th
--[[
    Node9 Integration Test Suite
    
    Tests for the Node9 OS framework integration with Torch7u
]]--

local test_count = 0
local passed_count = 0
local failed_tests = {}

local function test(name, fn)
    test_count = test_count + 1
    io.write(string.format("[%d] Testing %s... ", test_count, name))
    
    local success, err = pcall(fn)
    if success then
        passed_count = passed_count + 1
        print("✓ PASS")
        return true
    else
        print("✗ FAIL")
        print("  Error: " .. tostring(err))
        table.insert(failed_tests, {name = name, error = err})
        return false
    end
end

local function assert_equal(a, b, msg)
    if a ~= b then
        error(msg or string.format("Expected %s, got %s", tostring(b), tostring(a)))
    end
end

local function assert_true(val, msg)
    if not val then
        error(msg or "Expected true, got false")
    end
end

local function assert_not_nil(val, msg)
    if val == nil then
        error(msg or "Expected non-nil value")
    end
end

print("=" .. string.rep("=", 70))
print("  Node9 Integration Test Suite")
print("=" .. string.rep("=", 70))
print("")

-- Load node9
local node9 = require('node9')

-- Test 1: Module Loading
test("Node9 module loads", function()
    assert_not_nil(node9, "node9 module should not be nil")
    assert_not_nil(node9.init, "node9.init function should exist")
end)

-- Test 2: Initialization
test("Node9 initialization", function()
    assert_true(node9._initialized, "node9 should be initialized")
    assert_not_nil(node9.config, "node9.config should exist")
end)

-- Test 3: Configuration
test("Configuration management", function()
    assert_not_nil(node9.config.root, "config.root should be set")
    assert_equal(node9.config.enable_kernel, true, "kernel should be enabled by default")
end)

-- Test 4: Penrose Lua Loading
test("Penrose Lua (pl) loading", function()
    node9.pl.load()
    assert_not_nil(node9.pl.List, "List should be loaded")
    assert_not_nil(node9.pl.Set, "Set should be loaded")
    assert_not_nil(node9.pl.Map, "Map should be loaded")
    assert_not_nil(node9.pl.stringx, "stringx should be loaded")
    assert_not_nil(node9.pl.tablex, "tablex should be loaded")
end)

-- Test 5: List Operations
test("List operations", function()
    node9.pl.load()
    local List = node9.pl.List
    local list = List({1, 2, 3, 4, 5})
    
    -- Test map
    local doubled = list:map(function(x) return x * 2 end)
    assert_equal(doubled[1], 2, "First element should be doubled")
    assert_equal(doubled[5], 10, "Last element should be doubled")
    
    -- Test filter
    local evens = list:filter(function(x) return x % 2 == 0 end)
    assert_equal(#evens, 2, "Should have 2 even numbers")
    
    -- Test reduce
    local sum = list:reduce('+')
    assert_equal(sum, 15, "Sum should be 15")
end)

-- Test 6: Set Operations
test("Set operations", function()
    local Set = node9.pl.Set
    local set1 = Set({1, 2, 3, 4})
    local set2 = Set({3, 4, 5, 6})
    
    -- Test union
    local union = set1 + set2
    assert_true(union[1], "Union should contain 1")
    assert_true(union[6], "Union should contain 6")
    
    -- Test intersection
    local intersection = set1 * set2
    assert_true(intersection[3], "Intersection should contain 3")
    assert_true(intersection[4], "Intersection should contain 4")
    assert_true(not intersection[1], "Intersection should not contain 1")
end)

-- Test 7: String Utilities
test("String utilities", function()
    local stringx = node9.pl.stringx
    
    -- Test strip
    local stripped = stringx.strip("  hello  ")
    assert_equal(stripped, "hello", "Should strip whitespace")
    
    -- Test split
    local parts = stringx.split("a,b,c", ",")
    assert_equal(#parts, 3, "Should split into 3 parts")
    assert_equal(parts[1], "a", "First part should be 'a'")
    
    -- Test startswith
    assert_true(stringx.startswith("hello", "hel"), "Should start with 'hel'")
    
    -- Test endswith
    assert_true(stringx.endswith("world", "rld"), "Should end with 'rld'")
end)

-- Test 8: Table Utilities
test("Table utilities", function()
    local tablex = node9.pl.tablex
    
    -- Test merge
    local t1 = {a = 1, b = 2}
    local t2 = {c = 3, d = 4}
    local merged = tablex.merge(t1, t2, true)
    assert_equal(merged.a, 1, "Should have a=1")
    assert_equal(merged.c, 3, "Should have c=3")
    
    -- Test keys
    local keys = tablex.keys({a = 1, b = 2, c = 3})
    assert_equal(#keys, 3, "Should have 3 keys")
    
    -- Test deepcopy
    local nested = {a = 1, b = {c = 2}}
    local copy = tablex.deepcopy(nested)
    copy.b.c = 999
    assert_equal(nested.b.c, 2, "Original should not be modified")
    assert_equal(copy.b.c, 999, "Copy should be modified")
end)

-- Test 9: Path Utilities
test("Path utilities", function()
    local path = node9.pl.path
    
    -- Test join
    local joined = path.join("home", "user", "file.txt")
    assert_not_nil(joined, "Path should be joined")
    
    -- Test basename
    local base = path.basename("/home/user/file.txt")
    assert_equal(base, "file.txt", "Basename should be 'file.txt'")
    
    -- Test extension
    local ext = path.extension("file.txt")
    assert_equal(ext, ".txt", "Extension should be '.txt'")
end)

-- Test 10: Map Data Structure
test("Map data structure", function()
    local Map = node9.pl.Map
    local map = Map({a = 1, b = 2})
    
    assert_equal(map.a, 1, "Map should have a=1")
    assert_equal(map.b, 2, "Map should have b=2")
    
    map.c = 3
    assert_equal(map.c, 3, "Map should allow setting new keys")
end)

-- Test 11: OrderedMap
test("OrderedMap maintains order", function()
    local OrderedMap = node9.pl.OrderedMap
    local omap = OrderedMap()
    
    omap:set("first", 1)
    omap:set("second", 2)
    omap:set("third", 3)
    
    local keys = {}
    for key, value in omap:iter() do
        table.insert(keys, key)
    end
    
    assert_equal(keys[1], "first", "First key should be 'first'")
    assert_equal(keys[2], "second", "Second key should be 'second'")
    assert_equal(keys[3], "third", "Third key should be 'third'")
end)

-- Test 12: Pretty Printing
test("Pretty printing", function()
    local pretty = node9.pl.pretty
    local data = {a = 1, b = {c = 2, d = 3}}
    local output = pretty.write(data)
    assert_not_nil(output, "Pretty print should return output")
end)

-- Test 13: Utils Module
test("Utils module", function()
    local utils = node9.pl.utils
    
    assert_true(utils.is_type({}, 'table'), "Should detect table type")
    assert_true(utils.is_type("hello", 'string'), "Should detect string type")
    assert_true(utils.is_type(42, 'number'), "Should detect number type")
end)

-- Test 14: Module Loading Functions
test("Module loading functions", function()
    -- Test that load functions exist
    assert_not_nil(node9.pl.load, "pl.load should exist")
    assert_not_nil(node9.kernel.load, "kernel.load should exist")
    assert_not_nil(node9.scheduler.load, "scheduler.load should exist")
    assert_not_nil(node9.fs.load, "fs.load should exist")
    assert_not_nil(node9.appl.load, "appl.load should exist")
end)

-- Test 15: Application Framework
test("Application framework initialization", function()
    node9.appl.load()
    assert_true(node9.appl.available, "Application framework should be available")
    assert_not_nil(node9.appl._module_dir, "Module directory should be set")
end)

-- Test 16: Filesystem Module Loading
test("Filesystem module loading", function()
    node9.fs.load()
    -- Note: fs modules may not load if dependencies are missing
    -- This test just verifies the load function works
    assert_not_nil(node9.fs, "Filesystem module should exist")
end)

-- Test 17: Torch Integration
test("Torch integration helpers", function()
    assert_not_nil(node9.torch, "torch integration should exist")
    assert_not_nil(node9.torch.Module, "torch.Module wrapper should exist")
    assert_not_nil(node9.torch.integrate, "torch.integrate should exist")
end)

-- Test 18: loadAll Function
test("loadAll function", function()
    node9.loadAll()
    -- Verify that core modules are loaded
    assert_not_nil(node9.pl.List, "List should be loaded after loadAll")
    assert_not_nil(node9.pl.stringx, "stringx should be loaded after loadAll")
end)

-- Test 19: Info Function
test("Info function", function()
    -- Just verify it doesn't error
    node9.info()
    assert_true(true, "info() should execute without error")
end)

-- Test 20: Version Information
test("Version information", function()
    assert_not_nil(node9.VERSION, "VERSION should be defined")
    assert_not_nil(node9.DESCRIPTION, "DESCRIPTION should be defined")
end)

-- Print Summary
print("")
print("=" .. string.rep("=", 70))
print("  Test Summary")
print("=" .. string.rep("=", 70))
print(string.format("Total tests: %d", test_count))
print(string.format("Passed: %d", passed_count))
print(string.format("Failed: %d", test_count - passed_count))
print("")

if #failed_tests > 0 then
    print("Failed tests:")
    for i, test in ipairs(failed_tests) do
        print(string.format("  %d. %s", i, test.name))
        print(string.format("     %s", test.error))
    end
    print("")
end

local success_rate = (passed_count / test_count) * 100
print(string.format("Success rate: %.1f%%", success_rate))
print("=" .. string.rep("=", 70))

-- Return exit code
if passed_count == test_count then
    print("\n✓ All tests passed!")
    os.exit(0)
else
    print("\n✗ Some tests failed")
    os.exit(1)
end
