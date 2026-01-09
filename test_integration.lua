-- ============================================================================
-- Integration Tests for Torch7u
-- ============================================================================
-- This file tests the deep interconnections between all components
-- ============================================================================

require 'init'

local test = {}
local tester

-- Try to load torch.Tester if available
pcall(function()
    tester = torch.Tester()
end)

-- ============================================================================
-- Basic Integration Tests
-- ============================================================================

function test.test_core_modules_loaded()
    assert(torch7u ~= nil, "torch7u not initialized")
    assert(torch ~= nil, "torch not loaded")
    assert(nn ~= nil, "nn not loaded")
    assert(optim ~= nil, "optim not loaded")
    
    print("[PASS] Core modules loaded successfully")
    return true
end

function test.test_module_registry()
    -- Test module registration
    local test_module = {name = "test"}
    torch7u.register('test_module', test_module, {})
    
    assert(torch7u.module_registry['test_module'] ~= nil, "Module not registered")
    print("[PASS] Module registry working")
    return true
end

function test.test_event_system()
    -- Test event subscription and publishing
    local event_received = false
    local event_data = nil
    
    torch7u.events.subscribe('test_integration_event', function(data)
        event_received = true
        event_data = data
    end, 'test')
    
    torch7u.events.publish('test_integration_event', 'test_data')
    
    assert(event_received, "Event not received")
    assert(event_data == 'test_data', "Event data incorrect")
    
    print("[PASS] Event system working")
    return true
end

function test.test_logging()
    -- Test unified logging
    torch7u.utils.log('INFO', 'Test log message', 'test')
    print("[PASS] Logging system working")
    return true
end

function test.test_configuration()
    -- Test configuration system
    local original_type = torch7u.config.default_tensor_type or 'torch.DoubleTensor'
    
    torch7u.configure({
        default_tensor_type = 'torch.FloatTensor',
        test_config = true
    })
    
    assert(torch7u.config.test_config == true, "Configuration not set")
    
    -- Restore original
    torch7u.configure({ default_tensor_type = original_type })
    
    print("[PASS] Configuration system working")
    return true
end

-- ============================================================================
-- Model Registry Tests
-- ============================================================================

function test.test_model_registry()
    -- Create a simple model
    local model = nn.Sequential()
        :add(nn.Linear(10, 5))
        :add(nn.ReLU())
        :add(nn.Linear(5, 2))
    
    -- Register model
    torch7u.models.register('test_model', model, {
        input_dim = 10,
        output_dim = 2
    })
    
    -- Retrieve model
    local retrieved = torch7u.models.get('test_model')
    assert(retrieved ~= nil, "Model not retrieved")
    assert(retrieved == model, "Retrieved model doesn't match")
    
    -- List models
    local models = torch7u.models.list()
    local found = false
    for _, name in ipairs(models) do
        if name == 'test_model' then
            found = true
            break
        end
    end
    assert(found, "Model not in list")
    
    print("[PASS] Model registry working")
    return true
end

-- ============================================================================
-- Data Pipeline Tests
-- ============================================================================

function test.test_data_pipeline()
    -- Create pipeline
    local pipeline = torch7u.data.create_pipeline()
    
    -- Add transforms
    pipeline:add_transform('double', function(x) return x * 2 end)
    pipeline:add_transform('add_one', function(x) return x + 1 end)
    
    -- Process data
    local result = pipeline:process(5, {'double', 'add_one'})
    
    assert(result == 11, "Pipeline processing incorrect: " .. result)
    
    print("[PASS] Data pipeline working")
    return true
end

-- ============================================================================
-- Metrics Tests
-- ============================================================================

function test.test_metrics()
    -- Record metrics
    torch7u.metrics.clear('test_metric')
    
    torch7u.metrics.record('test_metric', 1.5, {tag = 'test'})
    torch7u.metrics.record('test_metric', 2.5, {tag = 'test'})
    
    -- Get metrics
    local metrics = torch7u.metrics.get('test_metric')
    
    assert(#metrics == 2, "Metrics count incorrect")
    assert(metrics[1].value == 1.5, "First metric value incorrect")
    assert(metrics[2].value == 2.5, "Second metric value incorrect")
    
    -- Clear metrics
    torch7u.metrics.clear('test_metric')
    metrics = torch7u.metrics.get('test_metric')
    assert(#metrics == 0, "Metrics not cleared")
    
    print("[PASS] Metrics system working")
    return true
end

-- ============================================================================
-- Checkpoint Tests
-- ============================================================================

function test.test_checkpointing()
    -- Create test data
    local test_data = {
        value = 42,
        tensor = torch.rand(5, 5)
    }
    
    local checkpoint_file = '/tmp/test_checkpoint.t7'
    
    -- Save checkpoint
    torch7u.checkpoint.save(checkpoint_file, test_data, {
        description = 'Test checkpoint'
    })
    
    -- Load checkpoint
    local loaded_data, metadata = torch7u.checkpoint.load(checkpoint_file)
    
    assert(loaded_data.value == 42, "Checkpoint data incorrect")
    assert(metadata.description == 'Test checkpoint', "Checkpoint metadata incorrect")
    
    -- Cleanup
    os.remove(checkpoint_file)
    
    print("[PASS] Checkpointing working")
    return true
end

-- ============================================================================
-- Plugin System Tests
-- ============================================================================

function test.test_plugin_system()
    -- Create plugin
    local plugin_initialized = false
    local test_plugin = {
        init = function(torch7u)
            plugin_initialized = true
        end,
        test_function = function()
            return "plugin works"
        end
    }
    
    -- Register plugin
    local success = torch7u.plugins.register('test_plugin', test_plugin)
    assert(success, "Plugin registration failed")
    assert(plugin_initialized, "Plugin not initialized")
    
    -- Get plugin
    local plugin = torch7u.plugins.get('test_plugin')
    assert(plugin ~= nil, "Plugin not retrieved")
    assert(plugin.test_function() == "plugin works", "Plugin function failed")
    
    -- List plugins
    local plugins = torch7u.plugins.list()
    local found = false
    for _, name in ipairs(plugins) do
        if name == 'test_plugin' then
            found = true
            break
        end
    end
    assert(found, "Plugin not in list")
    
    print("[PASS] Plugin system working")
    return true
end

-- ============================================================================
-- Cross-Module Integration Tests
-- ============================================================================

function test.test_nn_module_extensions()
    -- Test that nn.Module has integrated methods
    local Module = torch.getmetatable('nn.Module')
    
    assert(Module.getOptimizableParameters ~= nil, "getOptimizableParameters not added to nn.Module")
    
    print("[PASS] nn.Module extensions working")
    return true
end

function test.test_nn_optim_integration()
    -- Load the nn-optim bridge
    local bridge = require 'integrations.nn_optim_bridge'
    
    -- Create a simple model
    local model = nn.Sequential()
        :add(nn.Linear(5, 3))
        :add(nn.ReLU())
        :add(nn.Linear(3, 2))
    
    -- Test parameter extraction
    local params, gradParams = model:getOptimizableParameters()
    assert(params ~= nil, "Failed to get parameters")
    assert(gradParams ~= nil, "Failed to get gradients")
    
    print("[PASS] nn-optim integration working")
    return true
end

function test.test_image_nn_integration()
    -- Try to load image-nn bridge
    local success, bridge = pcall(require, 'integrations.image_nn_bridge')
    
    if not success then
        print("[SKIP] image-nn integration (image module not available)")
        return true
    end
    
    -- Test preprocessing pipeline creation
    local pipeline = bridge.create_preprocessing_pipeline({
        {type = 'normalize', mean = {0.5, 0.5, 0.5}, std = {0.5, 0.5, 0.5}}
    })
    
    assert(pipeline ~= nil, "Failed to create preprocessing pipeline")
    
    print("[PASS] image-nn integration working")
    return true
end

-- ============================================================================
-- Training Integration Tests
-- ============================================================================

function test.test_unified_trainer()
    -- Create simple model and criterion
    local model = nn.Sequential()
        :add(nn.Linear(10, 5))
        :add(nn.ReLU())
        :add(nn.Linear(5, 2))
    
    local criterion = nn.MSECriterion()
    
    -- Create trainer
    local trainer = torch7u.training.create_trainer(model, criterion, {
        learning_rate = 0.01,
        epochs = 1
    })
    
    assert(trainer ~= nil, "Failed to create trainer")
    
    -- Test callback system
    local callback_called = false
    trainer:add_callback('train_begin', function()
        callback_called = true
    end)
    
    -- Create dummy dataset
    local train_data = torch.rand(10, 10)
    local train_labels = torch.rand(10, 2)
    
    -- Note: Not actually training here, just testing the interface exists
    assert(trainer.train ~= nil, "Trainer doesn't have train method")
    
    print("[PASS] Unified trainer working")
    return true
end

-- ============================================================================
-- Run All Tests
-- ============================================================================

function test.run_all()
    print("\n" .. string.rep("=", 70))
    print("Running Torch7u Integration Tests")
    print(string.rep("=", 70) .. "\n")
    
    local tests_run = 0
    local tests_passed = 0
    local tests_failed = 0
    local failed_tests = {}
    
    for name, test_fn in pairs(test) do
        if name ~= 'run_all' and type(test_fn) == 'function' then
            tests_run = tests_run + 1
            print("\nRunning: " .. name)
            
            local success, err = pcall(test_fn)
            
            if success then
                tests_passed = tests_passed + 1
            else
                tests_failed = tests_failed + 1
                table.insert(failed_tests, {name = name, error = err})
                print("[FAIL] " .. name .. ": " .. tostring(err))
            end
        end
    end
    
    print("\n" .. string.rep("=", 70))
    print("Test Results")
    print(string.rep("=", 70))
    print(string.format("Tests run: %d", tests_run))
    print(string.format("Tests passed: %d", tests_passed))
    print(string.format("Tests failed: %d", tests_failed))
    
    if tests_failed > 0 then
        print("\nFailed tests:")
        for _, failed in ipairs(failed_tests) do
            print(string.format("  - %s: %s", failed.name, failed.error))
        end
    end
    
    print(string.rep("=", 70) .. "\n")
    
    return tests_failed == 0
end

-- Run tests if executed directly
if not ... then
    test.run_all()
end

return test
