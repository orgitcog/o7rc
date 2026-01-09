#!/usr/bin/env th
--[[
    Node9 Integration Example for Torch7u
    
    This example demonstrates the deep integration of Node9 OS framework
    with Torch7u neural network framework, showing function-by-function
    integration on various levels.
]]--

print("=" .. string.rep("=", 70))
print("  Node9 OS Framework Integration Example")
print("=" .. string.rep("=", 70))
print("")

-- Load torch7u with node9 integration
require 'init'

-- Access the integrated node9 module
local node9 = torch7u.node9 or require('node9')

print("1. Checking Node9 Integration Status")
print(string.rep("-", 70))
node9.info()
print("")

-- Example 1: Penrose Lua (pl) Utilities
print("\n2. Example: Penrose Lua (pl) Utilities")
print(string.rep("-", 70))

-- Load Penrose Lua utilities
node9.pl.load()

-- List operations
local List = node9.pl.List
print("\nList Operations:")
local mylist = List({1, 2, 3, 4, 5, 6, 7, 8, 9, 10})
print("  Original list:", mylist)
print("  Doubled:", mylist:map(function(x) return x * 2 end))
print("  Evens:", mylist:filter(function(x) return x % 2 == 0 end))
print("  Sum:", mylist:reduce('+'))
print("  Product:", mylist:reduce('*'))

-- Set operations
local Set = node9.pl.Set
print("\nSet Operations:")
local set1 = Set({1, 2, 3, 4, 5})
local set2 = Set({4, 5, 6, 7, 8})
print("  Set 1:", set1)
print("  Set 2:", set2)
print("  Union:", set1 + set2)
print("  Intersection:", set1 * set2)
print("  Difference (set1 - set2):", set1 - set2)

-- String utilities
local stringx = node9.pl.stringx
print("\nString Utilities:")
local text = "  hello world from node9  "
print("  Original: '" .. text .. "'")
print("  Stripped: '" .. stringx.strip(text) .. "'")
print("  Split: ", stringx.split("a,b,c,d,e", ","))
print("  Join: ", stringx.join(" | ", {"torch", "node9", "integration"}))
print("  Starts with 'hello':", stringx.startswith(stringx.strip(text), "hello"))
print("  Ends with 'node9':", stringx.endswith(stringx.strip(text), "node9"))

-- Table utilities
local tablex = node9.pl.tablex
print("\nTable Utilities:")
local t1 = {a = 1, b = 2, c = 3}
local t2 = {d = 4, e = 5, f = 6}
local merged = tablex.merge(t1, t2, true)
print("  Table 1:", t1)
print("  Table 2:", t2)
print("  Merged:", merged)
print("  Keys:", tablex.keys(merged))
print("  Values:", tablex.values(merged))
print("  Size:", tablex.size(merged))

-- Deep copy
local nested = {a = 1, b = {c = 2, d = {e = 3}}}
local copy = tablex.deepcopy(nested)
copy.b.d.e = 999  -- Modify copy
print("  Original nested table e:", nested.b.d.e)  -- Should still be 3
print("  Copied nested table e:", copy.b.d.e)      -- Should be 999

-- Path operations
local path = node9.pl.path
print("\nPath Utilities:")
local filepath = path.join("home", "user", "models", "classifier.t7")
print("  Constructed path:", filepath)
print("  Basename:", path.basename(filepath))
print("  Dirname:", path.dirname(filepath))
print("  Extension:", path.extension(filepath))

-- Example 2: Integration with Neural Networks
print("\n3. Example: Integration with Neural Networks")
print(string.rep("-", 70))

-- Load nn if not already loaded
pcall(require, 'nn')

if nn then
    print("\nCreating a neural network with node9 utilities...")
    
    -- Create a simple neural network
    local model = nn.Sequential()
    model:add(nn.Linear(100, 50))
    model:add(nn.ReLU())
    model:add(nn.Linear(50, 25))
    model:add(nn.ReLU())
    model:add(nn.Linear(25, 10))
    model:add(nn.LogSoftMax())
    
    print("  Model architecture:")
    print(model)
    
    -- Register with torch7u model registry
    if torch7u and torch7u.models then
        torch7u.models.register('node9_classifier', model, {
            description = "Classifier with node9 integration",
            node9_enabled = true,
            input_size = 100,
            output_size = 10,
            created_with = "node9 integration example"
        })
        print("\n  Model registered in torch7u registry as 'node9_classifier'")
    end
    
    -- Use node9 utilities for data preprocessing
    print("\n  Using node9 utilities for data preprocessing:")
    
    local data_list = List({1, 2, 3, 4, 5})
    local normalized = data_list:map(function(x) return x / 10.0 end)
    print("  Original data:", data_list)
    print("  Normalized:", normalized)
    
    -- Create sample input
    local input = torch.randn(100)
    local output = model:forward(input)
    print("\n  Forward pass:")
    print("    Input size:", input:size(1))
    print("    Output size:", output:size(1))
    print("    Output sample (first 5):", output[{{1,5}}])
    
    -- Use node9.pl.pretty for better output formatting
    local pretty = node9.pl.pretty
    print("\n  Using pretty printing for model stats:")
    local stats = {
        input_dim = 100,
        hidden_layers = {50, 25},
        output_dim = 10,
        total_params = model:getParameters():nElement(),
        activation = "ReLU",
        output_activation = "LogSoftMax"
    }
    print(pretty.write(stats, "  "))
else
    print("  nn module not available, skipping neural network example")
end

-- Example 3: Data Pipeline with Node9 Utilities
print("\n4. Example: Data Pipeline with Node9 Utilities")
print(string.rep("-", 70))

print("\nCreating data pipeline using node9 utilities...")

-- Simulate loading data with various preprocessing steps
local function create_data_pipeline()
    local pipeline = {}
    
    -- Step 1: Load and parse data (using stringx)
    function pipeline.load_csv(text)
        local lines = stringx.split(text, "\n")
        local data = List({})
        for _, line in ipairs(lines) do
            if line ~= "" then
                local values = stringx.split(line, ",")
                data:append(values)
            end
        end
        return data
    end
    
    -- Step 2: Filter and transform (using List and tablex)
    function pipeline.preprocess(data)
        return data:map(function(row)
            -- Convert strings to numbers and normalize
            return tablex.map(function(val)
                local num = tonumber(val) or 0
                return num / 100.0  -- Normalize
            end, row)
        end)
    end
    
    -- Step 3: Batch and convert to tensors
    function pipeline.to_tensors(data)
        local tensors = List({})
        for i, row in ipairs(data) do
            tensors:append(torch.Tensor(row))
        end
        return tensors
    end
    
    return pipeline
end

-- Use the pipeline
local pipeline = create_data_pipeline()
local csv_data = "10,20,30,40\n50,60,70,80\n90,100,110,120"
print("  Sample CSV data:")
print("    " .. csv_data:gsub("\n", "\n    "))

local parsed = pipeline.load_csv(csv_data)
print("\n  Parsed data:", parsed)

local preprocessed = pipeline.preprocess(parsed)
print("  Preprocessed (normalized):", preprocessed)

local tensors = pipeline.to_tensors(preprocessed)
print("  Converted to tensors:")
for i, tensor in ipairs(tensors) do
    print("    Row " .. i .. ":", tensor)
end

-- Example 4: Configuration Management
print("\n5. Example: Configuration Management")
print(string.rep("-", 70))

print("\nUsing node9.pl utilities for configuration...")

local config = {
    model = {
        type = "sequential",
        layers = {
            {type = "Linear", in_features = 100, out_features = 50},
            {type = "ReLU"},
            {type = "Linear", in_features = 50, out_features = 10}
        }
    },
    training = {
        batch_size = 32,
        learning_rate = 0.001,
        epochs = 100,
        optimizer = "SGD"
    },
    data = {
        train_path = "/data/train",
        test_path = "/data/test",
        validation_split = 0.2
    }
}

local pretty = node9.pl.pretty
print("  Configuration:")
print(pretty.write(config, "  "))

-- Extract specific config sections using tablex
print("\n  Model config only:")
print(pretty.write(config.model, "  "))

-- Example 5: Utility Functions Integration
print("\n6. Example: Utility Functions Integration")
print(string.rep("-", 70))

print("\nDemonstrating various node9 utility integrations...")

-- Map operations
local Map = node9.pl.Map
local metadata_map = Map({
    model_id = "classifier_v1",
    created = os.date(),
    author = "torch7u",
    version = "1.0.0"
})
print("  Metadata Map:")
for key, value in pairs(metadata_map) do
    print("    " .. key .. ": " .. tostring(value))
end

-- Ordered Map for maintaining insertion order
local OrderedMap = node9.pl.OrderedMap
local steps = OrderedMap()
steps:set("load_data", "Load training data")
steps:set("preprocess", "Preprocess and normalize")
steps:set("train", "Train the model")
steps:set("evaluate", "Evaluate on test set")
steps:set("save", "Save the trained model")

print("\n  Training Pipeline Steps (ordered):")
for key, value in steps:iter() do
    print("    " .. key .. ": " .. value)
end

-- Utils for various operations
local utils = node9.pl.utils
print("\n  Utility functions:")
print("    Type of list:", utils.type(mylist))
print("    Type of model:", utils.type(model or {}))
print("    Is table:", utils.is_type({}, 'table'))
print("    Is string:", utils.is_type("hello", 'string'))

-- Summary
print("\n" .. string.rep("=", 70))
print("  Integration Summary")
print(string.rep("=", 70))
print("\nNode9 integration provides:")
print("  ✓ 37+ Penrose Lua (pl) utility modules")
print("  ✓ Advanced collection types (List, Set, Map, OrderedMap)")
print("  ✓ Enhanced string and table operations")
print("  ✓ Path and file utilities")
print("  ✓ Pretty printing and formatting")
print("  ✓ Configuration management")
print("  ✓ Seamless integration with torch7u neural networks")
print("  ✓ Data pipeline utilities")
print("  ✓ Function-by-function deep integration")
print("")
print("All node9 components are accessible via:")
print("  - torch7u.node9.*")
print("  - require('node9')")
print("")
print("For more information, see: node9/README.md")
print(string.rep("=", 70))
