# Node9 Quick Reference Guide

## Loading Node9

```lua
-- Method 1: Via torch7u
require 'init'
local node9 = torch7u.node9

-- Method 2: Direct require
local node9 = require('node9')

-- Load all components
node9.loadAll()

-- Display info
node9.info()
```

## Penrose Lua (pl) Utilities

### List Operations

```lua
node9.pl.load()
local List = node9.pl.List

local list = List({1, 2, 3, 4, 5})

-- Functional operations
list:map(function(x) return x * 2 end)        -- {2, 4, 6, 8, 10}
list:filter(function(x) return x % 2 == 0 end) -- {2, 4}
list:reduce('+')                               -- 15
list:reduce('*')                               -- 120
list:foreach(print)                            -- Print each element

-- List manipulation
list:append(6)                                 -- Add element
list:extend({7, 8, 9})                        -- Add multiple
list:slice(2, 4)                              -- Extract sublist
```

### Set Operations

```lua
local Set = node9.pl.Set

local s1 = Set({1, 2, 3, 4})
local s2 = Set({3, 4, 5, 6})

s1 + s2  -- Union: {1, 2, 3, 4, 5, 6}
s1 * s2  -- Intersection: {3, 4}
s1 - s2  -- Difference: {1, 2}
```

### String Utilities

```lua
local stringx = node9.pl.stringx

stringx.strip("  hello  ")                    -- "hello"
stringx.split("a,b,c", ",")                   -- {"a", "b", "c"}
stringx.join(", ", {"a", "b", "c"})           -- "a, b, c"
stringx.startswith("hello", "hel")            -- true
stringx.endswith("world", "rld")              -- true
stringx.replace("hello", "l", "L")            -- "heLLo"
stringx.count("hello", "l")                   -- 2
```

### Table Utilities

```lua
local tablex = node9.pl.tablex

-- Merge tables
local t1 = {a=1, b=2}
local t2 = {c=3, d=4}
tablex.merge(t1, t2, true)                    -- {a=1, b=2, c=3, d=4}

-- Deep copy
local nested = {a=1, b={c=2}}
tablex.deepcopy(nested)

-- Extract keys/values
tablex.keys({a=1, b=2})                       -- {"a", "b"}
tablex.values({a=1, b=2})                     -- {1, 2}

-- Map/filter tables
tablex.map(function(v) return v*2 end, {1,2,3}) -- {2,4,6}
tablex.filter({1,2,3,4}, function(v) return v>2 end) -- {3,4}

-- Table info
tablex.size({a=1, b=2, c=3})                  -- 3
tablex.find({1,2,3,4}, 3)                     -- 3
```

### Path Operations

```lua
local path = node9.pl.path

path.join("home", "user", "file.txt")         -- "home/user/file.txt"
path.basename("/home/user/file.txt")          -- "file.txt"
path.dirname("/home/user/file.txt")           -- "/home/user"
path.extension("file.txt")                    -- ".txt"
path.splitext("file.txt")                     -- "file", ".txt"
path.isabs("/home/user")                      -- true
```

### File Operations

```lua
local file = node9.pl.file

file.read("data.txt")                         -- Read entire file
file.write("output.txt", "content")           -- Write to file
```

### Directory Operations

```lua
local dir = node9.pl.dir

dir.makepath("path/to/dir")                   -- Create directory
dir.getfiles(".", "*.lua")                    -- Get Lua files
dir.getdirectories(".")                       -- Get subdirectories
```

### Pretty Printing

```lua
local pretty = node9.pl.pretty

local data = {
    name = "model",
    config = {lr=0.01, epochs=100}
}
print(pretty.write(data))
```

### Map/OrderedMap

```lua
local Map = node9.pl.Map
local OrderedMap = node9.pl.OrderedMap

-- Regular map
local map = Map({a=1, b=2})
map.c = 3

-- Ordered map (maintains insertion order)
local omap = OrderedMap()
omap:set("first", 1)
omap:set("second", 2)
for k, v in omap:iter() do
    print(k, v)  -- Preserves order
end
```

### Date Operations

```lua
local Date = node9.pl.Date

local now = Date()
local tomorrow = now:add_days(1)
print(now:fmt("%Y-%m-%d"))
```

### Utilities

```lua
local utils = node9.pl.utils

utils.type(value)                             -- Get type
utils.is_type(value, 'string')                -- Check type
utils.assert_arg(1, value, 'string')          -- Assert argument type
```

## Integration with Neural Networks

```lua
local node9 = require('node9')
node9.loadAll()

-- Create model
local model = nn.Sequential()
    :add(nn.Linear(100, 50))
    :add(nn.ReLU())
    :add(nn.Linear(50, 10))

-- Use node9 utilities for preprocessing
local List = node9.pl.List
local data = List({1, 2, 3, 4, 5})
local normalized = data:map(function(x) return x / 10.0 end)

-- Register with torch7u
torch7u.models.register('classifier', model, {
    description = "Node9-enabled classifier",
    preprocessing = function(x)
        return node9.pl.tablex.map(function(v) return v / 255 end, x)
    end
})
```

## Common Patterns

### Data Pipeline

```lua
local List = node9.pl.List
local stringx = node9.pl.stringx

-- Parse CSV
function parse_csv(text)
    local lines = stringx.split(text, "\n")
    return List(lines):map(function(line)
        return stringx.split(line, ",")
    end)
end

-- Process data
local data = parse_csv(csv_text)
local processed = data:map(function(row)
    return tablex.map(tonumber, row)
end)
```

### Configuration Management

```lua
local tablex = node9.pl.tablex
local pretty = node9.pl.pretty

-- Load and merge configs
local default = {lr=0.01, batch=32}
local user = {lr=0.001}
local config = tablex.merge(default, user, true)

-- Print config
print(pretty.write(config))
```

### Model Path Management

```lua
local path = node9.pl.path

function get_checkpoint_path(name, epoch)
    return path.join("checkpoints", name, "epoch_" .. epoch .. ".t7")
end

function parse_checkpoint(filepath)
    local basename = path.basename(filepath)
    local name = path.splitext(basename)
    return name
end
```

## Configuration

```lua
-- Initialize with custom config
node9.init({
    verbose = true,              -- Enable verbose logging
    enable_kernel = false,       -- Disable kernel (if FFI not available)
    enable_scheduler = true,     -- Enable scheduler
    enable_fs = true,            -- Enable filesystem
})
```

## Available Components

| Component | Status | Description |
|-----------|--------|-------------|
| **pl.List** | ✓ | Enhanced list operations |
| **pl.Set** | ✓ | Set operations |
| **pl.Map** | ✓ | Key-value storage |
| **pl.OrderedMap** | ✓ | Ordered dictionary |
| **pl.MultiMap** | ✓ | One-to-many mapping |
| **pl.stringx** | ✓ | String utilities |
| **pl.tablex** | ✓ | Table utilities |
| **pl.path** | ✓ | Path operations |
| **pl.file** | ✓ | File I/O |
| **pl.dir** | ✓ | Directory operations |
| **pl.utils** | ✓ | General utilities |
| **pl.pretty** | ✓ | Pretty printing |
| **pl.Date** | ✓ | Date/time operations |
| **kernel** | ⚠️ | Kernel functions (requires FFI) |
| **scheduler** | ✓ | Process scheduling |
| **fs** | ⚠️ | Filesystem (partial FFI) |
| **appl** | ✓ | Application framework |

## Documentation

- **node9/README.md** - Complete integration guide
- **node9/example.lua** - Comprehensive examples
- **node9/test.lua** - Test suite
- **NODE9_INTEGRATION_SUMMARY.md** - Integration details

## Testing

```bash
# Run tests
th node9/test.lua

# Run examples
th node9/example.lua

# Validate integration
./validate_node9_integration.sh
```

---

For complete documentation, see **node9/README.md**
