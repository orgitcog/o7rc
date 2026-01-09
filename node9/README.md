# Node9 OS Framework Integration

## Overview

This directory contains the **Node9 OS Framework**, deeply integrated into the Torch7u framework. Node9 is a hosted 64-bit operating system based on Bell Labs' Inferno OS, using Lua scripting and LuaJIT for high performance.

## What is Node9?

Node9 provides:
- **Kernel Functions**: System calls and kernel-level operations
- **Penrose Lua (pl)**: Comprehensive utility library for Lua
- **Scheduler System**: Process scheduling and management
- **Filesystem Abstractions**: Virtual filesystem operations
- **Application Framework**: Tools for building system-level applications
- **Event-Driven I/O**: Efficient I/O using libuv

## Integration Status

### ✓ Integrated Components

1. **Penrose Lua (pl) Library** - `node9/lib/pl/`
   - List, Map, Set, MultiMap, OrderedMap collections
   - String utilities (stringx)
   - Table utilities (tablex)
   - Path and file operations
   - Pretty printing
   - Lexer and data parsing
   - Date/time handling
   - Application framework (app, lapp)
   - Configuration management

2. **Kernel Module** - `node9/lib/kernel.lua`
   - System call interface (requires FFI)
   - Kernel data structures
   - Process management primitives

3. **Scheduler** - `node9/lib/schedulers/`
   - Round-robin scheduler
   - Process scheduling abstractions

4. **Filesystem Modules** - `node9/fs/`
   - sys.lua - System module interface
   - arg.lua - Argument parsing

5. **Application Framework** - `node9/appl/`
   - listen.lua - Network listeners
   - mount.lua, unmount.lua - Filesystem mounting
   - export.lua - Filesystem export
   - ls.lua - Directory listing
   - sh.lua - Shell operations
   - syscall.lua - System call interface
   - test.lua, apptest.lua - Testing utilities

6. **Source Files** - `node9/src/`
   - node9.lua - Core Node9 system definitions

## Usage

### Basic Usage

```lua
-- Load torch7u with node9 integration
require 'init'

-- Access node9
local node9 = torch7u.node9

-- Initialize node9 (auto-initialized by default)
node9.init({
    verbose = true,
    enable_kernel = true,
    enable_scheduler = true
})

-- Display node9 status
node9.info()
```

### Using Penrose Lua Utilities

```lua
local node9 = require 'node9'

-- Load Penrose Lua utilities
node9.pl.load()

-- Use List operations
local List = node9.pl.List
local mylist = List({1, 2, 3, 4, 5})
print(mylist:map(function(x) return x * 2 end))

-- Use string utilities
local stringx = node9.pl.stringx
print(stringx.split("hello,world", ","))

-- Use table utilities
local tablex = node9.pl.tablex
local t1 = {a=1, b=2}
local t2 = {c=3, d=4}
local merged = tablex.merge(t1, t2, true)

-- Use path operations
local path = node9.pl.path
print(path.join("path", "to", "file.lua"))
```

### Using Filesystem Operations

```lua
local node9 = require 'node9'
node9.fs.load()

-- Access filesystem modules
if node9.fs.sys then
    -- Use system module
    local sys = node9.fs.sys
end

if node9.fs.arg then
    -- Use argument parsing
    local arg = node9.fs.arg
end
```

### Using Application Framework

```lua
local node9 = require 'node9'
node9.appl.load()

-- Load specific application modules
local ls = node9.appl.require('ls')
local syscall = node9.appl.require('syscall')
```

### Torch Integration

```lua
local node9 = require 'node9'
node9.loadAll()

-- Access node9 utilities from torch namespace
print(torch.node9.tablex)
print(torch.node9.stringx)
print(torch.node9.utils)

-- Create node9-aware neural network module
local MyModule = node9.torch.Module('nn.Module')
local module = MyModule()
print(module.node9_enabled)  -- true
```

## Function-by-Function Integration

The Node9 framework has been integrated on a function-by-function basis with the Torch7u framework:

### Penrose Lua (pl) Integration

| Module | Functions | Integration Point |
|--------|-----------|-------------------|
| `pl.List` | map, filter, reduce, foreach | Available via `node9.pl.List` |
| `pl.Map` | Map class for key-value storage | Available via `node9.pl.Map` |
| `pl.Set` | Set operations (union, intersection, etc.) | Available via `node9.pl.Set` |
| `pl.stringx` | split, join, strip, startswith, endswith | Available via `node9.pl.stringx` |
| `pl.tablex` | merge, copy, deepcopy, keys, values | Available via `node9.pl.tablex` |
| `pl.path` | join, split, basename, dirname, exists | Available via `node9.pl.path` |
| `pl.file` | read, write, copy, move | Available via `node9.pl.file` |
| `pl.dir` | makepath, rmtree, getfiles | Available via `node9.pl.dir` |
| `pl.pretty` | dump, write | Available via `node9.pl.pretty` |
| `pl.utils` | Various utility functions | Available via `node9.pl.utils` |

### Kernel Integration

| Function | Description | Integration Point |
|----------|-------------|-------------------|
| System calls | Low-level system operations | `node9.kernel` (requires FFI) |
| Process management | Process creation and control | `node9.kernel` (requires FFI) |
| Memory operations | Kernel memory management | `node9.kernel` (requires FFI) |

### Scheduler Integration

| Function | Description | Integration Point |
|----------|-------------|-------------------|
| Round-robin scheduling | Basic process scheduler | `node9.scheduler.roundrobin` |

### Filesystem Integration

| Function | Description | Integration Point |
|----------|-------------|-------------------|
| System module | System-level operations | `node9.fs.sys` |
| Argument parsing | Parse command arguments | `node9.fs.arg` |

## Integration Architecture

```
torch7u/
├── node9/
│   ├── init.lua              # Main integration module
│   ├── lib/                  # Core libraries
│   │   ├── kernel.lua        # Kernel module (requires FFI)
│   │   ├── environments.lua  # Environment management
│   │   ├── schedulers/       # Scheduler implementations
│   │   │   └── roundrobin.lua
│   │   └── pl/               # Penrose Lua utilities
│   │       ├── List.lua      # List operations
│   │       ├── Map.lua       # Map data structure
│   │       ├── Set.lua       # Set operations
│   │       ├── stringx.lua   # String utilities
│   │       ├── tablex.lua    # Table utilities
│   │       ├── path.lua      # Path operations
│   │       ├── file.lua      # File operations
│   │       ├── dir.lua       # Directory operations
│   │       ├── utils.lua     # General utilities
│   │       ├── pretty.lua    # Pretty printing
│   │       └── ... (37 more modules)
│   ├── fs/                   # Filesystem modules
│   │   ├── sys.lua           # System module
│   │   └── arg.lua           # Argument parsing
│   ├── appl/                 # Application framework
│   │   ├── listen.lua        # Network listener
│   │   ├── mount.lua         # Filesystem mounting
│   │   ├── export.lua        # Filesystem export
│   │   ├── ls.lua            # Directory listing
│   │   ├── sh.lua            # Shell operations
│   │   └── ... (10 modules)
│   └── src/                  # Source files
│       └── node9.lua         # Core definitions
```

## Examples

### Example 1: Using Penrose Lua Collections

```lua
require 'init'
local node9 = torch7u.node9
node9.pl.load()

local List = node9.pl.List
local Set = node9.pl.Set

-- Create and manipulate lists
local numbers = List({1, 2, 3, 4, 5})
local doubled = numbers:map(function(x) return x * 2 end)
local evens = numbers:filter(function(x) return x % 2 == 0 end)
local sum = numbers:reduce('+')

print("Original:", numbers)
print("Doubled:", doubled)
print("Evens:", evens)
print("Sum:", sum)

-- Set operations
local set1 = Set({1, 2, 3, 4})
local set2 = Set({3, 4, 5, 6})
print("Union:", set1 + set2)
print("Intersection:", set1 * set2)
print("Difference:", set1 - set2)
```

### Example 2: String and Path Operations

```lua
require 'init'
local node9 = torch7u.node9
node9.pl.load()

local stringx = node9.pl.stringx
local path = node9.pl.path

-- String operations
local text = "  hello world  "
print("Stripped:", stringx.strip(text))
print("Split:", stringx.split("a,b,c", ","))
print("Join:", stringx.join(", ", {"a", "b", "c"}))

-- Path operations
local filepath = path.join("home", "user", "documents", "file.txt")
print("Path:", filepath)
print("Basename:", path.basename(filepath))
print("Dirname:", path.dirname(filepath))
print("Extension:", path.extension(filepath))
```

### Example 3: Table Utilities

```lua
require 'init'
local node9 = torch7u.node9
node9.pl.load()

local tablex = node9.pl.tablex

-- Deep copy
local original = {a = 1, b = {c = 2, d = 3}}
local copy = tablex.deepcopy(original)

-- Merge tables
local t1 = {a = 1, b = 2}
local t2 = {c = 3, d = 4}
local merged = tablex.merge(t1, t2, true)

-- Extract keys and values
local keys = tablex.keys(merged)
local values = tablex.values(merged)

print("Merged:", tablex.pprint(merged))
print("Keys:", tablex.pprint(keys))
print("Values:", tablex.pprint(values))
```

### Example 4: Integration with Neural Networks

```lua
require 'init'
local node9 = torch7u.node9
node9.loadAll()

-- Use node9 utilities in data preprocessing
local stringx = node9.pl.stringx
local tablex = node9.pl.tablex
local List = node9.pl.List

-- Create a neural network with node9-aware capabilities
local model = nn.Sequential()
model:add(nn.Linear(100, 50))
model:add(nn.ReLU())
model:add(nn.Linear(50, 10))

-- Register with torch7u model registry
torch7u.models.register('my_classifier', model, {
    description = "Classifier with node9 integration",
    node9_enabled = true,
    preprocessing = function(data)
        -- Use node9 utilities for preprocessing
        local processed = tablex.map(function(x) return x * 2 end, data)
        return processed
    end
})

-- Use node9 utilities for data manipulation
local data_list = List({1, 2, 3, 4, 5})
local normalized = data_list:map(function(x) return x / 10.0 end)
```

## Requirements

- **LuaJIT**: Node9 requires LuaJIT for performance
- **FFI (Optional)**: Kernel functionality requires FFI bindings
- **Torch7u**: Main framework must be loaded first

## Notes

- **Kernel Module**: The kernel module requires FFI bindings to C code. It will be available only if the FFI bindings are properly set up.
- **Scheduler**: The scheduler module provides process scheduling abstractions but may require additional C bindings for full functionality.
- **Pure Lua Components**: All Penrose Lua (pl) utilities are pure Lua and work without additional dependencies.

## Integration Benefits

1. **Enhanced String Processing**: Powerful string utilities beyond standard Lua
2. **Advanced Collections**: List, Set, Map data structures with functional operations
3. **Path Operations**: Cross-platform path manipulation
4. **File Operations**: Higher-level file and directory operations
5. **Pretty Printing**: Better output formatting for debugging
6. **Extensible Framework**: Foundation for system-level operations in neural network applications

## Future Work

- Complete FFI bindings for kernel functionality
- Additional scheduler implementations
- Extended filesystem operations
- Integration with torch7u event system
- Node9-aware training loops
- Distributed computing support using Node9 scheduler

## Source

Original repository: https://github.com/o9nn/node9

Node9 is based on Bell Labs' Inferno OS and uses:
- Lua scripting language
- LuaJIT virtual machine
- libuv I/O library

## License

See individual component licenses in the original Node9 repository.

---

**Integration Version**: 1.0.0  
**Status**: ✓ Core components integrated, ready for use  
**Author**: Torch7u Integration Team
