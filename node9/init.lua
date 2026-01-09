-- node9: Integrated Node9 OS Framework for Torch7u
-- Deep integration of o9nn/node9 with torch7u neural network framework
-- 
-- Node9 is a hosted 64-bit operating system based on Bell Lab's Inferno OS,
-- using Lua scripting and LuaJIT for high performance.
--
-- This integration provides:
-- - Kernel functions and system calls
-- - Penrose Lua (pl) library utilities
-- - Filesystem abstractions
-- - Scheduler system
-- - Application framework
-- - Event-driven I/O

local node9 = {}

-- Version and metadata
node9.VERSION = "1.0.0"
node9.DESCRIPTION = "Node9 OS Framework Integration for Torch7u"

-- Module registry
node9._modules = {}
node9._initialized = false

-- Get the directory of this file
local script_path = debug.getinfo(1, "S").source:sub(2)
local script_dir = script_path:match("(.*[/\\])")

-- Configuration
node9.config = {
    root = script_dir or './node9',
    enable_kernel = true,
    enable_scheduler = true,
    enable_fs = true,
    verbose = false
}

-- Utility function for verbose logging
local function log(msg)
    if node9.config.verbose then
        print('[node9] ' .. msg)
    end
end

-- Initialize node9 integration
function node9.init(config)
    if node9._initialized then
        log("Already initialized")
        return node9
    end
    
    -- Merge user config with defaults
    if config then
        for k, v in pairs(config) do
            node9.config[k] = v
        end
    end
    
    log("Initializing node9 integration...")
    
    -- Set node9 root for compatibility with original node9 code
    _G._noderoot = node9.config.root
    
    -- Initialize core components
    node9._initialized = true
    log("Node9 initialized successfully")
    
    return node9
end

-- Penrose Lua (pl) library integration
-- Provides utility functions for tables, strings, paths, etc.
node9.pl = {}

function node9.pl.load()
    log("Loading Penrose Lua (pl) utilities...")
    
    -- Core pl modules
    node9.pl.List = require('node9.lib.pl.List')
    node9.pl.Map = require('node9.lib.pl.Map')
    node9.pl.Set = require('node9.lib.pl.Set')
    node9.pl.class = require('node9.lib.pl.class')
    node9.pl.tablex = require('node9.lib.pl.tablex')
    node9.pl.stringx = require('node9.lib.pl.stringx')
    node9.pl.utils = require('node9.lib.pl.utils')
    node9.pl.path = require('node9.lib.pl.path')
    node9.pl.file = require('node9.lib.pl.file')
    node9.pl.dir = require('node9.lib.pl.dir')
    node9.pl.pretty = require('node9.lib.pl.pretty')
    node9.pl.app = require('node9.lib.pl.app')
    node9.pl.lexer = require('node9.lib.pl.lexer')
    node9.pl.data = require('node9.lib.pl.data')
    node9.pl.Date = require('node9.lib.pl.Date')
    
    log("Penrose Lua utilities loaded")
    return node9.pl
end

-- Kernel integration
node9.kernel = {}

function node9.kernel.load()
    if not node9.config.enable_kernel then
        log("Kernel integration disabled")
        return nil
    end
    
    log("Loading kernel integration...")
    
    -- Load kernel module (requires FFI and C bindings)
    -- Note: This may require additional setup for FFI bindings
    local success, kernel = pcall(require, 'node9.lib.kernel')
    if success then
        node9.kernel = kernel
        log("Kernel loaded successfully")
    else
        log("Warning: Kernel could not be loaded (FFI bindings may be required): " .. tostring(kernel))
        node9.kernel.available = false
    end
    
    return node9.kernel
end

-- Scheduler integration
node9.scheduler = {}

function node9.scheduler.load()
    if not node9.config.enable_scheduler then
        log("Scheduler integration disabled")
        return nil
    end
    
    log("Loading scheduler...")
    
    local success, roundrobin = pcall(require, 'node9.lib.schedulers.roundrobin')
    if success then
        node9.scheduler.roundrobin = roundrobin
        log("Scheduler loaded successfully")
    else
        log("Warning: Scheduler could not be loaded: " .. tostring(roundrobin))
        node9.scheduler.available = false
    end
    
    return node9.scheduler
end

-- Filesystem integration
node9.fs = {}

function node9.fs.load()
    if not node9.config.enable_fs then
        log("Filesystem integration disabled")
        return nil
    end
    
    log("Loading filesystem modules...")
    
    local success, sys = pcall(require, 'node9.fs.sys')
    if success then
        node9.fs.sys = sys
        log("Filesystem sys module loaded")
    else
        log("Warning: Filesystem sys could not be loaded: " .. tostring(sys))
    end
    
    local success2, arg = pcall(require, 'node9.fs.arg')
    if success2 then
        node9.fs.arg = arg
        log("Filesystem arg module loaded")
    else
        log("Warning: Filesystem arg could not be loaded: " .. tostring(arg))
    end
    
    node9.fs.available = success or success2
    return node9.fs
end

-- Application framework
node9.appl = {}

function node9.appl.load()
    log("Loading application framework...")
    
    -- Store application module paths for dynamic loading
    node9.appl._module_dir = paths.concat(node9.config.root, 'appl')
    node9.appl.available = true
    
    log("Application framework ready")
    return node9.appl
end

-- Load a specific application module
function node9.appl.require(module_name)
    local module_path = 'node9.appl.' .. module_name
    log("Loading application module: " .. module_name)
    return require(module_path)
end

-- Torch7u integration helpers
node9.torch = {}

-- Create a node9-aware neural network module
function node9.torch.Module(module_class)
    local Node9Module = torch.class('node9.torch.Module', module_class or 'nn.Module')
    
    function Node9Module:__init()
        if module_class then
            module_class.__init(self)
        end
        self.node9_enabled = true
    end
    
    return Node9Module
end

-- Integrate node9 utilities with torch tensor operations
function node9.torch.integrate()
    log("Integrating node9 utilities with torch...")
    
    -- Add node9.pl utilities to torch namespace for convenience
    if node9.pl.tablex then
        torch.node9 = torch.node9 or {}
        torch.node9.tablex = node9.pl.tablex
        torch.node9.stringx = node9.pl.stringx
        torch.node9.utils = node9.pl.utils
    end
    
    log("Torch integration complete")
end

-- Load all modules
function node9.loadAll()
    log("Loading all node9 modules...")
    
    node9.pl.load()
    node9.kernel.load()
    node9.scheduler.load()
    node9.fs.load()
    node9.appl.load()
    node9.torch.integrate()
    
    log("All modules loaded")
    return node9
end

-- Info function to display integration status
function node9.info()
    print("=" .. string.rep("=", 70))
    print("  Node9 OS Framework Integration for Torch7u")
    print("=" .. string.rep("=", 70))
    print("")
    print("Version:     " .. node9.VERSION)
    print("Description: " .. node9.DESCRIPTION)
    print("Initialized: " .. tostring(node9._initialized))
    print("")
    print("Components:")
    print("  - Penrose Lua (pl):   " .. (node9.pl.List and "✓ Loaded" or "○ Not loaded"))
    print("  - Kernel:             " .. (node9.kernel.available == false and "✗ Unavailable (FFI required)" or node9.kernel.load and "○ Available" or "✓ Loaded"))
    print("  - Scheduler:          " .. (node9.scheduler.available == false and "✗ Unavailable" or node9.scheduler.roundrobin and "✓ Loaded" or "○ Available"))
    print("  - Filesystem:         " .. (node9.fs.available == false and "✗ Unavailable" or node9.fs.sys and "✓ Loaded" or "○ Available"))
    print("  - Applications:       " .. (node9.appl.available and "✓ Ready" or "○ Not loaded"))
    print("")
    print("Root:        " .. node9.config.root)
    print("=" .. string.rep("=", 70))
end

-- Auto-initialize with default config
node9.init()

return node9
