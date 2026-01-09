-- ============================================================================
-- Integration Bridge: nn <-> optim
-- ============================================================================
-- This module provides seamless integration between neural network modules
-- and optimization algorithms, enhancing their interoperability.
-- ============================================================================

local nn_optim_bridge = {}

-- Ensure required modules are loaded
if not torch7u then
    error("torch7u integration layer not loaded. Please require 'init' first.")
end

local nn = torch7u.load_module('nn')
local optim = torch7u.load_module('optim')

if not nn or not optim then
    print("Warning: nn or optim not available, bridge not fully initialized")
    return nn_optim_bridge
end

-- ============================================================================
-- Enhanced Parameter Management
-- ============================================================================

local Module = torch.getmetatable('nn.Module')

-- Get parameters in optimizer-ready format
function Module:getOptimizableParameters()
    local params, gradParams = self:parameters()
    
    if not params then
        return nil, nil
    end
    
    -- Flatten parameters for optimizers that expect flat parameters
    local flatParams, flatGradParams
    
    if #params > 0 then
        flatParams = nn.utils.recursiveFlatten(params)
        flatGradParams = nn.utils.recursiveFlatten(gradParams)
    end
    
    return flatParams, flatGradParams, params, gradParams
end

-- Create an optimizer for this module
function Module:createOptimizer(optim_method, config)
    config = config or {}
    optim_method = optim_method or optim.sgd
    
    local params, gradParams = self:getOptimizableParameters()
    
    if not params then
        error("Module has no parameters to optimize")
    end
    
    local optimizer = {
        module = self,
        params = params,
        gradParams = gradParams,
        optim_method = optim_method,
        config = config,
        state = {},
        iteration = 0,
    }
    
    function optimizer:step(feval)
        self.iteration = self.iteration + 1
        return self.optim_method(feval, self.params, self.config, self.state)
    end
    
    function optimizer:reset()
        self.state = {}
        self.iteration = 0
    end
    
    function optimizer:getConfig()
        return self.config
    end
    
    function optimizer:setConfig(new_config)
        for k, v in pairs(new_config) do
            self.config[k] = v
        end
    end
    
    -- Register with torch7u
    torch7u.events.publish('optimizer_created', optimizer)
    
    return optimizer
end

-- Learning rate scheduling
function Module:createLRScheduler(initial_lr, schedule_type, params)
    schedule_type = schedule_type or 'step'
    params = params or {}
    
    local scheduler = {
        current_lr = initial_lr,
        initial_lr = initial_lr,
        schedule_type = schedule_type,
        params = params,
        epoch = 0,
    }
    
    function scheduler:step(epoch)
        epoch = epoch or (self.epoch + 1)
        self.epoch = epoch
        
        if self.schedule_type == 'step' then
            -- Step decay: lr = initial_lr * gamma ^ (epoch / step_size)
            local step_size = self.params.step_size or 10
            local gamma = self.params.gamma or 0.1
            self.current_lr = self.initial_lr * math.pow(gamma, math.floor(epoch / step_size))
            
        elseif self.schedule_type == 'exponential' then
            -- Exponential decay: lr = initial_lr * gamma ^ epoch
            local gamma = self.params.gamma or 0.95
            self.current_lr = self.initial_lr * math.pow(gamma, epoch)
            
        elseif self.schedule_type == 'cosine' then
            -- Cosine annealing
            local T_max = self.params.T_max or 100
            self.current_lr = self.initial_lr * (1 + math.cos(math.pi * epoch / T_max)) / 2
            
        elseif self.schedule_type == 'plateau' then
            -- Reduce on plateau (manual trigger needed)
            -- lr remains unchanged unless manually reduced
            
        end
        
        torch7u.events.publish('lr_scheduler_step', self.current_lr, epoch)
        return self.current_lr
    end
    
    function scheduler:get_lr()
        return self.current_lr
    end
    
    function scheduler:reset()
        self.current_lr = self.initial_lr
        self.epoch = 0
    end
    
    return scheduler
end

-- ============================================================================
-- Training Loop Integration
-- ============================================================================

function nn_optim_bridge.create_training_loop(model, criterion, optimizer_method, config)
    config = config or {}
    
    -- Separate optimizer config from training config
    local optimizer_config = config.optimizer_config or {
        learningRate = config.learningRate or 0.01,
        momentum = config.momentum or 0.0,
    }
    
    local loop = {
        model = model,
        criterion = criterion,
        optimizer_method = optimizer_method or optim.sgd,
        config = optimizer_config,
        optimizer_state = {},
        epoch = 0,
        iteration = 0,
        metrics = {
            train_loss = {},
            val_loss = {},
            train_accuracy = {},
            val_accuracy = {},
        },
    }
    
    function loop:train_step(input, target)
        self.iteration = self.iteration + 1
        
        -- Get parameters
        local params, gradParams = self.model:getOptimizableParameters()
        
        -- Define closure for optimizer
        local feval = function(x)
            if x ~= params then
                params:copy(x)
            end
            
            gradParams:zero()
            
            -- Forward pass
            local output = self.model:forward(input)
            local loss = self.criterion:forward(output, target)
            
            -- Backward pass
            local dloss_doutput = self.criterion:backward(output, target)
            self.model:backward(input, dloss_doutput)
            
            return loss, gradParams
        end
        
        -- Optimization step
        local _, loss = self.optimizer_method(feval, params, self.config, self.optimizer_state)
        
        torch7u.metrics.record('iteration_loss', loss[1], {
            epoch = self.epoch,
            iteration = self.iteration
        })
        
        return loss[1]
    end
    
    function loop:train_epoch(dataset, batch_size)
        batch_size = batch_size or 32
        self.epoch = self.epoch + 1
        
        self.model:training()
        
        local total_loss = 0
        local n_samples = 0
        
        torch7u.events.publish('epoch_start', self.epoch, self)
        
        -- Training loop
        for i = 1, dataset:size(), batch_size do
            local actual_batch_size = math.min(batch_size, dataset:size() - i + 1)
            
            -- Get batch
            local inputs = dataset.data:narrow(1, i, actual_batch_size)
            local targets = dataset.labels:narrow(1, i, actual_batch_size)
            
            -- Train step
            local loss = self:train_step(inputs, targets)
            total_loss = total_loss + loss * actual_batch_size
            n_samples = n_samples + actual_batch_size
        end
        
        local avg_loss = total_loss / n_samples
        table.insert(self.metrics.train_loss, avg_loss)
        
        torch7u.metrics.record('epoch_loss', avg_loss, { epoch = self.epoch, split = 'train' })
        torch7u.events.publish('epoch_end', self.epoch, avg_loss, self)
        
        return avg_loss
    end
    
    function loop:evaluate(dataset, batch_size)
        batch_size = batch_size or 32
        
        self.model:evaluate()
        
        local total_loss = 0
        local n_samples = 0
        
        for i = 1, dataset:size(), batch_size do
            local actual_batch_size = math.min(batch_size, dataset:size() - i + 1)
            
            local inputs = dataset.data:narrow(1, i, actual_batch_size)
            local targets = dataset.labels:narrow(1, i, actual_batch_size)
            
            local output = self.model:forward(inputs)
            local loss = self.criterion:forward(output, targets)
            
            total_loss = total_loss + loss * actual_batch_size
            n_samples = n_samples + actual_batch_size
        end
        
        local avg_loss = total_loss / n_samples
        table.insert(self.metrics.val_loss, avg_loss)
        
        torch7u.metrics.record('epoch_loss', avg_loss, { epoch = self.epoch, split = 'val' })
        
        return avg_loss
    end
    
    function loop:fit(train_dataset, val_dataset, n_epochs, batch_size)
        n_epochs = n_epochs or 10
        batch_size = batch_size or 32
        
        torch7u.events.publish('training_start', self)
        
        for epoch = 1, n_epochs do
            local train_loss = self:train_epoch(train_dataset, batch_size)
            
            local val_loss
            if val_dataset then
                val_loss = self:evaluate(val_dataset, batch_size)
                torch7u.utils.log('INFO', 
                    string.format('Epoch %d/%d - train_loss: %.4f - val_loss: %.4f', 
                        epoch, n_epochs, train_loss, val_loss),
                    'nn_optim_bridge')
            else
                torch7u.utils.log('INFO', 
                    string.format('Epoch %d/%d - train_loss: %.4f', 
                        epoch, n_epochs, train_loss),
                    'nn_optim_bridge')
            end
        end
        
        torch7u.events.publish('training_end', self)
        
        return self.metrics
    end
    
    return loop
end

-- ============================================================================
-- Optimizer Factory
-- ============================================================================

nn_optim_bridge.optimizers = {}

function nn_optim_bridge.create_optimizer(name, config)
    config = config or {}
    
    local optimizer_map = {
        sgd = optim.sgd,
        adam = optim.adam,
        adamax = optim.adamax,
        rmsprop = optim.rmsprop,
        adagrad = optim.adagrad,
        adadelta = optim.adadelta,
        lbfgs = optim.lbfgs,
        cg = optim.cg,
        asgd = optim.asgd,
        rprop = optim.rprop,
    }
    
    local optim_method = optimizer_map[name:lower()]
    if not optim_method then
        error("Unknown optimizer: " .. name)
    end
    
    return optim_method, config
end

-- ============================================================================
-- Gradient Checking Integration
-- ============================================================================

function Module:checkGradients(input, epsilon)
    epsilon = epsilon or 1e-4
    
    local output = self:forward(input)
    local gradOutput = torch.randn(output:size())
    
    -- Analytical gradient
    self:zeroGradParameters()
    self:backward(input, gradOutput)
    local params, gradParams = self:parameters()
    
    -- Numerical gradient
    local numGradParams = {}
    for i, param in ipairs(params) do
        numGradParams[i] = torch.Tensor():resizeAs(param):zero()
        
        for j = 1, param:nElement() do
            local orig = param[j]
            
            -- f(x + epsilon)
            param[j] = orig + epsilon
            local output_plus = self:forward(input)
            local loss_plus = torch.sum(torch.cmul(output_plus, gradOutput))
            
            -- f(x - epsilon)
            param[j] = orig - epsilon
            local output_minus = self:forward(input)
            local loss_minus = torch.sum(torch.cmul(output_minus, gradOutput))
            
            -- Numerical gradient
            numGradParams[i][j] = (loss_plus - loss_minus) / (2 * epsilon)
            
            -- Restore original value
            param[j] = orig
        end
    end
    
    -- Compare gradients
    local max_error = 0
    for i, gradParam in ipairs(gradParams) do
        local error = (gradParam - numGradParams[i]):abs():max()
        max_error = math.max(max_error, error)
    end
    
    torch7u.utils.log('INFO', 
        string.format('Gradient check max error: %.2e', max_error),
        'nn_optim_bridge')
    
    return max_error < 1e-3, max_error
end

-- ============================================================================
-- Register Integration
-- ============================================================================

torch7u.register('nn_optim_bridge', nn_optim_bridge, {'nn', 'optim'})

torch7u.utils.log('INFO', 'nn-optim integration bridge loaded', 'nn_optim_bridge')

return nn_optim_bridge
