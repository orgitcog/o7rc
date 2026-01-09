-- ============================================================================
-- Integration Bridge: threads <-> nn
-- ============================================================================
-- This module provides seamless integration between multi-threading and
-- neural network training for parallel data loading and processing.
-- ============================================================================

local threads_nn_bridge = {}

-- Ensure required modules are loaded
if not torch7u then
    error("torch7u integration layer not loaded. Please require 'init' first.")
end

local threads = torch7u.load_module('threads')
local nn = torch7u.load_module('nn')

if not threads or not nn then
    print("Warning: threads or nn not available, bridge not fully initialized")
    return threads_nn_bridge
end

-- ============================================================================
-- Parallel Data Loader
-- ============================================================================

function threads_nn_bridge.create_data_loader(n_threads, dataset, options)
    options = options or {}
    
    local loader = {
        n_threads = n_threads,
        dataset = dataset,
        batch_size = options.batch_size or 32,
        shuffle = options.shuffle or true,
        transform = options.transform,
        pool = nil,
        queue = {},
    }
    
    -- Validate dataset interface
    if not dataset or type(dataset.size) ~= 'function' then
        error("Dataset must have a size() method")
    end
    
    -- Initialize thread pool
    function loader:init()
        self.pool = threads.Threads(
            self.n_threads,
            function()
                require 'torch'
            end,
            function(thread_id)
                -- Per-thread initialization
                torch.manualSeed(thread_id)
            end
        )
        
        torch7u.utils.log('INFO', 
            string.format('Data loader initialized with %d threads', self.n_threads),
            'threads_nn_bridge')
        
        return self
    end
    
    -- Load batch asynchronously
    function loader:load_batch_async(indices)
        self.pool:addjob(
            function(indices, transform)
                local batch_data = {}
                local batch_labels = {}
                
                for _, idx in ipairs(indices) do
                    local data, label = dataset[idx]
                    
                    if transform then
                        data = transform(data)
                    end
                    
                    table.insert(batch_data, data)
                    table.insert(batch_labels, label)
                end
                
                return batch_data, batch_labels
            end,
            function(batch_data, batch_labels)
                table.insert(self.queue, {data = batch_data, labels = batch_labels})
            end,
            indices,
            self.transform
        )
    end
    
    -- Get next batch (blocking)
    function loader:get_batch()
        while #self.queue == 0 do
            -- Wait for batch to be ready
        end
        
        local batch = table.remove(self.queue, 1)
        return batch.data, batch.labels
    end
    
    -- Create iterator
    function loader:iterator()
        local indices = torch.randperm(self.dataset:size())
        local current_idx = 1
        
        return function()
            if current_idx > indices:size(1) then
                return nil
            end
            
            local batch_end = math.min(current_idx + self.batch_size - 1, indices:size(1))
            local batch_indices = {}
            
            for i = current_idx, batch_end do
                table.insert(batch_indices, indices[i])
            end
            
            current_idx = batch_end + 1
            
            self:load_batch_async(batch_indices)
            return self:get_batch()
        end
    end
    
    -- Terminate thread pool
    function loader:terminate()
        if self.pool then
            self.pool:terminate()
            self.pool = nil
        end
    end
    
    return loader:init()
end

-- ============================================================================
-- Parallel Model Ensemble
-- ============================================================================

function threads_nn_bridge.create_model_ensemble(models, strategy)
    strategy = strategy or 'average'  -- average, vote, weighted
    
    local ensemble = {
        models = models,
        strategy = strategy,
        n_models = #models,
    }
    
    function ensemble:forward(input)
        local outputs = {}
        
        -- Forward pass through all models
        for i, model in ipairs(self.models) do
            outputs[i] = model:forward(input)
        end
        
        -- Combine outputs based on strategy
        if self.strategy == 'average' then
            local sum = outputs[1]:clone()
            for i = 2, #outputs do
                sum:add(outputs[i])
            end
            return sum:div(#outputs)
            
        elseif self.strategy == 'vote' then
            -- Majority voting for classification
            local votes = torch.zeros(outputs[1]:size())
            for i = 1, #outputs do
                local _, predicted = outputs[i]:max(2)
                for j = 1, predicted:size(1) do
                    votes[j][predicted[j][1]] = votes[j][predicted[j][1]] + 1
                end
            end
            return votes
            
        elseif self.strategy == 'weighted' and self.weights then
            local sum = outputs[1]:clone():mul(self.weights[1])
            for i = 2, #outputs do
                sum:add(self.weights[i], outputs[i])
            end
            return sum
        end
        
        return outputs[1]
    end
    
    function ensemble:evaluate()
        for _, model in ipairs(self.models) do
            model:evaluate()
        end
    end
    
    function ensemble:training()
        for _, model in ipairs(self.models) do
            model:training()
        end
    end
    
    return ensemble
end

-- ============================================================================
-- Parallel Gradient Computation
-- ============================================================================

function threads_nn_bridge.create_parallel_trainer(model, criterion, n_replicas)
    n_replicas = n_replicas or 2
    
    local trainer = {
        master_model = model,
        criterion = criterion,
        n_replicas = n_replicas,
        model_replicas = {},
        pool = nil,
    }
    
    function trainer:init()
        -- Create model replicas
        for i = 1, self.n_replicas do
            self.model_replicas[i] = self.master_model:clone()
        end
        
        -- Initialize thread pool
        self.pool = threads.Threads(
            self.n_replicas,
            function()
                require 'torch'
                require 'nn'
            end
        )
        
        torch7u.utils.log('INFO', 
            string.format('Parallel trainer initialized with %d replicas', self.n_replicas),
            'threads_nn_bridge')
        
        return self
    end
    
    function trainer:train_parallel(mini_batches)
        local gradients = {}
        local losses = {}
        
        -- Distribute work across threads
        for i, mini_batch in ipairs(mini_batches) do
            local replica_idx = ((i - 1) % self.n_replicas) + 1
            local model = self.model_replicas[replica_idx]
            
            self.pool:addjob(
                function(model, criterion, input, target)
                    -- Forward pass
                    local output = model:forward(input)
                    local loss = criterion:forward(output, target)
                    
                    -- Backward pass
                    local gradOutput = criterion:backward(output, target)
                    model:zeroGradParameters()
                    model:backward(input, gradOutput)
                    
                    -- Get gradients
                    local _, grad = model:parameters()
                    
                    return grad, loss
                end,
                function(grad, loss)
                    table.insert(gradients, grad)
                    table.insert(losses, loss)
                end,
                model,
                self.criterion,
                mini_batch.input,
                mini_batch.target
            )
        end
        
        -- Wait for all jobs to complete
        self.pool:synchronize()
        
        -- Average gradients
        local avg_gradients = {}
        if #gradients > 0 then
            for i, grad_param in ipairs(gradients[1]) do
                avg_gradients[i] = grad_param:clone()
                for j = 2, #gradients do
                    avg_gradients[i]:add(gradients[j][i])
                end
                avg_gradients[i]:div(#gradients)
            end
        end
        
        -- Average loss
        local avg_loss = 0
        for _, loss in ipairs(losses) do
            avg_loss = avg_loss + loss
        end
        avg_loss = avg_loss / #losses
        
        return avg_gradients, avg_loss
    end
    
    function trainer:synchronize_replicas()
        local master_params = self.master_model:parameters()
        
        for i = 1, self.n_replicas do
            local replica_params = self.model_replicas[i]:parameters()
            for j = 1, #master_params do
                replica_params[j]:copy(master_params[j])
            end
        end
    end
    
    function trainer:terminate()
        if self.pool then
            self.pool:terminate()
            self.pool = nil
        end
    end
    
    return trainer:init()
end

-- ============================================================================
-- Asynchronous Model Evaluation
-- ============================================================================

function threads_nn_bridge.create_async_evaluator(model, n_threads)
    n_threads = n_threads or 4
    
    local evaluator = {
        model = model,
        n_threads = n_threads,
        pool = nil,
        results = {},
    }
    
    function evaluator:init()
        self.pool = threads.Threads(
            self.n_threads,
            function()
                require 'torch'
                require 'nn'
            end
        )
        return self
    end
    
    function evaluator:evaluate_async(test_data, callback)
        local batch_size = math.ceil(#test_data / self.n_threads)
        
        for thread_id = 1, self.n_threads do
            local start_idx = (thread_id - 1) * batch_size + 1
            local end_idx = math.min(thread_id * batch_size, #test_data)
            
            if start_idx <= #test_data then
                local batch = {}
                for i = start_idx, end_idx do
                    table.insert(batch, test_data[i])
                end
                
                self.pool:addjob(
                    function(model, batch)
                        local predictions = {}
                        model:evaluate()
                        
                        for _, sample in ipairs(batch) do
                            local output = model:forward(sample.input)
                            table.insert(predictions, {
                                output = output,
                                target = sample.target
                            })
                        end
                        
                        return predictions
                    end,
                    function(predictions)
                        for _, pred in ipairs(predictions) do
                            table.insert(self.results, pred)
                        end
                        
                        if callback then
                            callback(predictions)
                        end
                    end,
                    self.model:clone(),
                    batch
                )
            end
        end
        
        -- Wait for completion
        self.pool:synchronize()
        
        return self.results
    end
    
    function evaluator:get_accuracy()
        local correct = 0
        local total = 0
        
        for _, result in ipairs(self.results) do
            local _, predicted = result.output:max(1)
            local _, actual = result.target:max(1)
            
            if predicted[1] == actual[1] then
                correct = correct + 1
            end
            total = total + 1
        end
        
        return correct / total
    end
    
    function evaluator:terminate()
        if self.pool then
            self.pool:terminate()
            self.pool = nil
        end
        self.results = {}
    end
    
    return evaluator:init()
end

-- ============================================================================
-- Parallel Data Preprocessing
-- ============================================================================

function threads_nn_bridge.create_parallel_preprocessor(transform_fn, n_threads)
    n_threads = n_threads or 4
    
    local preprocessor = {
        transform_fn = transform_fn,
        n_threads = n_threads,
        pool = nil,
    }
    
    function preprocessor:init()
        self.pool = threads.Threads(
            self.n_threads,
            function()
                require 'torch'
            end
        )
        return self
    end
    
    function preprocessor:process_batch(data)
        local results = {}
        local batch_size = math.ceil(#data / self.n_threads)
        
        for thread_id = 1, self.n_threads do
            local start_idx = (thread_id - 1) * batch_size + 1
            local end_idx = math.min(thread_id * batch_size, #data)
            
            if start_idx <= #data then
                local batch = {}
                for i = start_idx, end_idx do
                    table.insert(batch, data[i])
                end
                
                self.pool:addjob(
                    function(batch, transform_fn)
                        local processed = {}
                        for _, item in ipairs(batch) do
                            table.insert(processed, transform_fn(item))
                        end
                        return processed
                    end,
                    function(processed)
                        for _, item in ipairs(processed) do
                            table.insert(results, item)
                        end
                    end,
                    batch,
                    self.transform_fn
                )
            end
        end
        
        self.pool:synchronize()
        return results
    end
    
    function preprocessor:terminate()
        if self.pool then
            self.pool:terminate()
            self.pool = nil
        end
    end
    
    return preprocessor:init()
end

-- ============================================================================
-- Register Integration
-- ============================================================================

torch7u.register('threads_nn_bridge', threads_nn_bridge, {'threads', 'nn'})

torch7u.utils.log('INFO', 'threads-nn integration bridge loaded', 'threads_nn_bridge')

return threads_nn_bridge
