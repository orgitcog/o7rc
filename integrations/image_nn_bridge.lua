-- ============================================================================
-- Integration Bridge: image <-> nn
-- ============================================================================
-- This module provides seamless integration between image processing and
-- neural network modules for vision tasks.
-- ============================================================================

local image_nn_bridge = {}

-- Ensure required modules are loaded
if not torch7u then
    error("torch7u integration layer not loaded. Please require 'init' first.")
end

local image = torch7u.load_module('image')
local nn = torch7u.load_module('nn')

if not image or not nn then
    print("Warning: image or nn not available, bridge not fully initialized")
    return image_nn_bridge
end

-- ============================================================================
-- Image Preprocessing Pipeline for Neural Networks
-- ============================================================================

function image_nn_bridge.create_preprocessing_pipeline(transforms)
    transforms = transforms or {}
    
    local pipeline = {
        transforms = transforms,
        stats = nil,  -- For normalization statistics
    }
    
    function pipeline:add_transform(transform)
        table.insert(self.transforms, transform)
        return self
    end
    
    function pipeline:process(img)
        local result = img:clone()
        
        for _, transform in ipairs(self.transforms) do
            if transform.type == 'resize' then
                result = image.scale(result, transform.width, transform.height, transform.mode or 'bilinear')
                
            elseif transform.type == 'crop' then
                result = image.crop(result, transform.x1, transform.y1, transform.x2, transform.y2)
                
            elseif transform.type == 'center_crop' then
                result = image.crop(result, 'c', transform.width, transform.height)
                
            elseif transform.type == 'normalize' then
                local mean = transform.mean or self.stats.mean
                local std = transform.std or self.stats.std
                
                if mean and std then
                    for c = 1, result:size(1) do
                        result[c]:add(-mean[c]):div(std[c])
                    end
                end
                
            elseif transform.type == 'rgb2yuv' then
                result = image.rgb2yuv(result)
                
            elseif transform.type == 'rgb2hsv' then
                result = image.rgb2hsv(result)
                
            elseif transform.type == 'hflip' and transform.prob and math.random() < transform.prob then
                result = image.hflip(result)
                
            elseif transform.type == 'vflip' and transform.prob and math.random() < transform.prob then
                result = image.vflip(result)
                
            elseif transform.type == 'rotate' and transform.angle then
                result = image.rotate(result, transform.angle)
                
            elseif transform.type == 'custom' and transform.fn then
                result = transform.fn(result)
            end
        end
        
        return result
    end
    
    function pipeline:compute_stats(dataset, n_samples)
        n_samples = n_samples or math.min(1000, #dataset)
        
        local mean = torch.zeros(3)
        local std = torch.zeros(3)
        local count = 0
        
        for i = 1, n_samples do
            local img = dataset[i]
            for c = 1, math.min(3, img:size(1)) do
                mean[c] = mean[c] + img[c]:mean()
                std[c] = std[c] + img[c]:std()
            end
            count = count + 1
        end
        
        mean:div(count)
        std:div(count)
        
        self.stats = {mean = mean, std = std}
        return mean, std
    end
    
    return pipeline
end

-- ============================================================================
-- Image Data Augmentation for Training
-- ============================================================================

function image_nn_bridge.create_augmenter(config)
    config = config or {}
    
    local augmenter = {
        hflip_prob = config.hflip_prob or 0.5,
        vflip_prob = config.vflip_prob or 0.0,
        rotation_range = config.rotation_range or {-10, 10},
        scale_range = config.scale_range or {0.9, 1.1},
        crop_size = config.crop_size,
        color_jitter = config.color_jitter or 0.0,
    }
    
    function augmenter:augment(img)
        local result = img:clone()
        
        -- Horizontal flip
        if math.random() < self.hflip_prob then
            result = image.hflip(result)
        end
        
        -- Vertical flip
        if math.random() < self.vflip_prob then
            result = image.vflip(result)
        end
        
        -- Random rotation
        if self.rotation_range then
            local angle = (math.random() * (self.rotation_range[2] - self.rotation_range[1]) + self.rotation_range[1]) * math.pi / 180
            result = image.rotate(result, angle)
        end
        
        -- Random scale
        if self.scale_range then
            local scale = math.random() * (self.scale_range[2] - self.scale_range[1]) + self.scale_range[1]
            local new_h = math.floor(result:size(2) * scale)
            local new_w = math.floor(result:size(3) * scale)
            result = image.scale(result, new_w, new_h)
        end
        
        -- Random crop
        if self.crop_size then
            result = image.crop(result, 'c', self.crop_size, self.crop_size)
        end
        
        -- Color jitter
        if self.color_jitter > 0 then
            local jitter = torch.randn(result:size(1)) * self.color_jitter
            for c = 1, result:size(1) do
                result[c]:add(jitter[c])
            end
        end
        
        return result
    end
    
    return augmenter
end

-- ============================================================================
-- Convolutional Layer Visualizations
-- ============================================================================

function image_nn_bridge.visualize_filters(layer, output_file)
    if torch.type(layer) ~= 'nn.SpatialConvolution' then
        error("Layer must be a SpatialConvolution")
    end
    
    local filters = layer.weight
    local n_filters = filters:size(1)
    local n_channels = filters:size(2)
    local h = filters:size(3)
    local w = filters:size(4)
    
    -- Normalize filters for visualization
    local vis_filters = filters:clone()
    for i = 1, n_filters do
        vis_filters[i] = image.minmax{tensor = vis_filters[i], min = 0, max = 1}
    end
    
    -- Create grid of filters
    local grid = image.toDisplayTensor{
        input = vis_filters:view(n_filters, n_channels, h, w),
        padding = 1,
        nrow = math.ceil(math.sqrt(n_filters)),
        scaleeach = true
    }
    
    if output_file then
        image.save(output_file, grid)
    end
    
    return grid
end

function image_nn_bridge.visualize_activations(model, input_image, layer_idx, output_file)
    -- Forward pass up to specific layer
    local activations
    
    if layer_idx then
        local partial_model = nn.Sequential()
        for i = 1, layer_idx do
            partial_model:add(model:get(i))
        end
        activations = partial_model:forward(input_image)
    else
        activations = model:forward(input_image)
    end
    
    -- Visualize activations
    if activations:nDimension() == 3 then
        local grid = image.toDisplayTensor{
            input = activations,
            padding = 1,
            nrow = math.ceil(math.sqrt(activations:size(1))),
            scaleeach = true
        }
        
        if output_file then
            image.save(output_file, grid)
        end
        
        return grid
    end
    
    return activations
end

-- ============================================================================
-- Batch Processing for Images
-- ============================================================================

function image_nn_bridge.create_batch_processor(preprocess_fn, batch_size)
    batch_size = batch_size or 32
    
    local processor = {
        preprocess_fn = preprocess_fn,
        batch_size = batch_size,
    }
    
    function processor:process_directory(directory, pattern)
        pattern = pattern or '*.jpg'
        
        -- Check if paths module is available
        if not paths then
            torch7u.utils.log('ERROR', 'paths module not available', 'image_nn_bridge')
            return {}
        end
        
        local files = paths.dir(directory)
        
        local images = {}
        for _, file in ipairs(files) do
            if file:match(pattern) then
                local img_path = paths.concat(directory, file)
                local img = image.load(img_path)
                if self.preprocess_fn then
                    img = self.preprocess_fn(img)
                end
                table.insert(images, img)
            end
        end
        
        return images
    end
    
    function processor:create_batches(images)
        local batches = {}
        local n_batches = math.ceil(#images / self.batch_size)
        
        for b = 1, n_batches do
            local start_idx = (b - 1) * self.batch_size + 1
            local end_idx = math.min(b * self.batch_size, #images)
            local batch_size_actual = end_idx - start_idx + 1
            
            -- Create batch tensor
            local batch = torch.Tensor(batch_size_actual, unpack(images[1]:size():totable()))
            
            for i = 1, batch_size_actual do
                batch[i]:copy(images[start_idx + i - 1])
            end
            
            table.insert(batches, batch)
        end
        
        return batches
    end
    
    return processor
end

-- ============================================================================
-- Image Dataset Wrapper for nn
-- ============================================================================

function image_nn_bridge.create_image_dataset(image_dir, label_fn, transforms)
    local dataset = {
        image_dir = image_dir,
        label_fn = label_fn,
        transforms = transforms,
        image_paths = {},
        labels = {},
    }
    
    -- Load image paths
    local files = paths.dir(image_dir)
    for _, file in ipairs(files) do
        if file:match('%.jpg$') or file:match('%.png$') then
            local img_path = paths.concat(image_dir, file)
            table.insert(dataset.image_paths, img_path)
            
            if label_fn then
                table.insert(dataset.labels, label_fn(file))
            end
        end
    end
    
    function dataset:size()
        return #self.image_paths
    end
    
    function dataset:get(idx)
        local img = image.load(self.image_paths[idx])
        
        if self.transforms then
            img = self.transforms:process(img)
        end
        
        local label = self.labels[idx]
        
        return img, label
    end
    
    setmetatable(dataset, {
        __index = function(self, idx)
            if type(idx) == 'number' then
                return self:get(idx)
            end
        end,
        __len = function(self)
            return self:size()
        end
    })
    
    return dataset
end

-- ============================================================================
-- Saliency Maps and Gradient Visualization
-- ============================================================================

function image_nn_bridge.compute_saliency_map(model, input_image, target_class)
    model:evaluate()
    
    -- Ensure input requires gradient
    input_image = input_image:clone()
    
    -- Forward pass
    local output = model:forward(input_image)
    
    -- Create gradient signal
    local grad_output = torch.zeros(output:size())
    if target_class then
        grad_output[target_class] = 1
    else
        grad_output[output:max(1)[2]] = 1
    end
    
    -- Backward pass to get gradient w.r.t. input
    model:zeroGradParameters()
    local grad_input = model:backward(input_image, grad_output)
    
    -- Compute saliency as absolute value of gradient
    local saliency = grad_input:abs()
    
    -- If multi-channel, take max across channels
    if saliency:nDimension() == 3 and saliency:size(1) > 1 then
        saliency = saliency:max(1)[1]
    end
    
    -- Normalize for visualization
    saliency = image.minmax{tensor = saliency, min = 0, max = 1}
    
    return saliency
end

-- ============================================================================
-- Register Integration
-- ============================================================================

torch7u.register('image_nn_bridge', image_nn_bridge, {'image', 'nn'})

torch7u.utils.log('INFO', 'image-nn integration bridge loaded', 'image_nn_bridge')

return image_nn_bridge
