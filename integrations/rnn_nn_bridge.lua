-- ============================================================================
-- Integration Bridge: rnn <-> nn
-- ============================================================================
-- This module provides enhanced integration between standard neural networks
-- and recurrent neural networks, enabling seamless composition.
-- ============================================================================

local rnn_nn_bridge = {}

-- Ensure required modules are loaded
if not torch7u then
    error("torch7u integration layer not loaded. Please require 'init' first.")
end

local rnn_available, rnn = pcall(require, 'rnn')
local nn = torch7u.load_module('nn')

if not rnn_available or not nn then
    print("Warning: rnn or nn not available, bridge not fully initialized")
    return rnn_nn_bridge
end

-- ============================================================================
-- Sequence Model Builder
-- ============================================================================

function rnn_nn_bridge.create_sequence_model(config)
    config = config or {}
    
    local model_type = config.type or 'lstm'  -- lstm, gru, rnn
    local input_size = config.input_size or 100
    local hidden_size = config.hidden_size or 128
    local num_layers = config.num_layers or 1
    local output_size = config.output_size or 10
    local dropout = config.dropout or 0.0
    local bidirectional = config.bidirectional or false
    
    local model = nn.Sequential()
    
    -- Add recurrent layers
    for layer = 1, num_layers do
        local layer_input_size = (layer == 1) and input_size or hidden_size
        
        if model_type == 'lstm' then
            if bidirectional then
                model:add(rnn.BiSequencer(rnn.SeqLSTM(layer_input_size, hidden_size)))
            else
                model:add(rnn.Sequencer(rnn.LSTM(layer_input_size, hidden_size)))
            end
        elseif model_type == 'gru' then
            if bidirectional then
                model:add(rnn.BiSequencer(rnn.SeqGRU(layer_input_size, hidden_size)))
            else
                model:add(rnn.Sequencer(rnn.GRU(layer_input_size, hidden_size)))
            end
        elseif model_type == 'rnn' then
            model:add(rnn.Sequencer(nn.Linear(layer_input_size, hidden_size)))
            model:add(rnn.Sequencer(nn.Tanh()))
        end
        
        -- Add dropout if specified
        if dropout > 0 and layer < num_layers then
            model:add(rnn.Sequencer(nn.Dropout(dropout)))
        end
    end
    
    -- Add output layer
    model:add(rnn.Sequencer(nn.Linear(hidden_size, output_size)))
    
    if config.output_activation then
        model:add(rnn.Sequencer(config.output_activation))
    end
    
    return model
end

-- ============================================================================
-- Sequence-to-Sequence Model
-- ============================================================================

function rnn_nn_bridge.create_seq2seq_model(encoder_config, decoder_config)
    encoder_config = encoder_config or {}
    decoder_config = decoder_config or {}
    
    local seq2seq = {
        encoder = nil,
        decoder = nil,
        encoder_hidden = nil,
    }
    
    -- Create encoder
    seq2seq.encoder = rnn_nn_bridge.create_sequence_model(encoder_config)
    
    -- Create decoder
    seq2seq.decoder = rnn_nn_bridge.create_sequence_model(decoder_config)
    
    function seq2seq:forward(source_seq, target_seq)
        -- Encode
        self.encoder_hidden = self.encoder:forward(source_seq)
        
        -- Get final hidden state
        local final_hidden = self.encoder_hidden[#self.encoder_hidden]
        
        -- Initialize decoder with encoder's final state
        -- (This is simplified; real implementation would handle LSTM states properly)
        
        -- Decode
        local decoder_output = self.decoder:forward(target_seq)
        
        return decoder_output
    end
    
    function seq2seq:backward(source_seq, target_seq, grad_output)
        -- Backward through decoder
        local grad_decoder = self.decoder:backward(target_seq, grad_output)
        
        -- Backward through encoder
        local grad_encoder = self.encoder:backward(source_seq, grad_decoder)
        
        return grad_encoder
    end
    
    function seq2seq:parameters()
        local enc_params, enc_grad = self.encoder:parameters()
        local dec_params, dec_grad = self.decoder:parameters()
        
        local params = {}
        local grad_params = {}
        
        for _, p in ipairs(enc_params) do
            table.insert(params, p)
        end
        for _, p in ipairs(dec_params) do
            table.insert(params, p)
        end
        
        for _, g in ipairs(enc_grad) do
            table.insert(grad_params, g)
        end
        for _, g in ipairs(dec_grad) do
            table.insert(grad_params, g)
        end
        
        return params, grad_params
    end
    
    return seq2seq
end

-- ============================================================================
-- Attention Mechanism
-- ============================================================================

function rnn_nn_bridge.create_attention_layer(hidden_size, attention_type)
    attention_type = attention_type or 'dot'  -- dot, general, concat
    
    local attention = nn.Sequential()
    
    if attention_type == 'dot' then
        -- Dot product attention
        attention:add(nn.MM())  -- Matrix multiplication
        attention:add(nn.SoftMax())
        
    elseif attention_type == 'general' then
        -- General attention with learned weight matrix
        attention:add(nn.Linear(hidden_size, hidden_size))
        attention:add(nn.MM())
        attention:add(nn.SoftMax())
        
    elseif attention_type == 'concat' then
        -- Additive/concat attention
        local concat = nn.Sequential()
        concat:add(nn.JoinTable(2))
        concat:add(nn.Linear(hidden_size * 2, hidden_size))
        concat:add(nn.Tanh())
        concat:add(nn.Linear(hidden_size, 1))
        concat:add(nn.SoftMax())
        attention = concat
    end
    
    return attention
end

-- ============================================================================
-- Recurrent Regularization
-- ============================================================================

function rnn_nn_bridge.add_recurrent_regularization(model, config)
    config = config or {}
    
    local dropout = config.dropout or 0.0
    local weight_decay = config.weight_decay or 0.0
    local gradient_clip = config.gradient_clip
    
    -- Add dropout to recurrent connections
    if dropout > 0 then
        -- Find all recurrent modules and add dropout
        local function add_dropout_to_module(m)
            if torch.type(m) == 'rnn.LSTM' or 
               torch.type(m) == 'rnn.GRU' or
               torch.type(m) == 'rnn.FastLSTM' then
                -- Set dropout on recurrent module
                -- (Implementation depends on specific RNN module)
            end
        end
        
        model:apply(add_dropout_to_module)
    end
    
    -- Add gradient clipping
    if gradient_clip then
        local old_backward = model.backward
        model.backward = function(self, input, gradOutput)
            local gradInput = old_backward(self, input, gradOutput)
            
            -- Clip gradients
            local params, gradParams = self:parameters()
            for _, gradParam in ipairs(gradParams) do
                gradParam:clamp(-gradient_clip, gradient_clip)
            end
            
            return gradInput
        end
    end
    
    return model
end

-- ============================================================================
-- Sequence Data Utilities
-- ============================================================================

function rnn_nn_bridge.prepare_sequence_batch(sequences, padding_value)
    padding_value = padding_value or 0
    
    -- Find max length
    local max_len = 0
    for _, seq in ipairs(sequences) do
        max_len = math.max(max_len, seq:size(1))
    end
    
    -- Create batch tensor
    local batch_size = #sequences
    local feature_size = sequences[1]:size(2)
    local batch = torch.Tensor(batch_size, max_len, feature_size):fill(padding_value)
    local lengths = torch.LongTensor(batch_size)
    
    -- Fill batch
    for i, seq in ipairs(sequences) do
        local seq_len = seq:size(1)
        batch[i]:narrow(1, 1, seq_len):copy(seq)
        lengths[i] = seq_len
    end
    
    return batch, lengths
end

function rnn_nn_bridge.create_sequence_dataset(data, seq_length, overlap)
    overlap = overlap or 0
    
    local dataset = {
        data = data,
        seq_length = seq_length,
        overlap = overlap,
        sequences = {},
    }
    
    -- Create sequences
    local step = seq_length - overlap
    for i = 1, data:size(1) - seq_length + 1, step do
        local seq = data:narrow(1, i, seq_length)
        table.insert(dataset.sequences, seq)
    end
    
    function dataset:size()
        return #self.sequences
    end
    
    function dataset:get(idx)
        return self.sequences[idx]
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
-- Sequence Evaluation Metrics
-- ============================================================================

function rnn_nn_bridge.sequence_accuracy(predictions, targets, mask)
    local correct = 0
    local total = 0
    
    for t = 1, predictions:size(1) do
        local _, pred_class = predictions[t]:max(2)
        local _, true_class = targets[t]:max(2)
        
        for b = 1, predictions:size(2) do
            if not mask or mask[t][b] == 1 then
                if pred_class[b][1] == true_class[b][1] then
                    correct = correct + 1
                end
                total = total + 1
            end
        end
    end
    
    return correct / total
end

function rnn_nn_bridge.perplexity(predictions, targets, criterion)
    criterion = criterion or nn.ClassNLLCriterion()
    
    local total_loss = 0
    local total_tokens = 0
    
    for t = 1, predictions:size(1) do
        local loss = criterion:forward(predictions[t], targets[t])
        total_loss = total_loss + loss
        total_tokens = total_tokens + targets:size(2)
    end
    
    local avg_loss = total_loss / total_tokens
    return math.exp(avg_loss)
end

-- ============================================================================
-- Hybrid Models (RNN + CNN)
-- ============================================================================

function rnn_nn_bridge.create_cnn_rnn_model(cnn_config, rnn_config)
    local model = nn.Sequential()
    
    -- CNN feature extractor
    local cnn = nn.Sequential()
    for _, layer_config in ipairs(cnn_config) do
        if layer_config.type == 'conv' then
            cnn:add(nn.SpatialConvolution(
                layer_config.input_channels,
                layer_config.output_channels,
                layer_config.kernel_size,
                layer_config.kernel_size
            ))
        elseif layer_config.type == 'pool' then
            cnn:add(nn.SpatialMaxPooling(
                layer_config.kernel_size,
                layer_config.kernel_size,
                layer_config.stride,
                layer_config.stride
            ))
        elseif layer_config.type == 'relu' then
            cnn:add(nn.ReLU())
        end
    end
    
    model:add(cnn)
    
    -- Reshape for RNN
    model:add(nn.View(-1):setNumInputDims(3))
    
    -- RNN sequence processor
    local rnn_model = rnn_nn_bridge.create_sequence_model(rnn_config)
    model:add(rnn_model)
    
    return model
end

-- ============================================================================
-- Sequence Generation
-- ============================================================================

function rnn_nn_bridge.generate_sequence(model, start_token, max_length, temperature, end_token)
    temperature = temperature or 1.0
    max_length = max_length or 50
    end_token = end_token or nil
    
    model:evaluate()
    
    local sequence = {start_token}
    local current_input = start_token
    
    for i = 1, max_length do
        local output = model:forward(current_input)
        
        -- Sample from output distribution
        local probs = output:div(temperature):exp()
        probs:div(probs:sum())
        
        local next_token = torch.multinomial(probs, 1)[1]
        table.insert(sequence, next_token)
        
        -- Check for end token (if applicable)
        if end_token and next_token == end_token then
            break
        end
        
        current_input = next_token
    end
    
    return sequence
end

-- ============================================================================
-- Register Integration
-- ============================================================================

torch7u.register('rnn_nn_bridge', rnn_nn_bridge, {'rnn', 'nn'})

torch7u.utils.log('INFO', 'rnn-nn integration bridge loaded', 'rnn_nn_bridge')

return rnn_nn_bridge
