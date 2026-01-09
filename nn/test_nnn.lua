-- Tests for Nested Neural Nets (NNN) modules
-- Tests NestedEmbedding, NestedTensor, PrimeFactorType, and NestedNeuralNet

require 'nn'

local mytester = torch.Tester()
local nnntest = torch.TestSuite()

-- Test PrimeFactorType
function nnntest.PrimeFactorType_factorize()
   local PrimeFactorType = require('nn.PrimeFactorType')
   
   -- Test basic factorizations
   local factors = PrimeFactorType.factorize(12)
   mytester:eq(#factors, 3, 'factorize(12) should have 3 factors')
   mytester:eq(factors[1], 2, 'first factor of 12')
   mytester:eq(factors[2], 2, 'second factor of 12')
   mytester:eq(factors[3], 3, 'third factor of 12')
   
   -- Test prime number
   factors = PrimeFactorType.factorize(13)
   mytester:eq(#factors, 1, 'factorize(13) should have 1 factor')
   mytester:eq(factors[1], 13, 'factor of prime 13')
   
   -- Test power of 2
   factors = PrimeFactorType.factorize(16)
   mytester:eq(#factors, 4, 'factorize(16) should have 4 factors')
   for i = 1, 4 do
      mytester:eq(factors[i], 2, 'all factors of 16 are 2')
   end
end

function nnntest.PrimeFactorType_tensorSignature()
   local PrimeFactorType = require('nn.PrimeFactorType')
   
   -- Create tensor with specific shape
   local tensor = torch.Tensor(2, 3, 5)
   local signature = PrimeFactorType.getTensorSignature(tensor)
   
   mytester:eq(#signature, 3, 'signature should have 3 dimensions')
   mytester:eq(#signature[1], 1, 'dim 1: size 2 has 1 prime factor')
   mytester:eq(signature[1][1], 2, 'dim 1: factor is 2')
   mytester:eq(#signature[2], 1, 'dim 2: size 3 has 1 prime factor')
   mytester:eq(signature[2][1], 3, 'dim 2: factor is 3')
   mytester:eq(#signature[3], 1, 'dim 3: size 5 has 1 prime factor')
   mytester:eq(signature[3][1], 5, 'dim 3: factor is 5')
end

function nnntest.PrimeFactorType_metagraphType()
   local PrimeFactorType = require('nn.PrimeFactorType')
   
   local tensor = torch.Tensor(4, 6)
   local metagraphType = PrimeFactorType.getMetagraphType(tensor)
   
   mytester:assert(metagraphType.typeId ~= nil, 'should have type ID')
   mytester:eq(#metagraphType.shape, 2, 'should have 2 dimensions')
   mytester:eq(metagraphType.shape[1], 4, 'first dimension is 4')
   mytester:eq(metagraphType.shape[2], 6, 'second dimension is 6')
   mytester:eq(metagraphType.totalElements, 24, 'total elements is 24')
   mytester:eq(#metagraphType.dimensionalEmbeddings, 2, 'should have 2 dimensional embeddings')
end

function nnntest.PrimeFactorType_compatibility()
   local PrimeFactorType = require('nn.PrimeFactorType')
   
   local tensor1 = torch.Tensor(4, 6)
   local tensor2 = torch.Tensor(4, 6)
   local tensor3 = torch.Tensor(4, 5)
   
   mytester:assert(PrimeFactorType.isCompatible(tensor1, tensor2), 
      'tensors with same shape should be compatible')
   mytester:assert(not PrimeFactorType.isCompatible(tensor1, tensor3), 
      'tensors with different shape should not be compatible')
end

-- Test NestedTensor utilities
function nnntest.NestedTensor_depth()
   local NestedTensor = require('nn.NestedTensor')
   
   local tensor = torch.Tensor(2, 3)
   mytester:eq(NestedTensor.depth(tensor), 0, 'tensor has depth 0')
   
   local nested1 = {tensor, tensor}
   mytester:eq(NestedTensor.depth(nested1), 1, 'single nesting has depth 1')
   
   local nested2 = {nested1, nested1}
   mytester:eq(NestedTensor.depth(nested2), 2, 'double nesting has depth 2')
   
   local nested3 = {{tensor}, {tensor, {tensor}}}
   mytester:eq(NestedTensor.depth(nested3), 2, 'irregular nesting depth is max')
end

function nnntest.NestedTensor_clone()
   local NestedTensor = require('nn.NestedTensor')
   
   local tensor = torch.Tensor(2, 3):fill(1)
   local nested = {tensor, {tensor:clone():fill(2), tensor:clone():fill(3)}}
   
   local cloned = NestedTensor.clone(nested)
   
   -- Modify original
   tensor:fill(999)
   
   -- Check clone is independent
   mytester:assert(cloned[1][1] ~= 999, 'clone should be independent')
end

function nnntest.NestedTensor_operations()
   local NestedTensor = require('nn.NestedTensor')
   
   local t1 = torch.Tensor(2, 2):fill(1)
   local t2 = torch.Tensor(2, 2):fill(2)
   local nested = {t1:clone(), {t1:clone(), t2:clone()}}
   
   -- Test fill
   NestedTensor.fill(nested, 5)
   mytester:eq(nested[1][1][1], 5, 'fill should set all elements')
   mytester:eq(nested[2][2][1][1], 5, 'fill should work at all depths')
   
   -- Test count
   local count = NestedTensor.count(nested)
   mytester:eq(count, 3, 'should count 3 tensors')
   
   -- Test flatten
   local flat = NestedTensor.flatten(nested)
   mytester:eq(#flat, 3, 'flattened should have 3 tensors')
end

-- Test NestedEmbedding
function nnntest.NestedEmbedding_basic()
   local nIndex = 10
   local nOutput = 5
   local maxDepth = 3
   
   local module = nn.NestedEmbedding(nIndex, nOutput, maxDepth)
   
   mytester:eq(module.nIndex, nIndex, 'nIndex should be set')
   mytester:eq(module.nOutput, nOutput, 'nOutput should be set')
   mytester:eq(module.maxDepth, maxDepth, 'maxDepth should be set')
   mytester:eq(#module.weights, maxDepth, 'should have weights for each depth')
   
   -- Check weight sizes
   for depth = 1, maxDepth do
      mytester:eq(module.weights[depth]:size(1), nIndex, 'weight dim 1')
      mytester:eq(module.weights[depth]:size(2), nOutput, 'weight dim 2')
   end
end

function nnntest.NestedEmbedding_forward()
   local nIndex = 10
   local nOutput = 5
   local module = nn.NestedEmbedding(nIndex, nOutput, 2)
   
   -- Test simple tensor input
   local input = torch.LongTensor({1, 2, 3})
   local output = module:forward(input)
   
   mytester:eq(output:size(1), 3, 'output should have 3 embeddings')
   mytester:eq(output:size(2), nOutput, 'output should have correct dimension')
   
   -- Test nested input
   local nestedInput = {
      torch.LongTensor({1, 2}),
      torch.LongTensor({3, 4})
   }
   local nestedOutput = module:forward(nestedInput)
   
   mytester:assert(type(nestedOutput) == 'table', 'output should be table for nested input')
   mytester:eq(#nestedOutput, 2, 'output should have 2 elements')
end

function nnntest.NestedEmbedding_backward()
   local nIndex = 10
   local nOutput = 5
   local module = nn.NestedEmbedding(nIndex, nOutput, 2)
   
   local input = torch.LongTensor({1, 2, 3})
   local output = module:forward(input)
   
   local gradOutput = torch.Tensor(output:size()):fill(1)
   local gradInput = module:backward(input, gradOutput)
   
   mytester:assert(gradInput ~= nil, 'gradInput should not be nil')
   mytester:assert(gradInput:isSameSizeAs(input), 'gradInput should match input size')
end

-- Test NestedNeuralNet
function nnntest.NestedNeuralNet_creation()
   local nnn = nn.NestedNeuralNet({maxDepth = 3})
   
   mytester:eq(nnn.maxDepth, 3, 'maxDepth should be set')
   mytester:assert(nnn.useTypeSystem, 'type system should be enabled by default')
end

function nnntest.NestedNeuralNet_addModules()
   local nnn = nn.NestedNeuralNet({maxDepth = 2})
   
   -- Add embedder
   local embedder = nn.LookupTable(10, 5)
   nnn:addEmbedder(1, embedder)
   
   mytester:assert(nnn.embedders[1] ~= nil, 'embedder should be added')
   mytester:eq(nnn.embedders[1], embedder, 'correct embedder should be stored')
   
   -- Add processor
   local processor = nn.Linear(5, 3)
   nnn:addProcessor(1, processor)
   
   mytester:assert(nnn.processors[1] ~= nil, 'processor should be added')
   mytester:eq(nnn.processors[1], processor, 'correct processor should be stored')
end

function nnntest.NestedNeuralNet_forward()
   local nnn = nn.NestedNeuralNet({maxDepth = 2})
   
   -- Add simple embedder at depth 1
   local embedder = nn.LookupTable(10, 5)
   nnn:addEmbedder(1, embedder)
   
   -- Test with tensor input
   local input = torch.LongTensor({1, 2, 3})
   local output = nnn:forward(input)
   
   mytester:assert(torch.isTensor(output), 'output should be tensor')
   mytester:eq(output:size(1), 3, 'output size 1')
   mytester:eq(output:size(2), 5, 'output size 2')
end

function nnntest.NestedNeuralNet_analyzeStructure()
   local nnn = nn.NestedNeuralNet({maxDepth = 3})
   
   local input = {
      torch.Tensor(2, 3),
      {torch.Tensor(4, 5)}
   }
   
   local structure = nnn:analyzeStructure(input)
   
   mytester:assert(structure ~= nil, 'structure should not be nil')
   mytester:assert(structure.isNested, 'structure should be nested')
   mytester:eq(structure.depth, 0, 'root should be at depth 0')
end

function nnntest.NestedNeuralNet_createSimple()
   local nIndex = 10
   local nOutput = 5
   local maxDepth = 3
   
   local nnn = nn.NestedNeuralNet.createSimple(nIndex, nOutput, maxDepth)
   
   mytester:eq(nnn.maxDepth, maxDepth, 'maxDepth should be set')
   mytester:eq(#nnn.embedders, maxDepth, 'should have embedders for each depth')
   
   -- Check all embedders are present
   for depth = 1, maxDepth do
      mytester:assert(nnn.embedders[depth] ~= nil, 
         string.format('embedder at depth %d should exist', depth))
   end
end

-- Integration test
function nnntest.NestedNeuralNet_integration()
   -- Create a simple nested neural net
   local nnn = nn.NestedNeuralNet.createSimple(20, 8, 2)
   
   -- Create nested input structure
   local input = {
      torch.LongTensor({1, 2, 3}),
      torch.LongTensor({4, 5})
   }
   
   -- Forward pass
   local output = nnn:forward(input)
   
   mytester:assert(type(output) == 'table', 'output should be table')
   mytester:eq(#output, 2, 'output should have 2 elements')
   mytester:assert(torch.isTensor(output[1]), 'first element should be tensor')
   mytester:assert(torch.isTensor(output[2]), 'second element should be tensor')
   
   -- Check dimensions
   mytester:eq(output[1]:size(1), 3, 'first output size')
   mytester:eq(output[2]:size(1), 2, 'second output size')
   
   -- Backward pass
   local gradOutput = {
      torch.Tensor(output[1]:size()):fill(1),
      torch.Tensor(output[2]:size()):fill(1)
   }
   local gradInput = nnn:backward(input, gradOutput)
   
   mytester:assert(type(gradInput) == 'table', 'gradInput should be table')
   mytester:eq(#gradInput, 2, 'gradInput should have 2 elements')
end

-- Run tests
mytester:add(nnntest)
mytester:run()

return nnntest
