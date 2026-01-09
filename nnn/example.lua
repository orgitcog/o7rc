-- Examples for NNN Functional Operator System
-- Demonstrates how nnn.* extends nn.* to work with nested tensors

require 'nn'
local nnn = require 'nnn'

print("=== NNN Functional Operator Examples ===\n")

-- Example 1: Basic Transformation
print("1. Basic Module Transformation")
print("-------------------------------")

-- Create a standard nn.Linear module
local linear = nn.Linear(10, 5)
print("Created nn.Linear(10, 5)")

-- Transform it to work with nested tensors
local nestedLinear = nnn.transform(linear)
print("Transformed to nnn.NestedOperator")

-- Test with single tensor (works as normal)
local singleInput = torch.randn(3, 10)
local singleOutput = nestedLinear:forward(singleInput)
print(string.format("Single tensor input [3x10] -> output [%dx%d]", 
    singleOutput:size(1), singleOutput:size(2)))

-- Test with nested tensor (preserves structure)
local nestedInput = {
    torch.randn(2, 10),
    torch.randn(4, 10)
}
local nestedOutput = nestedLinear:forward(nestedInput)
print(string.format("Nested input {[2x10], [4x10]} -> output {[%dx%d], [%dx%d]}",
    nestedOutput[1]:size(1), nestedOutput[1]:size(2),
    nestedOutput[2]:size(1), nestedOutput[2]:size(2)))

-- Example 2: Pre-built nnn.* Modules
print("\n2. Pre-built nnn.* Modules")
print("---------------------------")

-- Create nnn versions directly
local nnnLinear = nnn.Linear(10, 5)
local nnnReLU = nnn.ReLU()
local nnnTanh = nnn.Tanh()

print("Created nnn.Linear(10, 5)")
print("Created nnn.ReLU()")
print("Created nnn.Tanh()")

-- Example 3: nnn.Sequential
print("\n3. Building Models with nnn.Sequential")
print("---------------------------------------")

local model = nnn.Sequential()
    :add(nnn.Linear(20, 15))
    :add(nnn.ReLU())
    :add(nnn.Linear(15, 10))
    :add(nnn.Tanh())

print("Created nnn.Sequential model:")
print("  Layer 1: nnn.Linear(20, 15)")
print("  Layer 2: nnn.ReLU()")
print("  Layer 3: nnn.Linear(15, 10)")
print("  Layer 4: nnn.Tanh()")

-- Test with nested input
local hierarchicalInput = {
    {
        torch.randn(20),  -- Level 2, branch 1
        torch.randn(20)   -- Level 2, branch 2
    },
    torch.randn(20)       -- Level 1, branch 1
}

print("\nInput structure: {{Tensor[20], Tensor[20]}, Tensor[20]}")
local modelOutput = model:forward(hierarchicalInput)
print("Output preserves the same nested structure")

-- Example 4: Modal Classifier (Criterion)
print("\n4. Modal Classifiers with nnn.Criterion")
print("----------------------------------------")

-- Create nested criterion
local criterion = nnn.MSECriterion()
print("Created nnn.MSECriterion()")

-- Test with nested predictions and targets
local predictions = {
    torch.randn(5),
    torch.randn(5)
}
local targets = {
    torch.randn(5),
    torch.randn(5)
}

local loss = criterion:forward(predictions, targets)
print(string.format("Nested loss (averaged): %.6f", loss))

local gradInput = criterion:backward(predictions, targets)
print("Gradient computed for nested structure")

-- Example 5: Transforming Existing Models
print("\n5. Transforming Existing Models")
print("--------------------------------")

-- Create a standard nn model
local existingModel = nn.Sequential()
    :add(nn.Linear(100, 50))
    :add(nn.Tanh())
    :add(nn.Linear(50, 10))

print("Created existing nn.Sequential model")

-- Transform it to work with nested inputs
local transformedModel = nnn.transform(existingModel)
print("Transformed to nnn.NestedOperator")

-- Now it works with nested inputs
local nestedTestInput = {
    torch.randn(100),
    {torch.randn(100), torch.randn(100)}
}

local transformedOutput = transformedModel:forward(nestedTestInput)
print("Model now works with nested inputs!")
print("Output structure matches input structure")

-- Example 6: Using nnn.fromNN Factory
print("\n6. Dynamic Module Creation with nnn.fromNN")
print("-------------------------------------------")

-- Create nnn versions dynamically
local dynamicSigmoid = nnn.fromNN('Sigmoid')
local dynamicSoftMax = nnn.fromNN('SoftMax')

print("Created nnn.Sigmoid using nnn.fromNN('Sigmoid')")
print("Created nnn.SoftMax using nnn.fromNN('SoftMax')")

-- Example 7: Utility Functions
print("\n7. NNN Utility Functions")
print("------------------------")

local testNested = {
    torch.randn(5),
    {torch.randn(3), torch.randn(4)},
    torch.randn(2)
}

-- Check if nested
local isNested = nnn.isNested(testNested)
print(string.format("Is nested: %s", tostring(isNested)))

-- Get depth
local depth = nnn.depth(testNested)
print(string.format("Nesting depth: %d", depth))

-- Flatten
local flattened = nnn.flatten(testNested)
print(string.format("Flattened: %d tensors", #flattened))

-- Clone
local cloned = nnn.clone(testNested)
print("Cloned nested structure")

-- Map operation
local scaled = nnn.map(testNested, function(t) return t * 2 end)
print("Applied scaling operation to all tensors")

-- Example 8: Integration with Type System
print("\n8. Integration with Metagraph Type System")
print("------------------------------------------")

local PrimeFactorType = nnn.PrimeFactorType

local tensor1 = torch.Tensor(4, 6)
local metagraphType = PrimeFactorType.getMetagraphType(tensor1)

print(string.format("Tensor shape: [4, 6]"))
print(string.format("Type ID: %s", metagraphType.typeId))
print(string.format("Prime factors: [%s]", 
    table.concat({table.concat(metagraphType.signature[1], '.'), 
                  table.concat(metagraphType.signature[2], '.')}, '], [')))

-- Example 9: Hierarchical Document Processing
print("\n9. Hierarchical Document Processing")
print("------------------------------------")

-- Build a document processing model
local docModel = nnn.Sequential()
    :add(nnn.Linear(128, 64))  -- Process embeddings
    :add(nnn.Tanh())
    :add(nnn.Linear(64, 32))

print("Created document processing model")

-- Simulated document structure: paragraphs -> sentences -> word embeddings
local document = {
    -- Paragraph 1
    {
        torch.randn(5, 128),  -- Sentence 1: 5 words x 128 dims
        torch.randn(3, 128)   -- Sentence 2: 3 words x 128 dims
    },
    -- Paragraph 2
    {
        torch.randn(4, 128),  -- Sentence 1: 4 words x 128 dims
        torch.randn(6, 128)   -- Sentence 2: 6 words x 128 dims
    }
}

print("Document structure:")
print("  Paragraph 1: {Sentence[5x128], Sentence[3x128]}")
print("  Paragraph 2: {Sentence[4x128], Sentence[6x128]}")

local docOutput = docModel:forward(document)
print("Processed document (structure preserved)")

-- Example 10: Training Loop with Nested Data
print("\n10. Training Loop with Nested Data")
print("-----------------------------------")

-- Model and criterion
local trainModel = nnn.Sequential()
    :add(nnn.Linear(50, 25))
    :add(nnn.ReLU())
    :add(nnn.Linear(25, 10))

local trainCriterion = nnn.MSECriterion()

print("Created training model and criterion")

-- Training function
local function trainStep(model, criterion, input, target, lr)
    -- Forward
    local output = model:forward(input)
    local loss = criterion:forward(output, target)
    
    -- Backward
    local gradOutput = criterion:backward(output, target)
    model:zeroGradParameters()
    model:backward(input, gradOutput)
    
    -- Update
    model:updateParameters(lr)
    
    return loss
end

-- Nested training data
local trainInput = {
    torch.randn(50),
    {torch.randn(50), torch.randn(50)}
}
local trainTarget = {
    torch.randn(10),
    {torch.randn(10), torch.randn(10)}
}

print("Training with nested input/target...")
local trainingLoss = trainStep(trainModel, trainCriterion, trainInput, trainTarget, 0.01)
print(string.format("Training loss: %.6f", trainingLoss))

-- Example 11: Multi-branch Classification
print("\n11. Multi-branch Classification")
print("--------------------------------")

local classifier = nnn.Sequential()
    :add(nnn.Linear(100, 50))
    :add(nnn.ReLU())
    :add(nnn.Linear(50, 10))
    :add(nnn.SoftMax())

local classificationCriterion = nnn.MSECriterion()

print("Created multi-branch classifier")

-- Multiple classification branches
local multiBranchInput = {
    torch.randn(100),  -- Branch 1
    torch.randn(100),  -- Branch 2
    torch.randn(100)   -- Branch 3
}

local classificationOutput = classifier:forward(multiBranchInput)
print("Classified 3 branches independently")
print("Each branch produces probability distribution over 10 classes")

-- Example 12: Comparing nn vs nnn
print("\n12. Comparison: nn vs nnn")
print("-------------------------")

-- nn.Linear (standard)
local nnLinear = nn.Linear(10, 5)
local nnInput = torch.randn(3, 10)
local nnOutput = nnLinear:forward(nnInput)
print(string.format("nn.Linear: Input [3x10] -> Output [%dx%d]", 
    nnOutput:size(1), nnOutput:size(2)))

-- nnn.Linear (nested)
local nnnLinear = nnn.Linear(10, 5)
local nnnInput = {torch.randn(2, 10), torch.randn(4, 10)}
local nnnOutput = nnnLinear:forward(nnnInput)
print(string.format("nnn.Linear: Input {[2x10], [4x10]} -> Output {[%dx%d], [%dx%d]}",
    nnnOutput[1]:size(1), nnnOutput[1]:size(2),
    nnnOutput[2]:size(1), nnnOutput[2]:size(2)))

print("\nKey difference: nnn.* preserves nested structure!")

print("\n=== Examples Complete ===")
print("\nKey Takeaways:")
print("1. nnn.transform() converts any nn module to work with nested tensors")
print("2. nnn.* modules (Linear, ReLU, etc.) work like nn.* but for nested data")
print("3. nnn.Criterion extends criteria to be modal classifiers")
print("4. Output structure always matches input structure")
print("5. Useful for hierarchical data: documents, trees, multi-branch networks")
print("\nSee nnn/README.md for complete API documentation")
