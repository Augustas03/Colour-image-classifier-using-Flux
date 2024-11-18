### A Pluto.jl notebook ###
# v0.19.0

using Markdown
using InteractiveUtils

# ╔═╡ 1234abcd-1234-1234-1234-123456789abc
begin
    using Pkg
    Pkg.activate("..")
    using Plots
    using PlutoUI
    using Images
    using Statistics
    using Flux
    using MLDatasets
    using CUDA
    using DataFrames
end

# ╔═╡ 2345bcde-2345-2345-2345-23456789abcd
md"""
# Training a Colour Image Classifier using Flux

This notebook demonstrates the training process for our CIFAR10 image classifier. We'll go through:

1. Loading and preparing the dataset
2. Creating the model
3. Training the classifier
4. Evaluating performance
5. Visualizing results
"""

# ╔═╡ 3456cdef-3456-3456-3456-3456789abcde
md"""
## Setup and Configuration
First, let's set up our environment and define some constants
"""

# ╔═╡ 4567defg-4567-4567-4567-456789abcdef
begin
    # Constants
    const BATCH_SIZE = 1000
    const NUM_EPOCHS = 100
    const LEARNING_RATE = 0.01
    const CLASSES = ["airplane", "automobile", "bird", "cat",
                    "deer", "dog", "frog", "horse", "ship", "truck"]
    
    # Enable GPU if available
    has_cuda = CUDA.functional()
    device = has_cuda ? gpu : cpu
    
    md"Environment configured. Using $(has_cuda ? "GPU" : "CPU") for computation."
end

# ╔═╡ 5678efgh-5678-5678-5678-56789abcdefg
md"""
## Data Loading and Preprocessing
Let's load the CIFAR10 dataset and prepare it for training
"""

# ╔═╡ 6789fghi-6789-6789-6789-6789abcdefgh
begin
    # Load training data
    ENV["DATADEPS_ALWAYS_ACCEPT"] = true
    train_x, train_y = CIFAR10(split=:train)[:]
    train_labels = Flux.onehotbatch(train_y, 0:9)
    
    # Split into training and validation
    n_train = size(train_x, 4)
    split_idx = floor(Int, 0.98 * n_train)
    
    # Training data
    train_x_data = train_x[:, :, :, 1:split_idx]
    train_y_data = train_labels[:, 1:split_idx]
    
    # Validation data
    validate_x = train_x[:, :, :, (split_idx+1):end]
    validate_y = train_labels[:, (split_idx+1):end]
    
    # Create batches
    train_data = [(train_x_data[:, :, :, i:(i+BATCH_SIZE-1)], 
                   train_y_data[:, i:(i+BATCH_SIZE-1)])
                  for i in 1:BATCH_SIZE:split_idx] |> device
    
    validate_data = [(validate_x[:, :, :, i:(i+BATCH_SIZE-1)], 
                     validate_y[:, i:(i+BATCH_SIZE-1)])
                    for i in 1:BATCH_SIZE:size(validate_x, 4)] |> device
    
    md"Dataset loaded and split into $(length(train_data)) training batches and $(length(validate_data)) validation batches"
end

# ╔═╡ 7890ghij-7890-7890-7890-7890abcdefgh
md"""
## Model Definition
Define our Convolutional Neural Network architecture
"""

# ╔═╡ 8901hijk-8901-8901-8901-8901abcdefgh
begin
    model = Chain(
        Conv((5,5), 3=>16, pad=SamePad(), relu),
        Conv((5,5), 16=>16, pad=SamePad(), relu),
        MaxPool((2,2)),
        Conv((5,5), 16=>8, pad=SamePad(), relu),
        MaxPool((2,2)),
        Flux.flatten,
        Dense(512, 256),
        Dense(256, 128),
        Dense(128, 10),
        softmax
    ) |> device
    
    md"Model created with $(sum(length, Flux.params(model))) parameters"
end

# ╔═╡ 9012ijkl-9012-9012-9012-9012abcdefgh
md"""
## Training Functions
Define the loss function and accuracy metrics
"""

# ╔═╡ 0123jklm-0123-0123-0123-0123abcdefgh
begin
    # Loss function
    loss(x, y) = Flux.crossentropy(model(x), y)
    
    # Accuracy calculation
    function accuracy(x, y)
        ŷ = model(x)
        return mean(Flux.onecold(ŷ) .== Flux.onecold(y))
    end
    
    # Optimizer
    opt = Momentum(LEARNING_RATE)
end

# ╔═╡ 1234klmn-1234-1234-1234-1234abcdefgh
md"""
## Training Loop
Train the model and track performance
"""

# ╔═╡ 2345lmno-2345-2345-2345-2345abcdefgh
begin
    # Training button
    @bind start_training Button("Start Training")
end

# ╔═╡ 3456mnop-3456-3456-3456-3456abcdefgh
begin
    start_training
    
    accuracy_history = Float32[]
    loss_history = Float32[]
    
    # Training loop
    for epoch in 1:NUM_EPOCHS
        # Training
        for d in train_data
            gradients = gradient(Flux.params(model)) do
                l = loss(d...)
                push!(loss_history, l)
                return l
            end
            Flux.update!(opt, Flux.params(model), gradients)
        end
        
        # Validation
        val_accuracy = accuracy(validate_data[1]...)
        push!(accuracy_history, val_accuracy)
        
        # Progress update every 10 epochs
        if epoch % 10 == 0
            println("Epoch $epoch: Validation accuracy = $(round(val_accuracy, digits=3))")
        end
    end
    
    md"Training completed with final validation accuracy: $(round(accuracy_history[end], digits=3))"
end

# ╔═╡ 4567nopq-4567-4567-4567-4567abcdefgh
md"""
## Training Visualization
Plot the training progress
"""

# ╔═╡ 5678opqr-5678-5678-5678-5678abcdefgh
begin
    # Create subplot layout
    p1 = plot(accuracy_history,
             ylabel="Accuracy",
             xlabel="Epoch",
             title="Training Accuracy",
             legend=false)
    
    p2 = plot(loss_history[1:100:end],  # Sample loss to reduce noise
             ylabel="Loss",
             xlabel="Iteration/100",
             title="Training Loss",
             legend=false)
    
    plot(p1, p2, layout=(2,1), size=(600,800))
end

# ╔═╡ 6789pqrs-6789-6789-6789-6789abcdefgh
md"""
## Model Evaluation
Test the model on some random samples
"""

# ╔═╡ 7890qrst-7890-7890-7890-7890abcdefgh
begin
    # Create a slider for viewing different test images
    @bind test_idx Slider(1:10, show_value=true)
end

# ╔═╡ 8901rstu-8901-8901-8901-8901abcdefgh
begin
    # Load test data
    test_x, test_y = CIFAR10(split=:test)[:]
    
    # Select random indices for visualization
    rand_indices = rand(1:10000, 10)
    
    # Get current test image
    current_idx = rand_indices[test_idx]
    test_image = test_x[:,:,:,current_idx]
    true_label = test_y[current_idx] + 1
    
    # Make prediction
    pred = model(reshape(test_image, size(test_image)..., 1) |> device) |> cpu
    pred_label = Flux.onecold(pred, 1:10)
    
    # Visualize
    p1 = plot(colorview(RGB, permutedims(test_image, (3,2,1))),
             title="True: $(CLASSES[true_label])\nPred: $(CLASSES[pred_label])",
             axis=false)
    
    p2 = bar(CLASSES, vec(pred),
            title="Class Probabilities",
            xrotation=45,
            ylabel="Probability")
    
    plot(p1, p2, layout=(2,1), size=(600,800))
end

# ╔═╡ 9012stuv-9012-9012-9012-9012abcdefgh
md"""
## Class-wise Performance
Calculate and display performance metrics for each class
"""

# ╔═╡ 0123tuvw-0123-0123-0123-0123abcdefgh
begin
    # Evaluate on test set
    test_data = [(test_x[:,:,:,i:(i+BATCH_SIZE-1)], 
                  Flux.onehotbatch(test_y[i:(i+BATCH_SIZE-1)], 0:9))
                 for i in 1:BATCH_SIZE:10000] |> device
    
    # Calculate class-wise accuracy
    class_correct = zeros(10)
    class_total = zeros(10)
    
    for (x, y) in test_data
        predictions = model(x)
        for i in 1:size(predictions, 2)
            pred_class = Flux.onecold(predictions[:,i])
            true_class = Flux.onecold(y[:,i])
            if pred_class == true_class
                class_correct[pred_class] += 1
            end
            class_total[true_class] += 1
        end
    end
    
    # Create performance DataFrame
    performance_df = DataFrame(
        Class = CLASSES,
        Accuracy = round.(class_correct ./ class_total, digits=3)
    )
    
    md"### Class-wise Accuracy"
end

# ╔═╡ 1234uvwx-1234-1234-1234-1234abcdefgh
performance_df

# ╔═╡ 2345vwxy-2345-2345-2345-2345abcdefgh
md"""
## Save Model
Option to save the trained model
"""

# ╔═╡ 3456wxyz-3456-3456-3456-3456abcdefgh
begin
    @bind save_model Button("Save Model")
    
    if save_model
        using BSON: @save
        model_cpu = model |> cpu
        @save "cifar10_model.bson" model_cpu
        md"Model saved to 'cifar10_model.bson'"
    end
end

# ╔═╡ 4567xyza-4567-4567-4567-4567abcdefgh
md"""
## Conclusion

Our model achieved:
- Final validation accuracy: $(round(accuracy_history[end], digits=3))
- Best performing class: $(CLASSES[argmax(performance_df.Accuracy)])
- Worst performing class: $(CLASSES[argmin(performance_df.Accuracy)])
"""