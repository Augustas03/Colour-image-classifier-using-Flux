module ColourImageClassifier

using Flux
using MLDatasets
using Images
using DataFrames
using CUDA
using Statistics
using Plots

# Include all module components
include("data.jl")
include("model.jl")
include("training.jl")
include("utils.jl")
include("visualization.jl")

# Export public functions
export load_cifar10_data,
       create_model,
       train_model!,
       evaluate_model,
       visualize_prediction,
       accuracy,
       plot_training_history

end