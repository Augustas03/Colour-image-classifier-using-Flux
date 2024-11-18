# Colour Image Classifier using Flux

A deep learning image classifier built with Flux.jl for the CIFAR-10 dataset. This project implements a Convolutional Neural Network (CNN) to classify color images into 10 different categories.

## Features

- Convolutional Neural Network implementation using Flux.jl
- CUDA GPU acceleration support
- Interactive training notebook
- Visualization tools for model predictions
- Comprehensive test suite

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Colour-image-classifier-using-Flux.git
cd Colour-image-classifier-using-Flux
```

2. Start Julia and activate the environment:
```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

## Project Structure

```
├── src/                  # Source code
│   ├── data.jl          # Data loading and preprocessing
│   ├── model.jl         # Neural network architecture
│   ├── training.jl      # Training functionality
│   ├── utils.jl         # Utility functions
│   └── visualization.jl  # Visualization tools
├── notebooks/           # Pluto notebooks
├── test/                # Test suite
└── .vscode/            # VS Code configuration
```

## Usage

### Training the Model

You can train the model either through the Pluto notebook or directly using the API:

```julia
using ColourImageClassifier

# Load data
train_data, val_data = load_cifar10_data()

# Create model
model = create_model()

# Train
history = train_model!(model, train_data, val_data, epochs=100)

# Visualize training progress
plot_training_history(history)
```

### Making Predictions

```julia
# Load test data
test_data = load_test_data()

# Evaluate model
results = evaluate_model(model, test_data)

# Visualize a prediction
image = test_data[1][1][:,:,:,1]
prediction = visualize_prediction(model, image, true_label)
```

## Model Architecture

The CNN architecture consists of:
- Multiple convolutional layers with ReLU activation
- MaxPooling layers for dimensional reduction
- Fully connected layers
- Softmax output layer

## Requirements

- Julia 1.6+
- Flux.jl
- CUDA.jl (optional, for GPU support)
- MLDatasets
- Images.jl
- Plots.jl
- DataFrames.jl

## License

This project is licensed under the MIT License - see the LICENSE file for details.