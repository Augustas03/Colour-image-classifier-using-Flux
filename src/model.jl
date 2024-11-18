"""
Neural network model architecture definition
"""

"""
    create_model()

Create and initialize the CNN model architecture.
"""
function create_model()
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
    )
    
    if CUDA.functional()
        model = gpu(model)
    end
    
    return model
end