"""
Model training functionality
"""

"""
    train_model!(model, train_data, validate_data; epochs=100, learning_rate=0.01)

Train the model using the provided training data and validate using validation data.
"""
function train_model!(model, train_data, validate_data; 
                     epochs=100, learning_rate=0.01)
    loss(x, y) = Flux.crossentropy(model(x), y)
    optimizer = Momentum(learning_rate)
    accuracy_history = Float32[]
    
    for epoch in 1:epochs
        # Training loop
        for d in train_data
            gradients = gradient(Flux.params(model)) do
                loss(d...)
            end
            Flux.update!(optimizer, Flux.params(model), gradients)
        end
        
        # Validation accuracy
        acc = accuracy(model, validate_data[1]...)
        push!(accuracy_history, acc)
        println("Epoch $epoch: Validation accuracy = $acc")
    end
    
    return accuracy_history
end