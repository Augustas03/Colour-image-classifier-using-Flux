"""
Visualization functions for model predictions and training progress
"""

"""
    visualize_prediction(model, image, true_label)

Create a visualization of model prediction for a single image.
"""
function visualize_prediction(model, image, true_label)
    # Prepare image for model
    img_tensor = reshape(image, size(image)..., 1)
    if CUDA.functional()
        img_tensor = gpu(img_tensor)
    end
    
    # Get prediction
    prediction = cpu(model(img_tensor))
    pred_label = Flux.onecold(prediction, 0:9)
    
    # Create visualization
    p = plot(layout=(2,1), size=(600,800))
    
    # Plot image
    plot!(p[1], colorview(RGB, permutedims(image, (3,2,1))),
          title="Prediction: $(CLASSES[pred_label])\nTrue: $(CLASSES[true_label])",
          axis=false)
    
    # Plot probabilities
    bar!(p[2], CLASSES, vec(prediction),
         title="Class Probabilities",
         xrotation=45,
         ylabel="Probability")
    
    return p
end

"""
    plot_training_history(history)

Plot the training history showing accuracy over epochs.
"""
function plot_training_history(history)
    plot(history,
         xlabel="Epoch",
         ylabel="Accuracy",
         title="Training History",
         legend=false)
end