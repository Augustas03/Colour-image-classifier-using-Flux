"""
Utility functions for model evaluation
"""

const CLASSES = ["airplane", "automobile", "bird", "cat",
                "deer", "dog", "frog", "horse", "ship", "truck"]

"""
    accuracy(model, x, y)

Calculate the accuracy of model predictions.
"""
function accuracy(model, x, y)
    ŷ = model(x)
    return mean(Flux.onecold(ŷ) .== Flux.onecold(y))
end

"""
    evaluate_model(model, test_data)

Evaluate model performance on test data.
"""
function evaluate_model(model, test_data)
    class_correct = zeros(10)
    class_total = zeros(10)
    
    for (x, y) in test_data
        predictions = model(x)
        for i in 1:size(predictions, 2)
            pred_class = Flux.onecold(predictions[:, i])
            true_class = Flux.onecold(y[:, i])
            if pred_class == true_class
                class_correct[pred_class] += 1
            end
            class_total[true_class] += 1
        end
    end
    
    accuracies = class_correct ./ class_total
    return DataFrame(
        accuracy = round.(accuracies, digits=3),
        class = CLASSES
    )
end