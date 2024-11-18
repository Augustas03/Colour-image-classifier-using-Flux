"""
Functions for loading and preprocessing CIFAR10 dataset
"""

"""
    load_cifar10_data(; split_ratio=0.98)

Load and preprocess the CIFAR10 dataset, splitting into training and validation sets.
"""
function load_cifar10_data(; split_ratio=0.98)
    ENV["DATADEPS_ALWAYS_ACCEPT"] = true
    
    # Load CIFAR10 training data
    train_x, train_y = CIFAR10(split=:train)[:]
    train_labels = Flux.onehotbatch(train_y, 0:9)
    
    # Calculate split indices
    n_train = size(train_x, 4)
    split_idx = floor(Int, split_ratio * n_train)
    
    # Split data
    train_x_data = train_x[:, :, :, 1:split_idx]
    train_y_data = train_labels[:, 1:split_idx]
    validate_x = train_x[:, :, :, (split_idx+1):end]
    validate_y = train_labels[:, (split_idx+1):end]
    
    # Create batches
    batch_size = 1000
    train_data = [(train_x_data[:, :, :, i:(i+batch_size-1)], 
                   train_y_data[:, i:(i+batch_size-1)])
                  for i in 1:batch_size:split_idx]
    
    validate_data = [(validate_x[:, :, :, i:(i+batch_size-1)], 
                     validate_y[:, i:(i+batch_size-1)])
                    for i in 1:batch_size:size(validate_x, 4)]
    
    if CUDA.functional()
        train_data = gpu.(train_data)
        validate_data = gpu.(validate_data)
    end
    
    return train_data, validate_data
end

"""
    load_test_data()

Load and prepare CIFAR10 test dataset.
"""
function load_test_data()
    test_x, test_y = CIFAR10(split=:test)[:]
    test_x = reshape(test_x, 32, 32, 3, :)
    test_labels = Flux.onehotbatch(test_y, 0:9)
    
    batch_size = 1000
    test_data = [(test_x[:, :, :, i:(i+batch_size-1)], 
                  test_labels[:, i:(i+batch_size-1)])
                 for i in 1:batch_size:10000]
    
    if CUDA.functional()
        test_data = gpu.(test_data)
    end
    
    return test_data
end