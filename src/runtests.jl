using Test
using ColourImageClassifier
using Flux
using CUDA

@testset "ColourImageClassifier.jl" begin
    @testset "Data Loading" begin
        train_data, val_data = load_cifar10_data(split_ratio=0.98)
        @test length(train_data) > 0
        @test length(val_data) > 0
        
        x, y = first(train_data)
        @test size(x, 1) == size(x, 2) == 32
        @test size(x, 3) == 3
    end
    
    @testset "Model Architecture" begin
        model = create_model()
        @test model isa Chain
        
        x = randn(Float32, 32, 32, 3, 1)
        if CUDA.functional()
            x = gpu(x)
        end
        y = model(x)
        @test size(y) == (10, 1)
    end
    
    @testset "Utilities" begin
        model = create_model()
        x = randn(Float32, 32, 32, 3, 1)
        y = Flux.onehotbatch([1], 1:10)
        
        if CUDA.functional()
            x, y = gpu(x), gpu(y)
        end
        
        acc = accuracy(model, x, y)
        @test 0 ≤ acc ≤ 1
    end
end