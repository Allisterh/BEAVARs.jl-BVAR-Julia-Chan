using BEAVARs
using Test
using TimeSeries
using Parameters
using LinearAlgebra, SparseArrays, Statistics
@testset "BEAVARs.jl" begin
    # Write your tests here.
    @test 1 == 1

    @testset "HBMatrix lower block-Toeplitz indexing" begin
        B1 = [1 2; 3 4]
        B2 = [5 6; 7 8]
        B3 = [9 10; 11 12]
        mBI = hcat(B1, B2, B3)
        HB = BEAVARs.HBMatrix(mBI, 4)

        expected = zeros(Int, 8, 8)
        expected[1:2, 1:2] = B1
        expected[3:4, 1:2] = B2
        expected[3:4, 3:4] = B1
        expected[5:6, 1:2] = B3
        expected[5:6, 3:4] = B2
        expected[5:6, 5:6] = B1
        expected[7:8, 3:4] = B3
        expected[7:8, 5:6] = B2
        expected[7:8, 7:8] = B1

        @test Matrix(HB) == expected
        @test HB[1:2, 1:2] == B1
        @test HB[3:4, 1:2] == B2
        @test HB[5:6, 1:2] == B3
        @test HB[1:2, 3:4] == zeros(Int, 2, 2)

        selector = sparse([1, 6, 7], [1, 2, 3], [1.0, 1.0, 1.0], 8, 3)
        product = zeros(Float64, 8, 3)
        @test mul!(product, HB, selector) ≈ Matrix(HB) * Matrix(selector)

        weighted_selector = sparse([1, 6, 7], [1, 2, 3], [2.0, -1.0, 0.5], 8, 3)
        initial = fill(0.25, 8, 3)
        product .= initial
        @test mul!(product, HB, weighted_selector, 1.5, -2.0) ≈
              1.5 .* Matrix(HB) * Matrix(weighted_selector) .- 2.0 .* initial
    end

    # Test for whether the XSurFormMatrix structure gives the correct result
    @testset "testset_sur.jl" begin
        include("testset_sur.jl")    
    end



    # @testset "testset_CPZ2023.jl" begin
    #     include("testset_CPZ2023.jl")    
    # end
end
