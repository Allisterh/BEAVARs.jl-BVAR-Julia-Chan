using BEAVARs
using Test
using TimeSeries
using Parameters
using LinearAlgebra, Statistics
@testset "BEAVARs.jl" begin
    # Write your tests here.
    @test 1 == 1
    @test true

    @testset "testset_CPZ2023.jl" begin
        include("testset_CPZ2023.jl")    
    end
end



