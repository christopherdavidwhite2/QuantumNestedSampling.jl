using TestItems
using LinearAlgebra
using Revise

module QuantumNestedSampling
    asserting() = true
end

include("test_linear-algebra.jl")
include("test_constraints.jl")
include("test_galilean-monte-carlo.jl")