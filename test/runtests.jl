using TestItems
using LinearAlgebra

module QuantumNestedSampling
    asserting() = true
end

include("test_linear-algebra.jl")
include("test_constraints.jl")
include("test_galilean-monte-carlo.jl")
include("test_hamiltonians-states.jl")
include("test_linear-algebra.jl")
include("test_nested-sampling.jl")
include("test_util.jl")