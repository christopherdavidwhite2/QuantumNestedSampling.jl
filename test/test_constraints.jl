
@testmodule TestModule begin
using LinearAlgebra
using Test
using QuantumNestedSampling
function test_constraint_gradient(C :: Constraint, dim)
    dt = 1e-5

    u = sample_haar(dim)
    v = sample_haar(dim)
    project_out!(u,v)
    v /= norm(v)

    @assert norm(u) ≈ 1
    @assert abs(u'*v) < 1e-10

    @show 
    Δ = ( C.constraint_function(u + dt*v/2) - C.constraint_function(u - dt * v/2) ) / dt
    grad = Array{ComplexF64}(undef, dim)
    C.gradient_function(grad,u)

    discrepancyP = Δ - apply_cx_gradient(grad,v) < 100 * dt^2

    vRI = Array{Float64}(undef,2*dim)
    uRI = Array{Float64}(undef,2*dim)
    v = cx_gradient_householder_reflection!(grad,v,vRI,uRI)
    discrepancyM = Δ + apply_cx_gradient(grad,v) < 100 * dt^2
    return discrepancyP, discrepancyM
end
end

@testitem "fast_adjoint_mul" begin
    using LinearAlgebra

    N = 4
    Nr = 10
    for T = [Float64, ComplexF64]
        A = randn(T,N,N)
        c = randn(T,N)
        b = randn(T,N)
        α = randn(T)
        β = randn(T)

        answer = α*A'*b + β*c
        fast_adjoint_mul!(c,A,b,α,β)
        @test norm(answer - c) < 1e-10


        answer = A'*b 
        fast_adjoint_mul!(c,A,b)
        @test norm(answer - c) < 1e-10
    end
end



@testitem "energy gradient via constraint structure" setup=[TestModule] begin
    using LinearAlgebra

    dim = 10

    H = GOE(dim)
    C = energy_constraint(H)
    for r = 1:10
        P,M = TestModule.test_constraint_gradient(C,dim)
        @test P
        @test M
    end

    for r = 1:10
        C = energy_constraint(H)
        P,M = TestModule.test_constraint_gradient(C,dim)
        @test P
        @test M
    end

 end

@testitem "fourth moment via constraints structure" setup=[TestModule] begin
    using LinearAlgebra

    dim = 10

    H = GOE(dim)
    C = quart_constraint(H)
    for r = 1:1
        P,M = TestModule.test_constraint_gradient(C,dim)
        @test P 
        @test M 
    end
end

@testitem "effective" begin
    using LinearAlgebra

    dim = 10
    for r = 1:10
        H = GOE(10)
        u = sample_haar(dim)
        v = sample_haar(dim)
        project_out!(u,v)
        v /= norm(v)

        Heff = effective(H,u,v)

        φ = randn(ComplexF64,2)
        ψ = φ[1]*u + φ[2]*v
        @test abs(φ'*Heff*φ - ψ'*H*ψ) < 1e-10
    end
end

@testitem "effective" begin
    using LinearAlgebra

    dim = 10
    s = Array{Float64}(undef, dim)
    for r = 1:10
        H = GOE(10)
        u = sample_haar(Float64,dim)
        v = sample_haar(Float64,dim)
        project_out!(u,v)
        v /= norm(v)

        Heff = fast_effective(H,u,v,s)

        φ = randn(ComplexF64,2)
        ψ = φ[1]*u + φ[2]*v
        @test abs(φ'*Heff*φ - ψ'*H*ψ) < 1e-10
    end
end

@testitem "quart vs quart expanded" begin
    dim = 10
    for r = 1:10
        H = GOE(dim)
        u = sample_haar(dim)
        @test abs( quart(H,u) - quart_expanded((H,H^2, H^3,H^4),u)) < 1e-10
    end
end

@testitem "fast_quart" begin
    dim = 10
    for r = 1:10
        H = GOE(dim)
        u = sample_haar(Float64,dim)
        @test abs( quart(H,u) - fast_quart(H)(u) ) < 1e-10
    end
end
