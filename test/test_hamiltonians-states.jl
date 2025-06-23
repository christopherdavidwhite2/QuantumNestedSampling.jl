
@testitem "sample_haar" begin
    dim = 10
    for r = 1:10
        u = sample_haar(dim)
        @test u'*u ≈ 1
        @test u isa Array{ComplexF64}
    end
end

@testitem "sample_haar_perp" begin
    using LinearAlgebra
    for dim = 2 .^ 1:10
        for r = 1:10
            u = sample_haar(dim)
            v = sample_haar_perp(u)
            @test v'*v ≈ 1
            @test norm(v) ≈ 1
            @test abs(u'*v) < 1e-10
            @test v isa Array{ComplexF64}
        end
    end
end

@testitem "sample_haar_perp!" begin
    using LinearAlgebra
    for dim = 2 .^ 1:10
        v = zeros(ComplexF64, dim)
        for r = 1:10
            u = sample_haar(dim)
            sample_haar_perp!(u,v)
            @test v'*v ≈ 1
            @test norm(v) ≈ 1
            @test abs(u'*v) < 1e-10
            @test v isa Array{ComplexF64}
        end
    end
end

@testitem "paramagnet" begin
    for L = 1:17
        @test QuantumNestedSampling.paramagnet_embed(2^L) == paramagnet(2^L)
    end
end

@testitem "ising_1d" begin
    using Random
    using LinearAlgebra
    for L = 2:4
        @test ( QuantumNestedSampling.ising_1d_embed(2^L, hx=0, hz=0, pbc=false)
                                         == ising_1d(2^L, hx=0, hz=0, pbc=false) )

        @test ( QuantumNestedSampling.ising_1d_embed(2^L, hx=0, hz=0, pbc=true)
                                         == ising_1d(2^L, hx=0, hz=0, pbc=true) )

        @test ( QuantumNestedSampling.ising_1d_embed(2^L, hx=1, hz=0, pbc=false)
                                         == ising_1d(2^L, hx=1, hz=0, pbc=false) )

        @test ( QuantumNestedSampling.ising_1d_embed(2^L, hx=0, hz=1, pbc=false)
                                         == ising_1d(2^L, hx=0, hz=1, pbc=false) )


        @test ( QuantumNestedSampling.ising_1d_embed(2^L, hx=1, hz=1, pbc=false)
                                         == ising_1d(2^L, hx=1, hz=1, pbc=false) )

        for r = 1:10
            hx = randn()
            hz = randn()
            pbc = rand(Bool)

            Δ = ( QuantumNestedSampling.ising_1d_embed(2^L;hx, hz, pbc) - ising_1d(2^L;hx, hz, pbc) )
            @test norm(Δ) < 1e-10
        end
    end
end

@testitem "heisenberg_1d" begin
    using LinearAlgebra
    using Random
    L = 2
    hx = zeros(L)
    hy = zeros(L)
    hz = zeros(L)
    A = QuantumNestedSampling.heisenberg_1d_embed(2^L, hx,hy,hz)
    B =                             heisenberg_1d(2^L, hx,hy,hz)
    @test (A == B)

    hx = ones(L)
    hy = zeros(L)
    hz = zeros(L)
    A = QuantumNestedSampling.heisenberg_1d_embed(2^L, hx,hy,hz)
    B =                             heisenberg_1d(2^L, hx,hy,hz)
    #display(A)
    #display(B)
    @test (A == B)

    hx = zeros(L)
    hy = ones(L)
    hz = zeros(L)
    A = QuantumNestedSampling.heisenberg_1d_embed(2^L, hx,hy,hz)
    B =                             heisenberg_1d(2^L, hx,hy,hz)
    #display(A)
    #display(B)
    @test (A == B)

    hx = zeros(L)
    hy = zeros(L)
    hz = ones(L)
    A = QuantumNestedSampling.heisenberg_1d_embed(2^L, hx,hy,hz)
    B =                             heisenberg_1d(2^L, hx,hy,hz)
    #display(A)
    #display(B)
    @test (A == B)

    for L = 2:4
        for r = 1:10
            hx = randn(L)
            hy = randn(L)
            hz = randn(L)
            A = QuantumNestedSampling.heisenberg_1d_embed(2^L, hx,hy,hz)
            B =                             heisenberg_1d(2^L, hx,hy,hz)
            #display(A)
            #display(B)
            @test norm(A - B) < 1e-10
        end
    end
end

@testitem "heisenberg_1d_gs" begin
    using LinearAlgebra

    for L = 2:4
        H = 2^L |> heisenberg_1d
        H += UniformScaling(L) # ground state energy
        gs = heisenberg_1d_gs(2^L)
        @test norm(gs'*gs - LinearAlgebra.I(L+1)) < 1e-10
        gs = 2^L |> heisenberg_1d_gs
        @test norm(H*gs) < 1e-10
    end

end