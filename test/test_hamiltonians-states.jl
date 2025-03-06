
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