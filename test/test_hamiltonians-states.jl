
@testitem "sample_haar" begin
    dim = 10
    for r = 1:10
        u = sample_haar(dim)
        @test u'*u ≈ 1
        @test u isa Array{ComplexF64}
    end
end

@testitem "sample_haar_perp" begin
    dim = 10
    for r = 1:10
        u = sample_haar(dim)
        v = sample_haar_perp(u)
        @test v'*v ≈ 1
        @test abs(u'*v) < 1e-10
        @test v isa Array{ComplexF64}
    end
end